"""Debate pipeline: model download/load -> research -> debate -> judging.

Runs in a worker thread (llama.cpp calls are blocking) and reports everything
through an `emit(type, **data)` callback supplied by the manager in main.py.
"""
from __future__ import annotations

import difflib
import re
import threading
from dataclasses import dataclass, field

from . import embeddings, judging, prep, research
from .config import CASE_PREP, FAKE_LLM, RAG_QUERIES, USE_EMBEDDINGS
from .llm import load_llm
from .models_registry import StopRequested, ensure_model_file, get_model
from .rag import ResearchIndex

DEFAULT_PERSONALITY = "A confident, articulate professional debater."

PHASE_TASKS = {
    "opening": (
        "Deliver your OPENING STATEMENT: begin with one unambiguous sentence "
        "stating your position on the motion, then frame the motion on your "
        "terms and present your two or three strongest arguments, each "
        "anchored in a specific piece of evidence — quote your best material. "
        "Under 280 words."
    ),
    "rebuttal": (
        "Deliver your REBUTTAL: single out your opponent's strongest point "
        "and dismantle it with specific evidence, defend your own case "
        "against their attacks, and advance your case with fresh evidence "
        "you have not yet used — do not restate your opening. Engage with "
        "what was actually said. Under 240 words."
    ),
    "closing": (
        "Deliver your CLOSING STATEMENT: crystallize the key clashes of the "
        "debate, explain why your side has won them — pointing to the "
        "decisive evidence — and end memorably. No new arguments. Under 220 "
        "words."
    ),
}


@dataclass
class DebateConfig:
    topic: str
    model_id: str
    pro_personality: str
    con_personality: str
    rounds: int = 2
    # User-uploaded research documents (research.Doc) and whether to add
    # Wikipedia/web research on top of them.
    materials: list = field(default_factory=list)
    use_web_research: bool = True


def build_schedule(rounds: int) -> list[dict]:
    """Classic format: PRO opens, alternating rebuttals, PRO gets the final
    word in closings."""
    schedule = [
        {"speaker": "pro", "phase": "opening", "round": 0, "label": "Opening · FOR"},
        {"speaker": "con", "phase": "opening", "round": 0, "label": "Opening · AGAINST"},
    ]
    for r in range(1, rounds + 1):
        schedule.append({"speaker": "pro", "phase": "rebuttal", "round": r,
                         "label": f"Rebuttal {r} · FOR"})
        schedule.append({"speaker": "con", "phase": "rebuttal", "round": r,
                         "label": f"Rebuttal {r} · AGAINST"})
    schedule += [
        {"speaker": "con", "phase": "closing", "round": 0, "label": "Closing · AGAINST"},
        {"speaker": "pro", "phase": "closing", "round": 0, "label": "Closing · FOR"},
    ]
    return schedule


def _stance(side: str, position: str = "") -> str:
    """Unambiguous statement of what this side must argue. Spelled out for
    both statement- and question-phrased motions, because a bare 'argue FOR
    the motion' is easily misread (especially by small models) as 'argue for
    the thing the motion is about'. `position` is the side's burden restated
    as a plain sentence (prep.derive_positions) — repeating it here saves
    the model from re-deriving double negations, which flips sides."""
    if side == "pro":
        text = (
            "You are the PROPOSITION (PRO). You AGREE with the motion and "
            "argue that it is TRUE. If the motion is phrased as a question, "
            "your answer is emphatically YES."
        )
    else:
        text = (
            "You are the OPPOSITION (CON). You DISAGREE with the motion and "
            "argue that it is FALSE. If the motion is phrased as a question, "
            "your answer is emphatically NO."
        )
    if position:
        text += (" In plain terms, every argument you make must convince "
                 f"the judge that: {position}")
    return text


def _opponent_last(transcript: list[dict], side: str) -> str:
    return next((t["text"] for t in reversed(transcript)
                 if t["speaker"] != side), "")


def _research_queries(llm, cfg: DebateConfig, turn: dict,
                      transcript: list[dict], position: str) -> list[str]:
    """Let the debater write its own search queries for this turn — far
    better recall over a large library than one fixed topic query."""
    side = turn["speaker"]
    opponent = _opponent_last(transcript, side)
    fallback = [f"{cfg.topic} {opponent[:400]}" if opponent else cfg.topic]
    if FAKE_LLM:
        return fallback
    prompt = (
        f'Motion under debate: "{cfg.topic}"\n'
        f"You argue {'FOR' if side == 'pro' else 'AGAINST'} the motion — "
        f"you must prove that: {position}\n"
        f"You are preparing your {turn['phase']}.\n"
    )
    if opponent:
        prompt += f"Your opponent's latest speech:\n{opponent[:1500]}\n\n"
    prompt += (
        f"Write {RAG_QUERIES} different short search queries (3-8 words "
        "each) to find material in the research library that best supports "
        "your side"
        + (" and refutes your opponent's latest points" if opponent else "")
        + ". One query per line. Output only the queries."
    )
    try:
        raw = llm.chat(
            [{"role": "system", "content":
              "You write search queries for a debater's research library. "
              "Reply with one short query per line, nothing else."},
             {"role": "user", "content": prompt}],
            max_tokens=120, temperature=0.3)
        queries = []
        for line in raw.splitlines():
            q = re.sub(r"^[\s\-•*\d.)\"']+|[\"']+$", "", line).strip()
            if len(q) >= 3 and q.lower() not in (x.lower() for x in queries):
                queries.append(q[:120])
        queries = queries[:RAG_QUERIES]
    except Exception:
        queries = []
    return queries + fallback  # the fixed query is always the safety net


def _debater_messages(cfg: DebateConfig, turn: dict, transcript: list[dict],
                      excerpts: str, brief: str = "",
                      position: str = "") -> list[dict]:
    side = turn["speaker"]
    personality = (cfg.pro_personality if side == "pro" else cfg.con_personality) \
        .strip() or DEFAULT_PERSONALITY

    system = (
        "You are a world-class competitive debater and subject-matter expert "
        "taking part in a formal debate.\n"
        f'The motion under debate: "{cfg.topic}"\n'
        f"{_stance(side, position)}\n"
        "Your opponent argues the exact opposite. Attack their arguments "
        "directly; never agree with their side, never switch sides, never "
        "concede the debate.\n"
        f"Your personality and speaking style: {personality}\n"
        "Argue from concrete evidence, never generalities: anchor every "
        "major argument in a specific fact, moment or short verbatim "
        "quotation from your evidence brief or the research excerpts, citing "
        "the source inline by the actual title shown in brackets. One exact "
        "quotation deployed at the right moment beats a paragraph of "
        "abstraction. Where research is thin, draw on your own expertise.\n"
        "The research may lean toward one side. If it favours your opponent, "
        "do not adopt its conclusions — use it to anticipate and rebut their "
        "case, and reframe its facts to serve your side.\n"
        "Never repeat or closely paraphrase sentences from earlier speeches "
        "— every speech must be made of fresh material.\n"
        "Refer to the other debater only as 'my opponent'. Never use "
        "placeholders like [Your Name] or [Source Title].\n"
        "Speak in flowing spoken prose — no markdown, no headings, no bullet "
        "lists, no stage directions."
    )

    # Present the debate as a real conversation: the opponent's speeches are
    # incoming ("user") messages, this debater's own speeches are its own
    # previous ("assistant") replies. Models respond to conversation far more
    # reliably than to a transcript pasted into a single prompt — this is
    # what makes them engage instead of parroting the transcript back.
    convo: list[dict] = []

    def add(role: str, content: str):
        if convo and convo[-1]["role"] == role:
            convo[-1]["content"] += "\n\n" + content
        else:
            convo.append({"role": role, "content": content})

    for t in transcript:
        if t["speaker"] == side:
            add("assistant", t["text"])
        else:
            add("user", f"Your opponent delivered their {t['label']}:\n\n{t['text']}")

    task = ""
    if brief:
        task += (
            "YOUR EVIDENCE BRIEF — verbatim quotations you gathered from "
            "the source material while preparing your case. Deploy the "
            "items that genuinely support YOUR claim, quoting them exactly. "
            "If an item actually favours your opponent's claim, do not "
            "build on it — rebut it or reframe it.\n\n" + brief + "\n\n"
        )
    task += (
        f"RESEARCH EXCERPTS you may cite:\n\n{excerpts}\n\n"
        f"It is now your turn ({turn['label']}). {PHASE_TASKS[turn['phase']]}\n"
        f"Remember your side: {_stance(side, position)}"
    )
    add("user", task)

    # Some chat templates behave oddly when the first non-system message is
    # from the assistant; give the debater's own opening a moderator cue.
    if convo[0]["role"] == "assistant":
        convo.insert(0, {"role": "user",
                         "content": "You have the floor for your opening statement."})

    return [{"role": "system", "content": system}, *convo]


def _clean_speech(text: str) -> str:
    """Trim whitespace, drop leading decoration-only lines ('---' etc.) and
    scrub name placeholders small models sometimes emit despite instructions."""
    lines = text.strip().splitlines()
    while lines and re.fullmatch(r"[-–—*_#=\s]*", lines[0]):
        lines.pop(0)
    text = "\n".join(lines).strip()
    return re.sub(r"\[(?:your |my |opponent'?s? ?)?name\]|\[opponent\]",
                  "my opponent", text, flags=re.IGNORECASE)


def _too_similar(text: str, transcript: list[dict], threshold: float = 0.6) -> bool:
    """True if the speech is mostly a copy of an earlier speech."""
    lowered = text.lower()
    return any(
        difflib.SequenceMatcher(None, lowered, t["text"].lower()).ratio() > threshold
        for t in transcript
    )


def _argued_wrong_side(llm, topic: str, side: str, text: str,
                       positions: dict[str, str]) -> bool:
    """Classify which side's claim a speech actually SUPPORTS. Small models
    drift onto the opponent's side mid-debate — and under a negated motion
    a confused speech can assert 'the motion is true' while its substance
    proves the opposite. Judging against the plain-language claims (not the
    speech's own TRUE/FALSE labels) catches both."""
    answer = llm.chat(
        [{"role": "system", "content":
          "You classify debate speeches by the substance of their "
          "arguments, ignoring what the speech asserts about which side it "
          "is on. Reply with a single letter: A or B."},
         {"role": "user", "content":
          f'Motion: "{topic}"\n\n'
          f"Claim A: {positions['pro']}\n"
          f"Claim B: {positions['con']}\n\n"
          f"Speech:\n{text[:2500]}\n\n"
          "Taken as a whole, which claim does the substance of this "
          "speech's evidence and reasoning actually support? Reply with "
          "one letter, A or B."}],
        max_tokens=4, temperature=0.0).strip().upper()
    wrong = "B" if side == "pro" else "A"
    right = "A" if side == "pro" else "B"
    # Only regenerate on an unambiguous wrong-side verdict.
    return wrong in answer and right not in answer


def run_pipeline(cfg: DebateConfig, emit, stop: threading.Event) -> None:
    try:
        model = get_model(cfg.model_id)

        emit("phase", phase="model", message=f"Preparing model: {model['name']}")
        def dl_progress(done, total, filename):
            emit("download_progress", filename=filename, done=done, total=total,
                 pct=round(100 * done / total, 1) if total else 0)
        if FAKE_LLM:
            path, how = "fake", "FAKE_LLM mode, no model file needed"
        else:
            path, how = ensure_model_file(model, dl_progress, stop)
        emit("status", message=f"Model ready — {how}")

        emit("status", message="Loading model into memory…")
        llm = load_llm(path)
        emit("status", message="Model loaded.")

        emit("phase", phase="research", message="Researching the topic…")
        docs = list(cfg.materials)
        for d in docs:
            emit("research_source", title=d.title, url=d.url,
                 chars=len(d.text), kind="upload")
        if cfg.use_web_research:
            docs += research.gather(cfg.topic, emit, stop)
        else:
            emit("status", message="Web research skipped — using your "
                                   "materials only.")

        embedder = None
        if docs and not FAKE_LLM and USE_EMBEDDINGS:
            emit("status", message="Preparing semantic search "
                                   "(embedding model)…")
            embedder = embeddings.load_embedder(dl_progress, stop)
            if embedder is None:
                emit("status", message="Embedding model unavailable — "
                                       "keyword-only retrieval.")
        index = ResearchIndex(docs, embedder)
        emit("research_done", num_sources=len(docs),
             num_chunks=len(index.chunks), semantic=index.semantic)

        positions = {"pro": "", "con": ""}
        briefs = {"pro": "", "con": ""}
        if not FAKE_LLM:
            emit("phase", phase="prep",
                 message="The debaters prepare their cases…")
            emit("status", message="Clarifying what each side must prove…")
            positions = prep.derive_positions(llm, cfg.topic)
            emit("prep_positions", pro=positions["pro"], con=positions["con"])
            emit("status", message=f"PRO must prove: {positions['pro']}")
            emit("status", message=f"CON must prove: {positions['con']}")
            if index.chunks and CASE_PREP:
                briefs = prep.build_briefs(llm, cfg.topic, positions, index,
                                           emit, stop)

        emit("phase", phase="debate", message="The debate begins.")
        transcript: list[dict] = []
        for turn in build_schedule(cfg.rounds):
            if stop.is_set():
                raise StopRequested()
            # Announce the turn before the (slow) research step so the UI
            # can show who is preparing and, once known, their queries.
            emit("turn_prep", speaker=turn["speaker"], label=turn["label"],
                 queries=[])
            if index.chunks:
                queries = _research_queries(llm, cfg, turn, transcript,
                                            positions[turn["speaker"]])
                if len(queries) > 1:
                    emit("turn_prep", speaker=turn["speaker"],
                         label=turn["label"],
                         queries=[q[:80] for q in queries[:-1]])
                    emit("status", message=f"{turn['label']} — researching: "
                         + " · ".join(f"“{q[:60]}”" for q in queries[:-1]))
                excerpts = index.gather_research(queries)
            else:
                excerpts = index.gather_research([cfg.topic])
            messages = _debater_messages(cfg, turn, transcript, excerpts,
                                         briefs[turn["speaker"]],
                                         positions[turn["speaker"]])
            text = ""
            for attempt in range(2):
                emit("turn_start", **turn)
                text = ""
                for token in llm.chat_stream(messages, max_tokens=600,
                                             temperature=0.8 + 0.2 * attempt):
                    if stop.is_set():
                        raise StopRequested()
                    text += token
                    emit("token", speaker=turn["speaker"], text=token)
                text = _clean_speech(text)
                if FAKE_LLM or attempt == 1:
                    break
                # Two failure modes of small models, each worth one retry:
                # echoing an earlier speech, or drifting onto the wrong side.
                if _too_similar(text, transcript):
                    problem = ("That speech repeated an earlier speech almost "
                               "word for word, which is not allowed. Deliver "
                               "a completely different speech in fresh words, "
                               "engaging directly with your opponent's latest "
                               "points.")
                    emit("status", message=f"{turn['label']} repeated earlier "
                         "material — regenerating.")
                elif _argued_wrong_side(llm, cfg.topic, turn["speaker"], text,
                                        positions):
                    problem = ("That speech argued the WRONG SIDE of the "
                               f"motion. "
                               f"{_stance(turn['speaker'], positions[turn['speaker']])} "
                               "Rewrite your speech so every argument "
                               "supports YOUR side and attacks your "
                               "opponent's.")
                    emit("status", message=f"{turn['label']} argued the wrong "
                         "side — regenerating.")
                else:
                    break
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": problem},
                ]
            entry = {**turn, "text": text}
            transcript.append(entry)
            emit("turn_end", **entry)

        emit("phase", phase="judging", message="The judge deliberates…")
        ballots = []
        n_criteria = len(judging.CRITERIA)
        for judge in judging.JUDGES:
            if stop.is_set():
                raise StopRequested()
            emit("judge_start", judge_id=judge["id"], name=judge["name"])

            scored = 0

            def on_criterion(crit, result, judge=judge):
                nonlocal scored
                scored += 1
                emit("judge_criterion", judge_id=judge["id"],
                     criterion=crit["key"], result=result)
                emit("status", message=f"Ballot: {crit['label']} scored "
                                       f"({scored}/{n_criteria})")
                if stop.is_set():
                    raise StopRequested()

            try:
                ballot = judging.judge_debate(llm, judge, cfg.topic,
                                              transcript, on_criterion,
                                              positions)
            except StopRequested:
                raise
            except RuntimeError as e:
                emit("status", message=str(e))
                zeros = {c["key"]: 0 for c in judging.CRITERIA}
                blanks = {c["key"]: "" for c in judging.CRITERIA}
                ballot = {"scores": {s: dict(zeros) for s in ("pro", "con")},
                          "reasons": {s: dict(blanks) for s in ("pro", "con")},
                          "totals": {"pro": 0, "con": 0}, "winner": "tie",
                          "summary": "Ballot invalid; judge abstained."}
            ballots.append(ballot)
            emit("judge_result", judge_id=judge["id"], name=judge["name"],
                 ballot=ballot)

        verdict = judging.tally(ballots)
        emit("verdict", **verdict)
        emit("phase", phase="done", message="Debate complete.")
    except StopRequested:
        emit("phase", phase="idle", message="Debate cancelled.")
    except Exception as e:
        emit("error", message=f"{type(e).__name__}: {e}")
