"""Debate pipeline: model download/load -> research -> debate -> judging.

Runs in a worker thread (llama.cpp calls are blocking) and reports everything
through an `emit(type, **data)` callback supplied by the manager in main.py.
"""
from __future__ import annotations

import difflib
import re
import threading
from dataclasses import dataclass, field

from . import embeddings, judging, research
from .config import FAKE_LLM, USE_EMBEDDINGS
from .llm import load_llm
from .models_registry import StopRequested, ensure_model_file, get_model
from .rag import ResearchIndex

DEFAULT_PERSONALITY = "A confident, articulate professional debater."

PHASE_TASKS = {
    "opening": (
        "Deliver your OPENING STATEMENT: begin with one unambiguous sentence "
        "stating your position on the motion, then frame the motion on your "
        "terms and present your two or three strongest arguments. Under 280 "
        "words."
    ),
    "rebuttal": (
        "Deliver your REBUTTAL: directly attack your opponent's most recent "
        "points, defend your own case against their attacks, and extend your "
        "strongest argument. Engage with what was actually said. Under 240 words."
    ),
    "closing": (
        "Deliver your CLOSING STATEMENT: crystallize the key clashes of the "
        "debate, explain why your side has won them, and end memorably. No new "
        "arguments. Under 220 words."
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


def _stance(side: str) -> str:
    """Unambiguous statement of what this side must argue. Spelled out for
    both statement- and question-phrased motions, because a bare 'argue FOR
    the motion' is easily misread (especially by small models) as 'argue for
    the thing the motion is about'."""
    if side == "pro":
        return (
            "You are the PROPOSITION (PRO). You AGREE with the motion and "
            "argue that it is TRUE. If the motion is phrased as a question, "
            "your answer is emphatically YES."
        )
    return (
        "You are the OPPOSITION (CON). You DISAGREE with the motion and "
        "argue that it is FALSE. If the motion is phrased as a question, "
        "your answer is emphatically NO."
    )


def _debater_messages(cfg: DebateConfig, turn: dict, transcript: list[dict],
                      index: ResearchIndex) -> list[dict]:
    side = turn["speaker"]
    personality = (cfg.pro_personality if side == "pro" else cfg.con_personality) \
        .strip() or DEFAULT_PERSONALITY

    system = (
        "You are a world-class competitive debater and subject-matter expert "
        "taking part in a formal debate.\n"
        f'The motion under debate: "{cfg.topic}"\n'
        f"{_stance(side)}\n"
        "Your opponent argues the exact opposite. Attack their arguments "
        "directly; never agree with their side, never switch sides, never "
        "concede the debate.\n"
        f"Your personality and speaking style: {personality}\n"
        "Ground your claims in the research excerpts you are given, citing a "
        "source inline by the actual title shown in brackets above its "
        "excerpt. Where research is thin, draw on your own expertise.\n"
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

    if turn["phase"] == "rebuttal" and transcript:
        opponent_last = next((t["text"] for t in reversed(transcript)
                              if t["speaker"] != side), "")
        query = f"{cfg.topic} {opponent_last[:400]}"
    else:
        query = cfg.topic
    excerpts = index.format_excerpts(query)

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

    task = (
        f"RESEARCH EXCERPTS you may cite:\n\n{excerpts}\n\n"
        f"It is now your turn ({turn['label']}). {PHASE_TASKS[turn['phase']]}\n"
        f"Remember your side: {_stance(side)}"
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


def _argued_wrong_side(llm, topic: str, side: str, text: str) -> bool:
    """Ask the model to classify which side a speech actually argued.
    Small models sometimes drift onto the opponent's side mid-debate,
    especially when the research material is one-sided; this catches it."""
    answer = llm.chat(
        [{"role": "system", "content":
          "You classify debate speeches. Reply with a single word: TRUE if "
          "the speech argues the motion is true (answer yes), or FALSE if it "
          "argues the motion is false (answer no)."},
         {"role": "user", "content":
          f'Motion: "{topic}"\n\nSpeech:\n{text[:2500]}\n\n'
          "Which side does this speech argue? Reply with one word, TRUE or "
          "FALSE."}],
        max_tokens=4, temperature=0.0).strip().upper()
    wrong = "FALSE" if side == "pro" else "TRUE"
    right = "TRUE" if side == "pro" else "FALSE"
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

        emit("phase", phase="debate", message="The debate begins.")
        transcript: list[dict] = []
        for turn in build_schedule(cfg.rounds):
            if stop.is_set():
                raise StopRequested()
            messages = _debater_messages(cfg, turn, transcript, index)
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
                elif _argued_wrong_side(llm, cfg.topic, turn["speaker"], text):
                    problem = ("That speech argued the WRONG SIDE of the "
                               f"motion. {_stance(turn['speaker'])} Rewrite "
                               "your speech so every argument supports YOUR "
                               "side and attacks your opponent's.")
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
                                              transcript, on_criterion)
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
