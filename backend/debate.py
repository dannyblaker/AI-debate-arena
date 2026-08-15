"""Debate pipeline: model download/load -> research -> debate -> judging.

Runs in a worker thread (llama.cpp calls are blocking) and reports everything
through an `emit(type, **data)` callback supplied by the manager in main.py.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from . import judging, research
from .config import FAKE_LLM
from .llm import load_llm
from .models_registry import StopRequested, ensure_model_file, get_model
from .rag import ResearchIndex

DEFAULT_PERSONALITY = "A confident, articulate professional debater."

PHASE_TASKS = {
    "opening": (
        "Deliver your OPENING STATEMENT: frame the motion on your terms and "
        "present your two or three strongest arguments. Under 280 words."
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


def _debater_messages(cfg: DebateConfig, turn: dict, transcript: list[dict],
                      index: ResearchIndex) -> list[dict]:
    side = turn["speaker"]
    stance = "FOR" if side == "pro" else "AGAINST"
    personality = (cfg.pro_personality if side == "pro" else cfg.con_personality) \
        .strip() or DEFAULT_PERSONALITY

    system = (
        f'You are a world-class competitive debater and subject-matter expert. '
        f'You are debating the motion: "{cfg.topic}".\n'
        f"You argue {stance} the motion — always. Never switch sides, never "
        "concede the debate, never break character.\n"
        f"Your personality and speaking style: {personality}\n"
        "Ground your claims in the RESEARCH EXCERPTS when they are relevant, "
        "citing a source inline by the actual title shown in brackets above "
        "its excerpt. Where research is thin, draw on your own expertise.\n"
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
    excerpts = index.format_excerpts(query, k=5)

    if transcript:
        history = "\n\n".join(f"--- {t['label']} ---\n{t['text']}"
                              for t in transcript)
        history_block = f"TRANSCRIPT SO FAR:\n\n{history}\n\n"
    else:
        history_block = "You are the first speaker.\n\n"

    user = (
        f"RESEARCH EXCERPTS:\n\n{excerpts}\n\n"
        + history_block
        + f"It is now your turn ({turn['label']}). "
        + PHASE_TASKS[turn["phase"]]
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


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
        docs = research.gather(cfg.topic, emit, stop)
        index = ResearchIndex(docs)
        emit("research_done", num_sources=len(docs), num_chunks=len(index.chunks))

        emit("phase", phase="debate", message="The debate begins.")
        transcript: list[dict] = []
        for turn in build_schedule(cfg.rounds):
            if stop.is_set():
                raise StopRequested()
            emit("turn_start", **turn)
            messages = _debater_messages(cfg, turn, transcript, index)
            text = ""
            for token in llm.chat_stream(messages, max_tokens=600, temperature=0.8):
                if stop.is_set():
                    raise StopRequested()
                text += token
                emit("token", speaker=turn["speaker"], text=token)
            entry = {**turn, "text": text.strip()}
            transcript.append(entry)
            emit("turn_end", **entry)

        emit("phase", phase="judging", message="The judges deliberate…")
        ballots = []
        for judge in judging.JUDGES:
            if stop.is_set():
                raise StopRequested()
            emit("judge_start", judge_id=judge["id"], name=judge["name"])
            try:
                ballot = judging.judge_debate(llm, judge, cfg.topic, transcript)
            except RuntimeError as e:
                emit("status", message=str(e))
                ballot = {"scores": {s: {k: 0 for k, _, _ in judging.CRITERIA}
                                     for s in ("pro", "con")},
                          "totals": {"pro": 0, "con": 0}, "winner": "tie",
                          "reasoning": "Ballot invalid; judge abstained."}
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
