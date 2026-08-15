"""Three neutral AI judges scoring the debate on a 100-point ballot."""
from __future__ import annotations

import json
import re

JUDGES = [
    {
        "id": "logician",
        "name": "Judge A · The Logician",
        "persona": (
            "a professor of formal logic who prizes valid reasoning, internal "
            "consistency and quality of evidence, and who penalizes logical "
            "fallacies, unsupported claims and appeals to emotion"
        ),
    },
    {
        "id": "analyst",
        "name": "Judge B · The Policy Analyst",
        "persona": (
            "a pragmatic policy analyst who prizes concrete evidence, "
            "real-world impacts and feasibility, and who rewards debaters who "
            "engage honestly with the strongest version of the opposing case"
        ),
    },
    {
        "id": "rhetorician",
        "name": "Judge C · The Rhetorician",
        "persona": (
            "a professor of rhetoric who prizes persuasion, narrative "
            "structure, clarity and command of language, and who rewards "
            "memorable framing and audience awareness"
        ),
    },
]

# The 100-point ballot: criterion key, label, maximum points.
CRITERIA = [
    ("content", "Content & Evidence", 30),
    ("rebuttal", "Refutation & Clash", 25),
    ("style", "Style & Persuasion", 25),
    ("organization", "Organization & Clarity", 20),
]


def _criteria_text() -> str:
    return "\n".join(f"- {key}: {label} (0-{mx} points)" for key, label, mx in CRITERIA)


def _format_transcript(topic: str, transcript: list[dict]) -> str:
    lines = [f'MOTION: "{topic}"', ""]
    for turn in transcript:
        lines.append(f"--- {turn['label']} ---")
        lines.append(turn["text"])
        lines.append("")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in response")


def _clean_scores(raw: dict) -> dict:
    scores = {}
    for side in ("pro", "con"):
        side_raw = raw.get(side)
        if not isinstance(side_raw, dict):
            raise ValueError(f"missing scores for '{side}'")
        side_scores = {}
        for key, _label, mx in CRITERIA:
            value = side_raw.get(key)
            if not isinstance(value, (int, float)):
                # tolerate "18/30"-style strings
                hit = re.match(r"\s*(\d+)", str(value or ""))
                if not hit:
                    raise ValueError(f"missing score {side}.{key}")
                value = int(hit.group(1))
            side_scores[key] = max(0, min(int(value), mx))
        scores[side] = side_scores
    return scores


def judge_debate(llm, judge: dict, topic: str, transcript: list[dict]) -> dict:
    """Return a ballot: per-criterion scores, totals, winner and reasoning."""
    system = (
        f"You are {judge['name']}, {judge['persona']}. You are a strictly "
        "neutral judge of a formal debate. You have no personal opinion on the "
        "motion; you score only what was said in the transcript. Speaker PRO "
        "argued FOR the motion, speaker CON argued AGAINST it."
    )
    user = (
        _format_transcript(topic, transcript)
        + "\n\nScore each speaker on this 100-point ballot:\n"
        + _criteria_text()
        + "\n\nBe a discerning, critical grader: near-maximum scores should be "
        "rare, and your scores must reflect real differences between the two "
        "speakers — do not give both sides identical scores on a criterion "
        "unless they truly performed equally."
        + "\n\nRespond with ONLY a JSON object in exactly this shape:\n"
        '{"pro": {"content": 0, "rebuttal": 0, "style": 0, "organization": 0},\n'
        ' "con": {"content": 0, "rebuttal": 0, "style": 0, "organization": 0},\n'
        ' "reasoning": "2-4 sentences explaining your scores"}'
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    last_error = None
    for attempt in range(2):
        text = llm.chat(messages, max_tokens=600, temperature=0.3, json_mode=True)
        try:
            raw = _extract_json(text)
            scores = _clean_scores(raw)
            break
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             "That was not valid. Respond with ONLY the JSON "
                             "object, integer scores, no other text."})
    else:
        raise RuntimeError(f"{judge['name']} failed to produce a valid ballot: {last_error}")

    totals = {side: sum(scores[side].values()) for side in ("pro", "con")}
    if totals["pro"] > totals["con"]:
        winner = "pro"
    elif totals["con"] > totals["pro"]:
        winner = "con"
    else:
        winner = "tie"
    reasoning = str(raw.get("reasoning", "")).strip()
    return {"scores": scores, "totals": totals, "winner": winner,
            "reasoning": reasoning}


def tally(ballots: list[dict]) -> dict:
    """Combine judge ballots into a final verdict."""
    totals = {"pro": 0, "con": 0}
    ballots_won = {"pro": 0, "con": 0}
    for ballot in ballots:
        for side in ("pro", "con"):
            totals[side] += ballot["totals"][side]
        if ballot["winner"] in ballots_won:
            ballots_won[ballot["winner"]] += 1

    if ballots_won["pro"] != ballots_won["con"]:
        winner = "pro" if ballots_won["pro"] > ballots_won["con"] else "con"
        method = f"won {max(ballots_won.values())} of {len(ballots)} judge ballots"
    elif totals["pro"] != totals["con"]:
        winner = "pro" if totals["pro"] > totals["con"] else "con"
        method = "won on total points after split ballots"
    else:
        winner = "tie"
        method = "dead tie on ballots and points"
    return {"totals": totals, "ballots_won": ballots_won, "winner": winner,
            "method": method}
