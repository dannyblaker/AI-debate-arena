"""A neutral AI judge scoring the debate on a 100-point ballot.

The ballot is broken into eight small criteria. Each criterion is scored in
its own focused LLM call that must produce written reasoning for each
speaker *before* the score — small models grade far better when they argue
the mark first. Afterwards the judge writes a free-prose summary statement
reviewing the whole scorecard and justifying the final result.
"""
from __future__ import annotations

import json
import re

JUDGES = [
    {
        "id": "judge",
        "name": "The Judge",
        "persona": (
            "a veteran debate adjudicator who prizes valid reasoning, "
            "concrete evidence and real-world feasibility as much as "
            "persuasion, clarity and command of language, and who penalizes "
            "logical fallacies, unsupported claims and failure to engage "
            "with the strongest version of the opposing case"
        ),
    },
]

# The 100-point ballot: eight fine-grained criteria.
CRITERIA = [
    {"key": "evidence", "label": "Evidence & Sourcing", "max": 15,
     "desc": "how well the speaker backs claims with concrete facts, "
             "examples and cited sources rather than bare assertion"},
    {"key": "logic", "label": "Logical Reasoning", "max": 15,
     "desc": "the validity and internal consistency of the speaker's "
             "arguments, and their freedom from logical fallacies"},
    {"key": "refutation", "label": "Direct Refutation", "max": 15,
     "desc": "how directly and effectively the speaker attacks the "
             "opponent's actual arguments, rather than a strawman"},
    {"key": "defense", "label": "Defense & Resilience", "max": 10,
     "desc": "how well the speaker repairs and reinforces their own case "
             "after the opponent's attacks on it"},
    {"key": "persuasion", "label": "Persuasive Impact", "max": 15,
     "desc": "how convincing the speaker's overall case would be to a "
             "neutral, intelligent audience"},
    {"key": "rhetoric", "label": "Language & Rhetoric", "max": 10,
     "desc": "command of language, memorable framing and rhetorical craft"},
    {"key": "structure", "label": "Structure & Signposting", "max": 10,
     "desc": "clear organization and logical flow within each speech and "
             "across the debate as a whole"},
    {"key": "clarity", "label": "Clarity & Concision", "max": 10,
     "desc": "how easy the speeches are to follow; repetition, waffle and "
             "restating earlier speeches must cost points here"},
]


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


def _clean_criterion(raw: dict, mx: int) -> dict:
    """Validate one criterion result: per-side reasoning + bounded score."""
    out = {}
    for side in ("pro", "con"):
        side_raw = raw.get(side)
        if not isinstance(side_raw, dict):
            raise ValueError(f"missing result for '{side}'")
        value = side_raw.get("score")
        if not isinstance(value, (int, float)):
            # tolerate "12/15"-style strings
            hit = re.match(r"\s*(\d+)", str(value or ""))
            if not hit:
                raise ValueError(f"missing score for '{side}'")
            value = int(hit.group(1))
        out[side] = {
            "score": max(0, min(int(value), mx)),
            "reasoning": str(side_raw.get("reasoning", "")).strip(),
        }
    return out


def _score_criterion(llm, system: str, transcript_text: str, crit: dict) -> dict:
    """One focused call: reason about both speakers on a single criterion,
    then score it. Reasoning comes before the score in the JSON so the
    model commits to an argument before committing to a number."""
    user = (
        transcript_text
        + f"\n\nYou are scoring ONE criterion only: {crit['label']} "
        f"(0 to {crit['max']} points) — {crit['desc']}.\n\n"
        "For each speaker, write 2-3 sentences of reasoning evaluating them "
        "on this criterion alone, citing specific moments from the "
        "transcript, and only then award the score. Be a discerning, "
        "critical grader: near-maximum scores should be rare, and identical "
        "scores are only justified when the speakers truly performed "
        "equally on this criterion.\n\n"
        "Respond with ONLY a JSON object in exactly this shape:\n"
        f'{{"pro": {{"reasoning": "2-3 sentences", '
        f'"score": <integer from 0 to {crit["max"]}>}},\n'
        f' "con": {{"reasoning": "2-3 sentences", '
        f'"score": <integer from 0 to {crit["max"]}>}}}}'
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    last_error = None
    for _attempt in range(2):
        text = llm.chat(messages, max_tokens=400, temperature=0.3, json_mode=True)
        try:
            return _clean_criterion(_extract_json(text), crit["max"])
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             "That was not valid. Respond with ONLY the JSON "
                             "object in the requested shape: reasoning "
                             "strings and integer scores, no other text."})
    raise RuntimeError(f"invalid ballot for '{crit['label']}': {last_error}")


def _write_summary(llm, system: str, transcript_text: str, reasons: dict,
                   scores: dict, totals: dict, winner: str) -> str:
    """The judge's closing statement, grounded in the completed scorecard."""
    card = []
    for crit in CRITERIA:
        card.append(f"{crit['label']} (max {crit['max']}):")
        for side in ("pro", "con"):
            card.append(f"  {side.upper()} {scores[side][crit['key']]} — "
                        f"{reasons[side][crit['key']]}")
    result_line = (
        "the debate is a dead tie" if winner == "tie" else
        f"{winner.upper()} wins, {totals['pro']} points to {totals['con']} "
        f"for PRO and CON respectively")
    user = (
        transcript_text
        + "\n\nYOUR COMPLETED SCORECARD:\n" + "\n".join(card)
        + f"\n\nOn your scorecard {result_line}."
        + "\n\nNow write your summary statement as the judge, in first "
        "person, 200-300 words of flowing prose (no headings, no lists, no "
        "score tallies). Explore the decisions behind your scores: what PRO "
        "did well and did less well, what CON did well and did less well, "
        "and finish with your ultimate finding and the justification for "
        "it. Your finding must be consistent with your scorecard."
    )
    return llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        max_tokens=600, temperature=0.5).strip()


def judge_debate(llm, judge: dict, topic: str, transcript: list[dict],
                 on_criterion=None) -> dict:
    """Return a ballot: per-criterion scores and written reasoning for each
    speaker, totals, winner, and a closing summary statement.

    `on_criterion(crit, result)` is called after each criterion is scored,
    so callers can stream partial ballots to the UI."""
    system = (
        f"You are {judge['name']}, {judge['persona']}. You are a strictly "
        "neutral judge of a formal debate. You have no personal opinion on "
        "the motion; you score only what was said in the transcript. Speaker "
        "PRO argued that the motion is true, speaker CON argued that it is "
        "false. Repetition of earlier speeches, failure to engage with the "
        "opponent's actual points, and arguing the wrong side must all be "
        "punished heavily."
    )
    transcript_text = _format_transcript(topic, transcript)

    scores = {"pro": {}, "con": {}}
    reasons = {"pro": {}, "con": {}}
    for crit in CRITERIA:
        try:
            result = _score_criterion(llm, system, transcript_text, crit)
        except RuntimeError:
            # A single bad criterion must not void the whole ballot; zero
            # both sides equally and move on.
            result = {side: {"score": 0, "reasoning":
                             "The judge failed to produce a valid score for "
                             "this criterion; both sides receive 0."}
                      for side in ("pro", "con")}
        for side in ("pro", "con"):
            scores[side][crit["key"]] = result[side]["score"]
            reasons[side][crit["key"]] = result[side]["reasoning"]
        if on_criterion:
            on_criterion(crit, result)

    totals = {side: sum(scores[side].values()) for side in ("pro", "con")}
    if totals["pro"] > totals["con"]:
        winner = "pro"
    elif totals["con"] > totals["pro"]:
        winner = "con"
    else:
        winner = "tie"

    summary = _write_summary(llm, system, transcript_text, reasons, scores,
                             totals, winner)
    return {"scores": scores, "reasons": reasons, "totals": totals,
            "winner": winner, "summary": summary}


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
        method = ("won on the judge's ballot" if len(ballots) == 1 else
                  f"won {max(ballots_won.values())} of {len(ballots)} judge ballots")
    elif totals["pro"] != totals["con"]:
        winner = "pro" if totals["pro"] > totals["con"] else "con"
        method = "won on total points after split ballots"
    else:
        winner = "tie"
        method = "dead tie on ballots and points"
    return {"totals": totals, "ballots_won": ballots_won, "winner": winner,
            "method": method}
