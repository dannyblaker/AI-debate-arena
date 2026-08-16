"""Pre-debate case preparation: each side gets a brief of verbatim evidence.

Per-turn retrieval only surfaces what a debater's ad-hoc queries happen to
hit, so decisive passages — a document's conclusion, a scene near the end of
a script — are easily missed when the same keywords also appear in dozens of
less important places. This stage sweeps the research library once,
window-by-window, extracting short verbatim quotations that help either
side. Every quotation is checked against the source text (hallucinated
quotes are dropped), then each side's model selects its strongest items.
The resulting briefs travel with the debaters through every turn.
"""
from __future__ import annotations

import re
import threading
from typing import Callable

from .config import (PREP_MAX_WINDOWS, PREP_QUOTES_PER_SIDE,
                     PREP_WINDOW_CHARS, PREP_WINDOW_OVERLAP)
from .models_registry import StopRequested
from .rag import ResearchIndex

# More candidates than this makes the selection prompt unwieldy for a small
# model without adding real signal.
MAX_CANDIDATES = 80

EXTRACT_SYSTEM = (
    "You mine source material for evidence usable in a formal debate. "
    "Reply only with quote lines in the exact format requested — no "
    "commentary, no numbering, no other text."
)


def derive_positions(llm, topic: str) -> dict[str, str]:
    """Rewrite each side's burden of proof as one plain positive sentence.

    Small models handle 'argue the motion is FALSE' badly when the motion
    itself is negated ('X never understood Y'): every step that re-derives
    the double negation — quote mining, brief selection, the speeches
    themselves — is a fresh chance to flip sides. Deriving the mapping ONCE
    and passing the plain sentences everywhere removes those re-derivations.
    """
    fallback = {
        "pro": f'The motion is true: {topic}',
        "con": f'The motion is false; the opposite of "{topic}" is the case.',
    }
    user = (
        f'A formal debate has the motion: "{topic}"\n\n'
        "The PRO side argues the motion is TRUE (if it is phrased as a "
        "question, PRO answers YES). The CON side argues the motion is "
        "FALSE (if it is phrased as a question, CON answers NO).\n\n"
        "State each side's claim as one short, direct sentence in plain "
        "words. CON's sentence must assert the opposite of the motion as a "
        "positive claim of its own — never just 'the motion is false'. If "
        "the motion says something never happened, CON's claim is that it "
        "DID happen.\n\n"
        "Reply in exactly this format:\n"
        "PRO: <one sentence>\n"
        "CON: <one sentence>"
    )
    try:
        raw = llm.chat(
            [{"role": "system", "content":
              "You restate debate positions precisely. Reply in exactly "
              "the requested format, nothing else."},
             {"role": "user", "content": user}],
            max_tokens=150, temperature=0.0)
    except Exception:
        return fallback
    out = dict(fallback)
    for line in raw.splitlines():
        m = re.match(r"\s*(?:PRO|CON)\s*:", line)
        if not m:
            continue
        side = line.split(":", 1)[0].strip().lower()
        claim = line.split(":", 1)[1].strip()
        if len(claim) > 10:
            out[side] = claim[:300]
    return out


def _normalize(text: str) -> str:
    """Reduce text to a form where 'the same words' compare equal despite
    punctuation, casing, curly quotes and whitespace differences."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _windows(index: ResearchIndex) -> list[tuple[int, int, int]]:
    """(doc_idx, start, end) spans covering every document completely.
    Overlap keeps a quotation that straddles a boundary intact in at least
    one window."""
    wins = []
    step = PREP_WINDOW_CHARS - PREP_WINDOW_OVERLAP
    for di, doc in enumerate(index.docs):
        for start in range(0, max(len(doc.text), 1), step):
            end = min(start + PREP_WINDOW_CHARS, len(doc.text))
            if end - start >= 300 or start == 0:
                wins.append((di, start, end))
            if end >= len(doc.text):
                break
    return wins


def _select_windows(index: ResearchIndex, topic: str,
                    wins: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """When the library is too large to sweep in full, keep the windows most
    relevant to the motion — plus each document's final window, because a
    narrative's resolution or a paper's conclusion is disproportionately
    likely to hold decisive material."""
    if len(wins) <= PREP_MAX_WINDOWS:
        return wins
    ranked = index.retrieve(topic, k=len(index.chunks))
    best_rank: dict[int, int] = {}
    for rank, chunk in enumerate(ranked):
        for wi, (di, s, e) in enumerate(wins):
            if di == chunk.doc_idx and chunk.start < e and chunk.end > s:
                best_rank[wi] = min(best_rank.get(wi, rank), rank)
    last_of_doc = {}
    for wi, (di, _s, _e) in enumerate(wins):
        last_of_doc[di] = wi
    chosen = set(last_of_doc.values())
    for wi in sorted(range(len(wins)), key=lambda w: best_rank.get(w, 10 ** 9)):
        if len(chosen) >= PREP_MAX_WINDOWS:
            break
        chosen.add(wi)
    return [wins[wi] for wi in sorted(chosen)]


def _extract_quotes(llm, topic: str, positions: dict[str, str],
                    title: str, text: str) -> dict[str, list]:
    """One focused call: pull verbatim quotations from a single passage for
    both sides at once. Returns {'pro': [(quote, reason)], 'con': [...]},
    keeping only quotes that genuinely appear in the passage."""
    user = (
        f'Motion under debate: "{topic}"\n'
        f"The FOR side must prove: {positions['pro']}\n"
        f"The AGAINST side must prove: {positions['con']}\n\n"
        f"Source: {title}\n---\n{text}\n---\n\n"
        "From the passage above, copy out up to 3 quotations (each under 60 "
        "words, copied EXACTLY word for word) that support what the FOR "
        "side must prove, and up to 3 that support what the AGAINST side "
        "must prove. Judge each quotation only by which side's claim it "
        "supports, exactly as those claims are stated above. Prefer lines that bear "
        "directly on the motion's exact claim — above all any line where a "
        "key phrase or idea from the motion itself is addressed head-on. "
        "Quote complete statements: never shorten a quotation with an "
        "ellipsis, and when the sentence after a key line completes its "
        "meaning, include it. Skip a side if the passage offers it "
        "nothing.\n\n"
        "One quotation per line, in exactly this format:\n"
        'FOR: "exact quotation" | why it matters\n'
        'AGAINST: "exact quotation" | why it matters'
    )
    raw = llm.chat([{"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": user}],
                   max_tokens=400, temperature=0.2)
    haystack = _normalize(text)
    out: dict[str, list] = {"pro": [], "con": []}
    for line in raw.splitlines():
        m = re.match(r"\s*(FOR|AGAINST)\s*:\s*(.+)", line)
        if not m:
            continue
        side = "pro" if m.group(1) == "FOR" else "con"
        body = m.group(2)
        quote, _, reason = body.partition("|")
        quote = quote.strip().strip('"“”‘’\'').strip()
        reason = re.sub(r"\s+", " ", reason).strip()[:200]
        # A quote the source does not actually contain is worse than no
        # quote at all — drop anything that fails verbatim verification.
        if len(quote) >= 15 and _normalize(quote) in haystack:
            out[side].append((quote, reason))
    return out


def _classify_quotes(llm, positions: dict[str, str],
                     pool: list[tuple[str, str, str]],
                     stop: threading.Event | None = None,
                     on_sorted=None) -> list[str]:
    """Label every pooled quote 'pro', 'con' or 'neutral' by which claim it
    supports — one focused call per quote, reasoning before the verdict
    (small models decide far better that way; batch classification anchors
    on each quote's opening words and misfiles quotes that turn, like
    'I disdained the motto... but now I understand it'). Without this step
    the most dramatic quote in the pool ends up in BOTH briefs."""
    labels = []
    for i, (_title, quote, _reason) in enumerate(pool):
        if stop is not None and stop.is_set():
            raise StopRequested()
        # Alternate which claim is presented as 'A': position order sways
        # borderline verdicts, and alternating cancels that bias out
        # instead of tilting every brief toward the same side.
        first, second = ("pro", "con") if i % 2 == 0 else ("con", "pro")
        user = (
            f"Claim A: {positions[first]}\n"
            f"Claim B: {positions[second]}\n\n"
            f'Evidence quote:\n"{quote}"\n\n'
            "Taken as a whole, which claim does this quote support? First "
            "write REASON: followed by one sentence weighing what the "
            "quote as a whole shows. Then write VERDICT: followed by A, "
            "B, or N (N only if it truly supports neither claim more than "
            "the other)."
        )
        try:
            raw = llm.chat(
                [{"role": "system", "content":
                  "You sort debate evidence between two opposing claims, "
                  "judging each quote by its overall meaning, not its "
                  "opening words. Reply with exactly one REASON line and "
                  "one VERDICT line."},
                 {"role": "user", "content": user}],
                max_tokens=90, temperature=0.0)
            m = re.search(r"VERDICT\s*[:\-]?\s*([ABN])\b", raw, re.IGNORECASE)
            verdict = m.group(1).upper() if m else "N"
        except Exception:
            verdict = "N"
        label = {"A": first, "B": second, "N": "neutral"}[verdict]
        labels.append(label)
        if on_sorted:
            on_sorted(i, label)
    return labels


def _select_strongest(llm, topic: str, positions: dict[str, str], side: str,
                      candidates: list[tuple[str, str, str]]) -> list:
    """Have the side's debater rank its candidate evidence. The model only
    picks numbers; the quotes themselves are reassembled from the verified
    candidates, so it cannot corrupt them."""
    if len(candidates) <= PREP_QUOTES_PER_SIDE:
        return candidates
    if len(candidates) > MAX_CANDIDATES:
        # Thin evenly rather than truncating: quotes arrive in document
        # order, and a document's ending is as likely to matter as its start.
        step = len(candidates) / MAX_CANDIDATES
        candidates = [candidates[int(i * step)] for i in range(MAX_CANDIDATES)]
    listing = "\n".join(
        f'{i + 1}. [{t}] "{q}"' + (f" — {r}" if r else "")
        for i, (t, q, r) in enumerate(candidates))
    user = (
        f'Motion under debate: "{topic}"\n'
        f"You must convince the judge that: {positions[side]}\n\n"
        f"Candidate evidence:\n{listing}\n\n"
        f"Pick the {PREP_QUOTES_PER_SIDE} items that most strongly support "
        "your claim, best first. Some items support the OPPOSITE claim — "
        "leave those out. Reply with only the chosen numbers, "
        "comma-separated."
    )
    try:
        raw = llm.chat(
            [{"role": "system", "content":
              "You rank evidence for a debater. Reply with only numbers, "
              "comma-separated."},
             {"role": "user", "content": user}],
            max_tokens=60, temperature=0.2)
        nums = [int(n) for n in re.findall(r"\d+", raw)]
    except Exception:
        nums = []
    picked = [candidates[n - 1] for n in dict.fromkeys(nums)
              if 1 <= n <= len(candidates)]
    # A short brief of chosen items beats one padded with quotes that may
    # favour the opponent; only fall back when the model chose nothing.
    return (picked or candidates)[:PREP_QUOTES_PER_SIDE]


def _brief_text(items: list[tuple[str, str, str]]) -> str:
    return "\n".join(
        f'{i + 1}. [{t}] "{q}"' + (f" — {r}" if r else "")
        for i, (t, q, r) in enumerate(items))


def build_briefs(llm, topic: str, positions: dict[str, str],
                 index: ResearchIndex,
                 emit: Callable[..., None], stop: threading.Event) -> dict[str, str]:
    """Mine the research library and return {'pro': brief, 'con': brief}
    (empty strings when nothing usable was found)."""
    if not index.chunks:
        return {"pro": "", "con": ""}
    wins = _select_windows(index, topic, _windows(index))
    emit("prep_start", total=len(wins))
    emit("status", message=f"Case prep: mining {len(wins)} passages of "
                           "source material for evidence…")
    # All verified quotes go into ONE pool that both sides select from:
    # small models regularly misjudge which side a quote helps (especially
    # under negated motions), and a hard FOR/AGAINST split at extraction
    # time would lose the misfiled evidence for good.
    pool: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for i, (di, s, e) in enumerate(wins, 1):
        if stop.is_set():
            raise StopRequested()
        doc = index.docs[di]
        emit("prep_window", index=i, total=len(wins), source=doc.title)
        try:
            found = _extract_quotes(llm, topic, positions, doc.title,
                                    doc.text[s:e])
        except Exception:
            continue
        for side in ("pro", "con"):
            for quote, reason in found[side]:
                key = _normalize(quote)
                if key not in seen:
                    seen.add(key)
                    pool.append((doc.title, quote, reason))
                    emit("prep_quote", quote=quote[:200], source=doc.title)
        emit("status", message=f"Case prep: passage {i}/{len(wins)} — "
             f"{len(pool)} verified quotes gathered")
    emit("prep_sort_start", total=len(pool))
    emit("status", message=f"Case prep: sorting {len(pool)} quotes "
                           "between the two sides…")
    labels = _classify_quotes(
        llm, positions, pool, stop,
        on_sorted=lambda i, label: emit("prep_sort", index=i, side=label))
    briefs = {}
    for side in ("pro", "con"):
        if stop.is_set():
            raise StopRequested()
        emit("status", message=f"{side.upper()} selects its strongest "
                               "evidence…")
        side_pool = [c for c, lab in zip(pool, labels)
                     if lab in (side, "neutral")]
        picked = _select_strongest(llm, topic, positions, side, side_pool)
        briefs[side] = _brief_text(picked)
        emit("prep_brief", side=side,
             quotes=[{"quote": q[:200], "source": t} for t, q, _r in picked])
    emit("prep_done")
    emit("status", message="Case prep complete — each side holds its "
                           "evidence brief.")
    return briefs
