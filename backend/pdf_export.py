"""PDF transcript export using fpdf2."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from .judging import CRITERIA

_DEJAVU = Path("/usr/share/fonts/truetype/dejavu")

SIDE_NAMES = {"pro": "PRO (for the motion)", "con": "CON (against the motion)"}


class _PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(True, margin=18)
        if (_DEJAVU / "DejaVuSans.ttf").exists():
            self.add_font("body", "", _DEJAVU / "DejaVuSans.ttf")
            self.add_font("body", "B", _DEJAVU / "DejaVuSans-Bold.ttf")
            self.add_font("body", "I", (_DEJAVU / "DejaVuSans-Oblique.ttf")
                          if (_DEJAVU / "DejaVuSans-Oblique.ttf").exists()
                          else _DEJAVU / "DejaVuSans.ttf")
            self.family = "body"
        else:
            self.family = "helvetica"

    def txt(self, s: str) -> str:
        if self.family == "helvetica":
            return s.encode("latin-1", "replace").decode("latin-1")
        return s

    def heading(self, text, size=16):
        self.set_font(self.family, "B", size)
        self.multi_cell(0, 8, self.txt(text))
        self.ln(2)

    def body(self, text, size=10.5, style=""):
        self.set_font(self.family, style, size)
        self.multi_cell(0, 5.5, self.txt(text))
        self.ln(2)


def build_pdf(state: dict) -> bytes:
    pdf = _PDF()
    pdf.add_page()

    pdf.heading("AI Debate Transcript", 20)
    pdf.body(f'Motion: "{state.get("topic", "")}"', 13, "B")

    cfg = state.get("config", {})
    model = state.get("model", {})
    meta = [
        f"Date: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        f"Model: {model.get('name', 'unknown')} ({model.get('repo', '')})",
        f"Rebuttal rounds: {cfg.get('rounds', '?')}",
        f"PRO personality: {cfg.get('pro_personality') or 'default'}",
        f"CON personality: {cfg.get('con_personality') or 'default'}",
        f"Research sources: {len(state.get('sources', []))}",
    ]
    pdf.body("\n".join(meta), 9.5)

    sources = state.get("sources", [])
    if sources:
        pdf.heading("Research Sources", 13)
        pdf.body("\n".join(f"- {s['title']} — {s['url']}" for s in sources), 8.5)

    pdf.add_page()
    pdf.heading("Transcript", 16)
    for turn in state.get("transcript", []):
        pdf.set_text_color(*(0x1a, 0x6e, 0x3c) if turn["speaker"] == "pro"
                           else (0x9e, 0x2a, 0x2a))
        pdf.body(turn["label"], 12, "B")
        pdf.set_text_color(20, 20, 20)
        pdf.body(turn["text"])
        pdf.ln(2)

    judges = state.get("judges", [])
    if any(j.get("ballot") for j in judges):
        pdf.add_page()
        pdf.heading("Judges' Ballots (100-point scale)", 16)
        for j in judges:
            ballot = j.get("ballot")
            if not ballot:
                continue
            pdf.body(j["name"], 12, "B")
            lines = []
            for key, label, mx in CRITERIA:
                lines.append(
                    f"  {label} (max {mx}):  PRO {ballot['scores']['pro'][key]}"
                    f"  ·  CON {ballot['scores']['con'][key]}")
            lines.append(f"  TOTAL:  PRO {ballot['totals']['pro']}/100"
                         f"  ·  CON {ballot['totals']['con']}/100")
            pdf.body("\n".join(lines), 9.5)
            if ballot.get("reasoning"):
                pdf.body(f"Reasoning: {ballot['reasoning']}", 9, "I")
            pdf.ln(2)

    verdict = state.get("verdict")
    if verdict:
        pdf.heading("Final Verdict", 16)
        winner = verdict["winner"]
        if winner == "tie":
            headline = "The debate is a TIE."
        else:
            headline = f"Winner: {SIDE_NAMES[winner]} — {verdict['method']}."
        pdf.body(headline, 12, "B")
        pdf.body(
            f"Combined points — PRO: {verdict['totals']['pro']}"
            f" · CON: {verdict['totals']['con']}   |   Ballots — PRO: "
            f"{verdict['ballots_won']['pro']} · CON: {verdict['ballots_won']['con']}",
            10)

    return bytes(pdf.output())
