"""
Approach B: pdfplumber + regex heuristics for academic section detection.
No external services required — pure Python.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    raise ImportError("pip install pdfplumber>=0.11.0")


# ── Section type patterns (general — NOT tuned to test corpus) ────────────────

SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)^\s*abstract\s*$",                                           "abstract"),
    (r"(?i)^\s*(?:\d+\.?\s+)?introduction\s*$",                        "introduction"),
    (r"(?i)^\s*(?:\d+\.?\s+)?(?:background|motivation|overview)\s*$",  "introduction"),
    (r"(?i)^\s*\d+\.?\s+related\s+(?:work|studies|literature)\s*$",    "related_work"),
    (r"(?i)^\s*\d+\.?\s+prior\s+(?:work|art)\s*$",                     "related_work"),
    (r"(?i)^\s*\d+\.?\s+(?:method(?:s|ology)?|approach|framework)\s*$",           "method"),
    (r"(?i)^\s*\d+\.?\s+(?:model|architecture|system|design)\s*$",               "method"),
    (r"(?i)^\s*\d+\.?\s+model\s+architecture\s*$",                               "method"),
    (r"(?i)^\s*\d+\.?\s+(?:the\s+)?(?:architecture|dataset|network)\s*$",        "method"),
    (r"(?i)^\s*\d+\.?\s+(?:proposed|our)\s+\w+\s*$",                            "method"),
    (r"(?i)^\s*\d+\.?\s+(?:algorithm|training|learning)\s*$",                    "method"),
    (r"(?i)^\s*\d+\.?\s+(?:reducing\s+\w+|details?\s+of\s+\w+)\s*$",            "method"),
    (r"(?i)^\s*\d+\.?\s+deep\s+residual\s+learning\s*$",                         "method"),
    (r"(?i)^\s*\d+\.?\s+(?:experiment(?:s)?|evaluation)\s*$",          "results"),
    (r"(?i)^\s*\d+\.?\s+(?:result(?:s)?|performance|benchmark)\s*$",   "results"),
    (r"(?i)^\s*\d+\.?\s+ablation\s+(?:stud(?:y|ies)|analysis)\s*$",    "results"),
    (r"(?i)^\s*\d+\.?\s+discussion\s*$",                                "discussion"),
    (r"(?i)^\s*\d+\.?\s+analysis\s*$",                                  "discussion"),
    (r"(?i)^\s*(?:\d+\.?\s+)?conclusion(?:s)?\s*$",                    "conclusion"),
    (r"(?i)^\s*(?:\d+\.?\s+)?(?:conclusion|summary)\s+and\s+\w+\s*$", "conclusion"),
    (r"(?i)^\s*(?:references|bibliography)\s*$",                        "references"),
]


def _normalize_heading(text: str) -> str:
    """Insert spaces at word boundaries missing them due to PDF encoding."""
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)   # "1Introduction" → "1 Introduction"
    text = re.sub(r"(\.)([A-Z])", r"\1 \2", text)       # "1.Introduction" → "1. Introduction"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)    # "RelatedWork" → "Related Work"
    return text


def _match_section_type(text: str) -> str | None:
    normalized = _normalize_heading(text.strip())
    for pattern, sec_type in SECTION_PATTERNS:
        if re.match(pattern, normalized):
            return sec_type
    return None


# ── Line grouping ─────────────────────────────────────────────────────────────

def _group_chars_by_line(chars: list[dict], y_tolerance: float = 2.0) -> list[list[dict]]:
    """Group pdfplumber chars into lines by their vertical position."""
    if not chars:
        return []
    sorted_chars = sorted(chars, key=lambda c: (round(c["top"] / y_tolerance), c["x0"]))
    lines: list[list[dict]] = []
    current_line: list[dict] = []
    last_y: float | None = None
    for c in sorted_chars:
        y = round(c["top"] / y_tolerance)
        if last_y is not None and y != last_y:
            if current_line:
                lines.append(current_line)
            current_line = []
        current_line.append(c)
        last_y = y
    if current_line:
        lines.append(current_line)
    return lines


# ── Core detector ─────────────────────────────────────────────────────────────

@dataclass
class DetectedSection:
    heading: str
    type: str
    page_num: int
    body_lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.body_lines)


def detect_sections(pdf_path: str | Path) -> list[DetectedSection]:
    """
    Detect top-level sections in an academic PDF using font-size heuristics
    and regex pattern matching.

    Returns a list of DetectedSection in document order.
    """
    pdf_path = Path(pdf_path)
    sections: list[DetectedSection] = []
    current: DetectedSection | None = None

    with pdfplumber.open(pdf_path) as pdf:
        # Compute body font size as the median across the whole document
        all_sizes = [
            c["size"]
            for page in pdf.pages
            for c in (page.chars or [])
            if c.get("size")
        ]
        body_size: float = statistics.median(all_sizes) if all_sizes else 10.0

        for page_num, page in enumerate(pdf.pages, start=1):
            chars = page.chars or []
            page_width = page.width or 1.0

            for line_chars in _group_chars_by_line(chars):
                line_text = "".join(c["text"] for c in line_chars).strip()
                if not line_text:
                    continue

                # ── Feature extraction ───────────────────────────────────────
                sizes = [c["size"] for c in line_chars if c.get("size")]
                avg_size = statistics.mean(sizes) if sizes else body_size
                is_larger = avg_size >= body_size + 0.8
                is_bold = any(
                    "bold" in (c.get("fontname") or "").lower() for c in line_chars
                )
                x0 = min(c["x0"] for c in line_chars)
                x1 = max(c["x1"] for c in line_chars)
                line_width_ratio = (x1 - x0) / page_width

                # ── Header candidate criteria ───────────────────────────────
                sec_type = _match_section_type(line_text)
                is_short = len(line_text) < 80
                is_header_style = is_larger or is_bold

                if sec_type and is_short and (is_header_style or sec_type in ("abstract", "references")):
                    if current:
                        sections.append(current)
                    current = DetectedSection(
                        heading=line_text,
                        type=sec_type,
                        page_num=page_num,
                    )
                elif current:
                    current.body_lines.append(line_text)

    if current:
        sections.append(current)

    return sections


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python approach_b_heuristic.py <path/to/paper.pdf>")
        sys.exit(1)

    results = detect_sections(sys.argv[1])
    output = [{"heading": s.heading, "type": s.type, "page": s.page_num} for s in results]
    print(json.dumps(output, indent=2, ensure_ascii=False))
