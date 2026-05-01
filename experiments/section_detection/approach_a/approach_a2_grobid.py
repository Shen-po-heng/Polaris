"""
Approach A2: GROBID + TEI-XML parsing, returning top-level AND subsections.
Returns a flat list in document order with depth and parent fields.

GROBID's body is structurally flat (all <div> are direct children of <body>).
Hierarchy is inferred from the n= attribute, not from XML nesting:
  n="1"     → depth=0, parent=None
  n="3.1"   → depth=1, parent=heading of n="3"
  n="3.2.1" → depth=2, parent=heading of n="3.2"

Requires GROBID service running at localhost:8070:
    docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / "approach_b"))
from approach_b_heuristic import _match_section_type  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from approach_a_grobid import get_tei_xml  # noqa: E402

GROBID_URL = "http://localhost:8070"

# Matches valid section n-attributes: "1", "2.", "3.1", "3.2.", "3.2.1", "A", "A.1"
_VALID_N = re.compile(r"^[A-Z\d]+(\.\d+)*\.?$")


def _n_depth(n: str) -> int:
    """Count dots in n to get depth. "1"→0, "3.1"→1, "3.2.1"→2"""
    return n.rstrip(".").count(".")


def _parent_n(n: str) -> str | None:
    """Return the n of the parent section, or None if top-level."""
    stripped = n.rstrip(".")
    parts = stripped.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def _normalise_n(n: str) -> str:
    """Strip trailing dot for map lookups: "3.1." → "3.1", "1." → "1"."""
    return n.rstrip(".")


@dataclass
class DetectedSection:
    heading: str
    type: str
    depth: int
    parent: str | None
    text: str = ""


def _parse_all_sections(tei_xml: str) -> list[dict]:
    """
    Parse GROBID TEI-XML and return all sections in document order.
    Parent is inferred from n= attribute since GROBID body is structurally flat.
    """
    soup = BeautifulSoup(tei_xml, features="xml")
    results = []

    # Abstract
    abstract_el = soup.find("abstract")
    if abstract_el:
        text = abstract_el.get_text(separator=" ", strip=True)
        if text:
            results.append({"heading": "Abstract", "n": "", "depth": 0, "parent": None, "text": text})

    # Collect all body divs (flat — GROBID doesn't nest them)
    body = soup.find("body")
    raw: list[dict] = []
    if body:
        for div in body.find_all("div", recursive=True):
            head = div.find("head")
            if not head:
                continue
            heading_text = head.get_text(strip=True)
            n = head.get("n", "")

            if not heading_text or not _VALID_N.match(n):
                continue  # skip noise (figure labels, blank headings)

            if heading_text.startswith(n):
                full_heading = heading_text
            else:
                full_heading = f"{n} {heading_text}"

            paragraphs = [p.get_text(separator=" ", strip=True) for p in div.find_all("p", recursive=False)]
            raw.append({
                "heading": full_heading,
                "n": n,
                "depth": _n_depth(n),
                "text": " ".join(paragraphs),
            })

    # Build n → heading map for parent lookup
    n_to_heading: dict[str, str] = {_normalise_n(s["n"]): s["heading"] for s in raw if s["n"]}

    # Deduplicate (GROBID sometimes returns the same div via recursive=True twice)
    seen: set[str] = set()
    unique_raw: list[dict] = []
    for s in raw:
        key = s["n"]
        if key not in seen:
            seen.add(key)
            unique_raw.append(s)

    for s in unique_raw:
        pn = _parent_n(s["n"])
        parent_heading = n_to_heading.get(pn) if pn else None
        results.append({
            "heading": s["heading"],
            "n": s["n"],
            "depth": s["depth"],
            "parent": parent_heading,
            "text": s["text"],
        })

    # References
    back = soup.find("back")
    if back and back.find("listBibl"):
        results.append({"heading": "References", "n": "", "depth": 0, "parent": None, "text": ""})

    return results


def detect_sections(pdf_path: str | Path) -> list[DetectedSection]:
    """
    Detect all sections and subsections using GROBID.
    Returns a flat list in document order with depth and parent fields.
    """
    tei_xml = get_tei_xml(pdf_path)
    raw_sections = _parse_all_sections(tei_xml)

    sections = []
    for sec in raw_sections:
        heading = sec["heading"]
        sec_type = _match_section_type(heading) or "other"
        sections.append(DetectedSection(
            heading=heading,
            type=sec_type,
            depth=sec["depth"],
            parent=sec["parent"],
            text=sec.get("text", ""),
        ))
    return sections


if __name__ == "__main__":
    import io
    import json

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python approach_a2_grobid.py <path/to/paper.pdf>")
        sys.exit(1)

    results = detect_sections(sys.argv[1])
    output = [
        {"heading": s.heading, "type": s.type, "depth": s.depth, "parent": s.parent}
        for s in results
    ]
    print(json.dumps(output, indent=2, ensure_ascii=False))
