"""
Approach A: GROBID + direct TEI-XML parsing for academic section detection.
Only top-level sections are returned (n="1", "2", ... with no dots).
Subsections → planned as Approach A2 (future work).

Requires GROBID service running at localhost:8070:
    docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Reuse the same type-inference logic as Approach B — keeps comparison fair
sys.path.insert(0, str(Path(__file__).parent.parent / "approach_b"))
from approach_b_heuristic import _match_section_type  # noqa: E402

GROBID_URL = "http://localhost:8070"
_TOP_LEVEL_N = re.compile(r"^\d+\.?$")  # "1", "2", "1.", "2." — single level, no sub-numbering


@dataclass
class DetectedSection:
    heading: str
    type: str
    text: str = ""


def _get_tei_xml(pdf_path: str | Path) -> str:
    """Call GROBID REST API and return raw TEI-XML string."""
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{GROBID_URL}/api/processFulltextDocument",
                files={"input": f},
                data={"consolidateHeader": "0"},
                timeout=120,
            )
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise RuntimeError(
            f"GROBID unavailable: {e}\n"
            "Start with: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0"
        )


def _parse_top_level_sections(tei_xml: str) -> list[dict]:
    """
    Parse GROBID TEI-XML, returning only top-level sections.

    Top-level = <div> direct children of <body> where head/@n is a
    single integer (e.g. "1", "2") — subsections have dotted n like "1.1".
    Abstract and References are added as special cases.
    """
    soup = BeautifulSoup(tei_xml, features="xml")
    results = []

    # Abstract lives in <profileDesc><abstract>
    abstract_el = soup.find("abstract")
    if abstract_el:
        text = abstract_el.get_text(separator=" ", strip=True)
        if text:
            results.append({"heading": "Abstract", "text": text})

    # Top-level body sections
    body = soup.find("body")
    if body:
        for div in body.find_all("div", recursive=False):
            head = div.find("head", recursive=False)
            if not head:
                continue
            heading_text = head.get_text(strip=True)
            if not heading_text:
                continue
            n = head.get("n", "")
            if not _TOP_LEVEL_N.match(n):
                continue  # skip subsections and unlabelled noise

            # Reconstruct heading: if text already has the number (e.g. "1. Introduction"),
            # use as-is; otherwise prepend n (e.g. n="1" + "Introduction" → "1 Introduction")
            if heading_text.startswith(n):
                full_heading = heading_text
            else:
                full_heading = f"{n} {heading_text}"

            paragraphs = [
                p.get_text(separator=" ", strip=True)
                for p in div.find_all("p", recursive=False)
            ]
            results.append({"heading": full_heading, "text": " ".join(paragraphs)})

    # References live in <back>
    back = soup.find("back")
    if back and back.find("listBibl"):
        results.append({"heading": "References", "text": ""})

    return results


def detect_sections(pdf_path: str | Path) -> list[DetectedSection]:
    """
    Detect top-level sections in an academic PDF using GROBID.
    Returns a list of DetectedSection in document order.
    """
    tei_xml = _get_tei_xml(pdf_path)
    raw_sections = _parse_top_level_sections(tei_xml)

    sections = []
    for sec in raw_sections:
        heading = sec["heading"]
        sec_type = _match_section_type(heading) or "other"
        sections.append(DetectedSection(
            heading=heading,
            type=sec_type,
            text=sec.get("text", ""),
        ))
    return sections


def get_tei_xml(pdf_path: str | Path) -> str:
    """Public accessor for raw TEI-XML (used by inspect_grobid_raw.py)."""
    return _get_tei_xml(pdf_path)


if __name__ == "__main__":
    import io
    import json

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python approach_a_grobid.py <path/to/paper.pdf>")
        sys.exit(1)

    results = detect_sections(sys.argv[1])
    output = [{"heading": s.heading, "type": s.type} for s in results]
    print(json.dumps(output, indent=2, ensure_ascii=False))
