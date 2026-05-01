"""
Inspect GROBID raw output for all benchmark papers.
Saves per-paper files to approach_a/raw_data/:
  - {paper_id}_raw.xml      : full TEI-XML from GROBID
  - {paper_id}_hierarchy.json : section hierarchy (all levels, for inspection)

Usage:
    python experiments/section_detection/approach_a/inspect_grobid_raw.py
Requires GROBID running at localhost:8070.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parent.parent
REF_PAPERS_DIR = REPO_ROOT / "ref_papers"
RAW_DATA_DIR = SCRIPT_DIR / "raw_data"
RAW_DATA_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from approach_a_grobid import get_tei_xml  # noqa: E402

PAPER_PDF_MAP = {
    "1412.6980_adam":        "1412.6980v9.pdf",
    "1512.03385_resnet":     "1512.03385v1.pdf",
    "1706.03762_transformer": "1706.03762v7.pdf",
    "1810.04805_bert":       "1810.04805v2.pdf",
    "NIPS-2012-alexnet":     "NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf",
}


def _extract_hierarchy(tei_xml: str) -> dict:
    """Extract full section hierarchy (all levels) from TEI-XML."""
    soup = BeautifulSoup(tei_xml, features="xml")

    def div_to_node(div: object, depth: int = 0) -> dict | None:
        head = div.find("head", recursive=False)
        if not head:
            return None
        heading_text = head.get_text(strip=True)
        if not heading_text:
            return None
        n = head.get("n", "")
        full_heading = f"{n} {heading_text}" if n else heading_text
        subsections = []
        for child in div.find_all("div", recursive=False):
            node = div_to_node(child, depth + 1)
            if node:
                subsections.append(node)
        return {
            "n": n,
            "heading": full_heading,
            "depth": depth,
            "subsections": subsections,
        }

    result: dict = {"title": None, "abstract": None, "body_sections": [], "has_references": False}

    title_el = soup.find("title", {"level": "a", "type": "main"})
    if title_el:
        result["title"] = title_el.get_text(strip=True)

    abstract_el = soup.find("abstract")
    if abstract_el:
        result["abstract"] = abstract_el.get_text(separator=" ", strip=True)[:500] + "..."

    body = soup.find("body")
    if body:
        for div in body.find_all("div", recursive=False):
            node = div_to_node(div, depth=0)
            if node:
                result["body_sections"].append(node)

    back = soup.find("back")
    if back and back.find("listBibl"):
        result["has_references"] = True

    return result


def main():
    print(f"Saving raw GROBID data to: {RAW_DATA_DIR}\n")

    for paper_id, pdf_name in PAPER_PDF_MAP.items():
        pdf_path = REF_PAPERS_DIR / pdf_name
        if not pdf_path.exists():
            print(f"[SKIP] {paper_id} — PDF not found")
            continue

        print(f"Processing {paper_id} ...", end=" ", flush=True)
        try:
            tei_xml = get_tei_xml(pdf_path)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            continue

        # Save raw TEI-XML
        xml_path = RAW_DATA_DIR / f"{paper_id}_raw.xml"
        xml_path.write_text(tei_xml, encoding="utf-8")

        # Save section hierarchy JSON
        hierarchy = _extract_hierarchy(tei_xml)
        json_path = RAW_DATA_DIR / f"{paper_id}_hierarchy.json"
        json_path.write_text(
            json.dumps(hierarchy, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        n_top = sum(1 for s in hierarchy["body_sections"] if s["n"] and "." not in s["n"])
        n_total = _count_all_sections(hierarchy["body_sections"])
        print(f"done  (top-level: {n_top}, total incl. subsections: {n_total})")
        print(f"  → {xml_path.name}")
        print(f"  → {json_path.name}")

    print("\nDone. Open raw_data/*.json to inspect full section hierarchy.")


def _count_all_sections(sections: list) -> int:
    count = 0
    for s in sections:
        count += 1
        count += _count_all_sections(s.get("subsections", []))
    return count


if __name__ == "__main__":
    main()
