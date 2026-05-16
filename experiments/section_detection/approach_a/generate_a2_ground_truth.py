"""
Generate draft A2 ground truth JSONs (top-level + subsections).
Runs approach_a2_grobid.detect_sections() on each paper and saves
draft JSON to ground_truth_a2/. User then verifies and sets verified=true.

Usage:
    python experiments/section_detection/approach_a/generate_a2_ground_truth.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parent.parent
REF_PAPERS_DIR = REPO_ROOT / "ref_papers"
GT_A2_DIR = EXP_DIR / "ground_truth_a2"
GT_A2_DIR.mkdir(exist_ok=True)

# Load existing top-level ground truth for metadata (title, format)
GT_DIR = EXP_DIR / "ground_truth"

sys.path.insert(0, str(SCRIPT_DIR))
from approach_a2_grobid import detect_sections  # noqa: E402

PAPER_PDF_MAP = {
    "1412.6980":         ("1412.6980v9.pdf",        "1412.6980_adam.json"),
    "1512.03385":        ("1512.03385v1.pdf",        "1512.03385_resnet.json"),
    "1706.03762":        ("1706.03762v7.pdf",        "1706.03762_transformer.json"),
    "1810.04805":        ("1810.04805v2.pdf",        "1810.04805_bert.json"),
    "NIPS-2012-alexnet": (
        "NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf",
        "nips2012_alexnet.json",
    ),
}


def main():
    print("Generating A2 ground truth drafts...\n")

    for paper_id, (pdf_name, gt_filename) in PAPER_PDF_MAP.items():
        pdf_path = REF_PAPERS_DIR / pdf_name
        gt_path = GT_DIR / gt_filename
        out_path = GT_A2_DIR / gt_filename

        if not pdf_path.exists():
            print(f"[SKIP] {paper_id} — PDF not found")
            continue

        # Load existing GT for metadata
        with open(gt_path, encoding="utf-8") as f:
            existing_gt = json.load(f)

        print(f"Processing {paper_id} ...", end=" ", flush=True)
        try:
            sections = detect_sections(pdf_path)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        draft = {
            "paper_id": paper_id,
            "title": existing_gt["title"],
            "format": existing_gt.get("format", ""),
            "verified": False,
            "note": "A2 ground truth — includes subsections. Set verified=true after checking.",
            "sections": [
                {
                    "heading": s.heading,
                    "type": s.type,
                    "depth": s.depth,
                    "parent": s.parent,
                }
                for s in sections
            ],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=2, ensure_ascii=False)

        n_top = sum(1 for s in sections if s.depth == 0)
        n_sub = sum(1 for s in sections if s.depth > 0)
        print(f"done  ({n_top} top-level, {n_sub} subsections) → {out_path.name}")

    print(f"\nDraft files saved to: {GT_A2_DIR}")
    print("Next steps:")
    print("  1. Open each JSON in ground_truth_a2/")
    print("  2. Check headings and fix type labels where needed")
    print("  3. Set verified=true when done")


if __name__ == "__main__":
    main()
