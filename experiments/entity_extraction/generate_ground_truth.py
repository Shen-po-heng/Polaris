"""
Generate draft ground truth JSONs for Experiment 2: Entity Extraction.

Runs approach_a2_grobid.detect_sections() on each paper to extract section text,
then saves 15 draft ground truth files to ground_truth/.
User then manually fills in entities/relations and sets verified=true.

Usage:
    python experiments/entity_extraction/generate_ground_truth.py

Requires: GROBID running at localhost:8070
    docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
REF_PAPERS = REPO_ROOT / "ref_papers"
GT_DIR = SCRIPT_DIR / "ground_truth"

sys.path.insert(0, str(REPO_ROOT / "experiments" / "section_detection" / "approach_a"))
from approach_a2_grobid import detect_sections  # noqa: E402

PAPER_PDF_MAP = {
    "1412.6980": "1412.6980v9.pdf",
    "1512.03385": "1512.03385v1.pdf",
    "1706.03762": "1706.03762v7.pdf",
    "1810.04805": "1810.04805v2.pdf",
    "NIPS-2012-alexnet": (
        "NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf"
    ),
}

PAPER_TITLE_MAP = {
    "1412.6980": "Adam: A Method for Stochastic Optimization",
    "1512.03385": "Deep Residual Learning for Image Recognition",
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers",
    "NIPS-2012-alexnet": "ImageNet Classification with Deep CNNs (AlexNet)",
}

# 15 target chunks: (paper_id, heading_keywords, section_type, output_slug)
# heading_keywords: list of lowercase substrings — first section whose heading
# contains ALL keywords wins.
TARGETS: list[tuple[str, list[str], str, str]] = [
    # Adam — method section is 2.1 (GROBID misses "2 ALGORITHM" parent)
    ("1412.6980", ["adam", "update rule"], "method", "adam_method"),
    ("1412.6980", ["experiments"], "results", "adam_results"),
    ("1412.6980", ["related work"], "related_work", "adam_related"),
    # ResNet
    ("1512.03385", ["deep residual learning"], "method", "resnet_method"),
    ("1512.03385", ["imagenet classification"], "results", "resnet_results"),
    ("1512.03385", ["related work"], "related_work", "resnet_related"),
    # Transformer
    ("1706.03762", ["model architecture"], "method", "transformer_method"),
    ("1706.03762", ["machine translation"], "results", "transformer_results"),
    ("1706.03762", ["background"], "related_work", "transformer_related"),
    # BERT
    ("1810.04805", ["pre-training bert"], "method", "bert_method"),
    ("1810.04805", ["glue"], "results", "bert_results"),
    ("1810.04805", ["related work"], "related_work", "bert_related"),
    # AlexNet
    ("NIPS-2012-alexnet", ["the architecture"], "method", "alexnet_method"),
    ("NIPS-2012-alexnet", ["results"], "results", "alexnet_results"),
    ("NIPS-2012-alexnet", ["introduction"], "introduction", "alexnet_intro"),
]

TEXT_MAX_CHARS = 1500
SHORT_TEXT_THRESHOLD = 400  # chars — if parent text is shorter, merge all children


def _slug_to_filename(slug: str) -> str:
    return f"{slug}.json"


def _find_section(sections, keywords: list[str]):
    """Return first DetectedSection whose heading (lowercased) contains all keywords."""
    for sec in sections:
        h = sec.heading.lower()
        if all(kw in h for kw in keywords):
            return sec
    return None


def _merge_children_text(sections, parent_heading: str) -> str:
    """Collect and join text of all direct children of parent_heading."""
    parts = []
    for sec in sections:
        if sec.parent == parent_heading and sec.text.strip():
            parts.append(f"[{sec.heading}] {sec.text.strip()}")
    return "\n\n".join(parts)


def _resolve_text(sections, sec) -> str:
    """Return best text for a section: own text if long enough, else merge children."""
    text = sec.text.strip()
    if len(text) >= SHORT_TEXT_THRESHOLD:
        return text
    children_text = _merge_children_text(sections, sec.heading)
    if children_text:
        prefix = (text + "\n\n") if text else ""
        return prefix + children_text
    return text


def main() -> None:
    GT_DIR.mkdir(exist_ok=True)

    # Cache detect_sections() per paper (expensive GROBID call)
    paper_sections: dict[str, list] = {}

    print("=" * 70)
    print("  Experiment 2 — Generate Ground Truth Drafts")
    print("=" * 70)

    for paper_id, keywords, section_type, slug in TARGETS:
        pdf_name = PAPER_PDF_MAP[paper_id]
        pdf_path = REF_PAPERS / pdf_name
        out_path = GT_DIR / _slug_to_filename(slug)

        if not pdf_path.exists():
            print(f"\n[SKIP] {paper_id} — PDF not found: {pdf_name}")
            continue

        # Run GROBID (cached per paper)
        if paper_id not in paper_sections:
            print(
                f"\nProcessing {paper_id} ({PAPER_TITLE_MAP[paper_id]}) ...", flush=True
            )
            try:
                paper_sections[paper_id] = detect_sections(pdf_path)
            except Exception as e:
                print(f"  [ERROR] {e}")
                paper_sections[paper_id] = []

        sections = paper_sections[paper_id]
        sec = _find_section(sections, keywords)

        if sec is None:
            print(f"  [MISS] '{keywords}' not found in {paper_id}")
            text = ""
            heading = " + ".join(keywords)
        else:
            heading = sec.heading
            text = _resolve_text(sections, sec)
            print(f"  ✓  {heading}  ({len(text)} chars)")

        # Truncate
        text_trimmed = text[:TEXT_MAX_CHARS]
        if len(text) > TEXT_MAX_CHARS:
            text_trimmed += "..."

        draft = {
            "paper_id": paper_id,
            "paper_title": PAPER_TITLE_MAP[paper_id],
            "section_heading": heading,
            "section_type": section_type,
            "text": text_trimmed,
            "verified": False,
            "note": "Fill in entities and relations, then set verified=true.",
            "entities": [],
            "relations": [],
        }

        out_path.write_text(
            json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\n{'=' * 70}")
    print(f"  Draft files saved to: {GT_DIR}")
    print()
    print(
        "  Entity types:  Contribution | Baseline | Concept | Metric | Artifact | Context"
    )
    print(
        "  Relation types: proposes | outperforms | uses | evaluated_on | measures | related_to"
    )
    print()
    print("  Next steps:")
    print("   1. Open each JSON in ground_truth/")
    print("   2. Read the 'text' field, fill in 'entities' and 'relations'")
    print("   3. Set verified=true when done")


if __name__ == "__main__":
    main()
