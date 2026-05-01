"""
Benchmark runner for Experiment 1: Section Detection.

Usage:
    python experiments/section_detection/benchmark.py

Compares detected sections (Approach B) against ground truth JSONs.
Results are printed as a table and saved to experiments/section_detection/results_b.json.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows (cp950 can't encode academic ligatures like ﬁ)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_DIR = REPO_ROOT / "experiments" / "section_detection" / "ground_truth"
REF_PAPERS_DIR = REPO_ROOT / "ref_papers"
RESULTS_FILE = REPO_ROOT / "experiments" / "section_detection" / "results_b.json"

# Map ground-truth paper_id to PDF filename
PAPER_PDF_MAP = {
    "1412.6980":        "1412.6980v9.pdf",
    "1512.03385":       "1512.03385v1.pdf",
    "1706.03762":       "1706.03762v7.pdf",
    "1810.04805":       "1810.04805v2.pdf",
    "NIPS-2012-alexnet": "NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})


def _normalize(s: str) -> str:
    """Normalize heading text for comparison.
    Space-insertion rules must run BEFORE lowercasing so camelCase splits work."""
    s = s.strip().translate(_LIGATURES)           # "overﬁtting" → "overfitting"
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)   # "1Introduction" → "1 Introduction"
    s = re.sub(r"(\.)([A-Z])", r"\1 \2", s)       # "1.Introduction" → "1. Introduction"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)    # "RelatedWork" → "Related Work"
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def compute_section_metrics(
    gt: list[dict], detected: list[dict]
) -> dict:
    gt_norm = [_normalize(s["heading"]) for s in gt]
    det_norm = [_normalize(s["heading"]) for s in detected]

    gt_set = set(gt_norm)
    det_set = set(det_norm)

    tp = len(gt_set & det_set)
    fp = len(det_set - gt_set)
    fn = len(gt_set - det_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Type accuracy: among correctly detected headings, how many have the right type?
    gt_type_map = {_normalize(s["heading"]): s["type"] for s in gt}
    det_type_map = {_normalize(s["heading"]): s["type"] for s in detected}
    correct_hits = [h for h in (gt_set & det_set)]
    type_correct = sum(
        1 for h in correct_hits if gt_type_map.get(h) == det_type_map.get(h)
    )
    type_accuracy = type_correct / len(correct_hits) if correct_hits else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "type_accuracy": round(type_accuracy, 3),
        "missed_sections":   sorted(gt_set - det_set),
        "extra_sections":    sorted(det_set - gt_set),
        "wrong_type": [
            {"heading": h, "gt": gt_type_map[h], "detected": det_type_map[h]}
            for h in correct_hits
            if gt_type_map.get(h) != det_type_map.get(h)
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark():
    from approach_b_heuristic import detect_sections

    gt_files = sorted(GROUND_TRUTH_DIR.glob("*.json"))
    gt_files = [f for f in gt_files if f.name != "README.md"]

    all_results = []
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1 — APPROACH B: pdfplumber + regex heuristics")
    print("=" * 80)

    for gt_file in gt_files:
        with open(gt_file, encoding="utf-8") as f:
            gt_data = json.load(f)

        if not gt_data.get("verified"):
            print(f"\n[SKIP] {gt_file.name} — not verified")
            continue

        paper_id = gt_data["paper_id"]
        pdf_name = PAPER_PDF_MAP.get(paper_id)
        if not pdf_name:
            print(f"\n[SKIP] {paper_id} — no PDF mapping")
            continue

        pdf_path = REF_PAPERS_DIR / pdf_name
        if not pdf_path.exists():
            print(f"\n[SKIP] {paper_id} — PDF not found: {pdf_path}")
            continue

        print(f"\n── {gt_data['title']} ({paper_id}) ──")
        print(f"   Format: {gt_data.get('format', '?')}")
        print(f"   PDF:    {pdf_name}")

        try:
            detected = detect_sections(pdf_path)
        except Exception as e:
            print(f"   [ERROR] {e}")
            continue

        detected_dicts = [{"heading": s.heading, "type": s.type} for s in detected]
        metrics = compute_section_metrics(gt_data["sections"], detected_dicts)

        print(f"   GT sections:       {len(gt_data['sections'])}")
        print(f"   Detected sections: {len(detected_dicts)}")
        print(f"   Precision:  {metrics['precision']:.1%}")
        print(f"   Recall:     {metrics['recall']:.1%}")
        print(f"   F1:         {metrics['f1']:.1%}")
        print(f"   Type Acc:   {metrics['type_accuracy']:.1%}")

        if metrics["missed_sections"]:
            print(f"   MISSED: {metrics['missed_sections']}")
        if metrics["extra_sections"]:
            print(f"   EXTRA:  {metrics['extra_sections']}")
        if metrics["wrong_type"]:
            print(f"   WRONG TYPE: {metrics['wrong_type']}")

        all_results.append({
            "paper_id": paper_id,
            "title": gt_data["title"],
            "format": gt_data.get("format"),
            "metrics": metrics,
            "detected": detected_dicts,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)
        print(f"  {'Paper':<40} {'F1':>6} {'Recall':>8} {'Precision':>10} {'TypeAcc':>8}")
        print("  " + "-" * 76)
        for r in all_results:
            m = r["metrics"]
            title_short = r["title"][:38]
            print(f"  {title_short:<40} {m['f1']:>6.1%} {m['recall']:>8.1%} {m['precision']:>10.1%} {m['type_accuracy']:>8.1%}")

        mean_f1       = sum(r["metrics"]["f1"] for r in all_results) / len(all_results)
        mean_recall   = sum(r["metrics"]["recall"] for r in all_results) / len(all_results)
        mean_precision= sum(r["metrics"]["precision"] for r in all_results) / len(all_results)
        mean_type     = sum(r["metrics"]["type_accuracy"] for r in all_results) / len(all_results)
        print("  " + "-" * 76)
        print(f"  {'MEAN':<40} {mean_f1:>6.1%} {mean_recall:>8.1%} {mean_precision:>10.1%} {mean_type:>8.1%}")
        print()

        # Save full results
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  Full results saved → {RESULTS_FILE.relative_to(REPO_ROOT)}")

    return all_results


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run_benchmark()
