"""
Benchmark runner for Experiment 1: Section Detection.

Usage:
    python experiments/section_detection/benchmark.py --approach B
    python experiments/section_detection/benchmark.py --approach A
    python experiments/section_detection/benchmark.py --approach A2
    python experiments/section_detection/benchmark.py --approach both    (A vs B)
    python experiments/section_detection/benchmark.py --approach all     (A2 vs A vs B)

Results saved to approach_b/results_b.json, approach_a/results_a.json,
or approach_a/results_a2.json.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_DIR    = REPO_ROOT / "experiments" / "section_detection" / "ground_truth"
GROUND_TRUTH_A2_DIR = REPO_ROOT / "experiments" / "section_detection" / "ground_truth_a2"
REF_PAPERS_DIR = REPO_ROOT / "ref_papers"
EXP_DIR = REPO_ROOT / "experiments" / "section_detection"

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
    s = s.strip().translate(_LIGATURES)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"(\.)([A-Z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def compute_section_metrics(gt: list[dict], detected: list[dict]) -> dict:
    gt_norm  = [_normalize(s["heading"]) for s in gt]
    det_norm = [_normalize(s["heading"]) for s in detected]

    gt_set  = set(gt_norm)
    det_set = set(det_norm)

    tp = len(gt_set & det_set)
    fp = len(det_set - gt_set)
    fn = len(gt_set - det_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    gt_type_map  = {_normalize(s["heading"]): s["type"] for s in gt}
    det_type_map = {_normalize(s["heading"]): s["type"] for s in detected}
    correct_hits = list(gt_set & det_set)
    type_correct = sum(1 for h in correct_hits if gt_type_map.get(h) == det_type_map.get(h))
    type_accuracy = type_correct / len(correct_hits) if correct_hits else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision":    round(precision, 3),
        "recall":       round(recall, 3),
        "f1":           round(f1, 3),
        "type_accuracy": round(type_accuracy, 3),
        "missed_sections": sorted(gt_set - det_set),
        "extra_sections":  sorted(det_set - gt_set),
        "wrong_type": [
            {"heading": h, "gt": gt_type_map[h], "detected": det_type_map[h]}
            for h in correct_hits
            if gt_type_map.get(h) != det_type_map.get(h)
        ],
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(approach: str) -> list[dict]:
    if approach == "A2":
        sys.path.insert(0, str(EXP_DIR / "approach_a"))
        from approach_a2_grobid import detect_sections
        label        = "APPROACH A2: GROBID + subsections"
        gt_dir       = GROUND_TRUTH_A2_DIR
        results_file = EXP_DIR / "approach_a" / "results_a2.json"
    elif approach == "A":
        sys.path.insert(0, str(EXP_DIR / "approach_a"))
        from approach_a_grobid import detect_sections
        label        = "APPROACH A: GROBID top-level only"
        gt_dir       = GROUND_TRUTH_DIR
        results_file = EXP_DIR / "approach_a" / "results_a.json"
    else:  # B
        sys.path.insert(0, str(EXP_DIR / "approach_b"))
        from approach_b_heuristic import detect_sections
        label        = "APPROACH B: pdfplumber + regex heuristics"
        gt_dir       = GROUND_TRUTH_DIR
        results_file = EXP_DIR / "approach_b" / "results_b.json"

    gt_files = sorted(gt_dir.glob("*.json"))
    gt_files = [f for f in gt_files if f.name != "README.md"]

    all_results = []
    print("\n" + "=" * 80)
    print(f"  EXPERIMENT 1 — {label}")
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
            print(f"\n[SKIP] {paper_id} — PDF not found")
            continue

        print(f"\n── {gt_data['title']} ({paper_id}) ──")
        print(f"   Format: {gt_data.get('format', '?')}")

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
            "title":    gt_data["title"],
            "format":   gt_data.get("format"),
            "metrics":  metrics,
            "detected": detected_dicts,
        })

    if all_results:
        _print_summary(all_results)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  Full results saved → {results_file.relative_to(REPO_ROOT)}")

    return all_results


def _print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  {'Paper':<40} {'F1':>6} {'Recall':>8} {'Precision':>10} {'TypeAcc':>8}")
    print("  " + "-" * 76)
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['title'][:38]:<40} {m['f1']:>6.1%} {m['recall']:>8.1%} {m['precision']:>10.1%} {m['type_accuracy']:>8.1%}")
    mean_f1  = sum(r["metrics"]["f1"] for r in all_results) / len(all_results)
    mean_rec = sum(r["metrics"]["recall"] for r in all_results) / len(all_results)
    mean_pre = sum(r["metrics"]["precision"] for r in all_results) / len(all_results)
    mean_typ = sum(r["metrics"]["type_accuracy"] for r in all_results) / len(all_results)
    print("  " + "-" * 76)
    print(f"  {'MEAN':<40} {mean_f1:>6.1%} {mean_rec:>8.1%} {mean_pre:>10.1%} {mean_typ:>8.1%}")
    print()


def _print_comparison(results_list_map: dict[str, list[dict]]) -> None:
    """Print side-by-side comparison. Input: {approach: [full result dicts]}."""
    approaches = list(results_list_map.keys())

    # Build {approach: {paper_id: metrics}}
    metrics_map = {
        a: {r["paper_id"]: r["metrics"] for r in results_list_map[a]}
        for a in approaches
    }
    # Build title lookup from first approach
    id_to_title = {r["paper_id"]: r["title"][:34] for r in results_list_map[approaches[0]]}
    paper_ids   = [r["paper_id"] for r in results_list_map[approaches[0]]]

    header = f"  {'Paper':<36}" + "".join(f" {a+': F1':>8}" for a in approaches) + "  Winner"
    print("\n" + "=" * 80)
    print(f"  SIDE-BY-SIDE COMPARISON: {' vs '.join(approaches)}")
    print("=" * 80)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for pid in paper_ids:
        title  = id_to_title.get(pid, pid)
        scores = {a: metrics_map[a].get(pid, {}).get("f1") for a in approaches}
        row    = f"  {title:<36}" + "".join(
            f" {scores[a]:>8.1%}" if scores[a] is not None else f" {'N/A':>8}"
            for a in approaches
        )
        valid = {a: s for a, s in scores.items() if s is not None}
        best  = max(valid.values()) if valid else None
        winners = [a for a, s in valid.items() if s == best] if best is not None else ["?"]
        print(row + f"  {'/'.join(winners)}")

    print("  " + "-" * (len(header) - 2))
    means = {
        a: sum(r["f1"] for r in metrics_map[a].values()) / len(metrics_map[a])
        for a in approaches
    }
    mean_row = f"  {'MEAN':<36}" + "".join(f" {means[a]:>8.1%}" for a in approaches)
    best_mean = max(means.values())
    mean_winners = [a for a, s in means.items() if s == best_mean]
    print(mean_row + f"  {'/'.join(mean_winners)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(description="Section detection benchmark")
    parser.add_argument(
        "--approach", choices=["A", "B", "A2", "both", "all"], default="B",
        help="Which approach to benchmark (default: B)"
    )
    args = parser.parse_args()

    if args.approach == "both":
        rb = run_benchmark("B")
        ra = run_benchmark("A")
        if rb and ra:
            _print_comparison({"B": rb, "A": ra})

    elif args.approach == "all":
        rb  = run_benchmark("B")
        ra  = run_benchmark("A")
        ra2 = run_benchmark("A2")
        results_list_map = {}
        if rb:  results_list_map["B"]  = rb
        if ra:  results_list_map["A"]  = ra
        if ra2: results_list_map["A2"] = ra2
        if len(results_list_map) >= 2:
            _print_comparison(results_list_map)

    else:
        run_benchmark(args.approach)
