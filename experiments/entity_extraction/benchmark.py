"""
Experiment 2 benchmark: evaluate entity extraction results against ground truth.

Reads results_{a,b,c,d}.json from each approach folder, computes precision/recall/F1
for entities and relations, then prints a comparison table.

Usage:
    # Run a specific approach first, then evaluate:
    python experiments/entity_extraction/approach_a/approach_a_ollama.py
    python experiments/entity_extraction/benchmark.py --approach A

    # Evaluate all available results:
    python experiments/entity_extraction/benchmark.py --approach all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

APPROACH_FILES = {
    "A": EXP_DIR / "approach_a" / "results_a.json",
    "B": EXP_DIR / "approach_b" / "results_b.json",
    "C": EXP_DIR / "approach_c" / "results_c.json",
    "D": EXP_DIR / "approach_d" / "results_d.json",
    "E": EXP_DIR / "approach_e" / "results_e.json",
    "F": EXP_DIR / "approach_f" / "results_f.json",
}


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _norm(s) -> str:
    if isinstance(s, list):
        s = ", ".join(str(x) for x in s)
    return str(s).strip().lower()


def _entity_key(e: dict) -> tuple[str, str]:
    return (_norm(e.get("name", "")), _norm(e.get("type", "")))


def _relation_key(r: dict) -> tuple[str, str, str]:
    return (
        _norm(r.get("head", "")),
        _norm(r.get("tail", "")),
        _norm(r.get("type", "")),
    )


def _prf(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def _type_accuracy(pred_entities: list[dict], gold_entities: list[dict]) -> float:
    """Among entities whose name matches gold, what fraction has the correct type."""
    gold_by_name = {_norm(e["name"]): _norm(e["type"]) for e in gold_entities}
    correct = total = 0
    for e in pred_entities:
        name = _norm(e.get("name", ""))
        if name in gold_by_name:
            total += 1
            if _norm(e.get("type", "")) == gold_by_name[name]:
                correct += 1
    return round(correct / total, 4) if total > 0 else 0.0


def _fuzzy_entity_prf(
    pred_entities: list[dict], gold_entities: list[dict]
) -> tuple[float, float, float]:
    """Entity P/R/F1 with substring name matching; type must still match exactly."""
    gold_items = [(_norm(e["name"]), _norm(e["type"])) for e in gold_entities]
    pred_items = [
        (_norm(e.get("name", "")), _norm(e.get("type", ""))) for e in pred_entities
    ]

    def _names_match(a: str, b: str) -> bool:
        return a == b or (len(a) > 1 and a in b) or (len(b) > 1 and b in a)

    matched_gold: set[int] = set()
    tp = 0
    for pname, ptype in pred_items:
        for i, (gname, gtype) in enumerate(gold_items):
            if i not in matched_gold and ptype == gtype and _names_match(pname, gname):
                matched_gold.add(i)
                tp += 1
                break

    fp = len(pred_items) - tp
    fn = len(gold_items) - len(matched_gold)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def _contrib_baseline_accuracy(
    pred_entities: list[dict], gold_entities: list[dict]
) -> float:
    """Accuracy specifically on Contribution vs Baseline classification (exact name match)."""
    gold_cb = {
        _norm(e["name"]): _norm(e["type"])
        for e in gold_entities
        if _norm(e["type"]) in ("contribution", "baseline")
    }
    if not gold_cb:
        return 1.0
    correct = 0
    for e in pred_entities:
        name = _norm(e.get("name", ""))
        ptype = _norm(e.get("type", ""))
        if name in gold_cb and ptype in ("contribution", "baseline"):
            if ptype == gold_cb[name]:
                correct += 1
    return round(correct / len(gold_cb), 4)


def _fuzzy_contrib_baseline_accuracy(
    pred_entities: list[dict], gold_entities: list[dict]
) -> float:
    """CB accuracy with fuzzy name matching (substring containment)."""
    gold_cb = [
        (_norm(e["name"]), _norm(e["type"]))
        for e in gold_entities
        if _norm(e["type"]) in ("contribution", "baseline")
    ]
    if not gold_cb:
        return 1.0

    def _names_match(a: str, b: str) -> bool:
        return a == b or (len(a) > 1 and a in b) or (len(b) > 1 and b in a)

    matched_gold: set[int] = set()
    correct = 0
    for e in pred_entities:
        pname = _norm(e.get("name", ""))
        ptype = _norm(e.get("type", ""))
        if ptype not in ("contribution", "baseline"):
            continue
        for i, (gname, gtype) in enumerate(gold_cb):
            if i not in matched_gold and _names_match(pname, gname):
                matched_gold.add(i)
                if ptype == gtype:
                    correct += 1
                break

    return round(correct / len(gold_cb), 4)


# ---------------------------------------------------------------------------
# Per-chunk evaluation
# ---------------------------------------------------------------------------


def evaluate_chunk(chunk: dict) -> dict:
    gt = chunk.get("ground_truth", {})
    pred = chunk.get("extracted") or {}

    gold_entities = gt.get("entities", [])
    gold_relations = gt.get("relations", [])
    pred_entities = pred.get("entities", []) if isinstance(pred, dict) else []
    pred_relations = pred.get("relations", []) if isinstance(pred, dict) else []

    gold_e_set = {_entity_key(e) for e in gold_entities}
    pred_e_set = {_entity_key(e) for e in pred_entities}
    gold_r_set = {_relation_key(r) for r in gold_relations}
    pred_r_set = {_relation_key(r) for r in pred_relations}

    ep, er, ef = _prf(pred_e_set, gold_e_set)
    rp, rr, rf = _prf(pred_r_set, gold_r_set)
    _fp, _fr, ff = _fuzzy_entity_prf(pred_entities, gold_entities)

    return {
        "paper_id": chunk["paper_id"],
        "section_heading": chunk["section_heading"],
        "section_type": chunk["section_type"],
        "parse_ok": chunk.get("parse_ok", False),
        "elapsed_s": chunk.get("elapsed_s", 0),
        "entity_P": ep,
        "entity_R": er,
        "entity_F1": ef,
        "fuzzy_entity_F1": ff,
        "type_accuracy": _type_accuracy(pred_entities, gold_entities),
        "cb_accuracy": _contrib_baseline_accuracy(pred_entities, gold_entities),
        "fuzzy_cb_accuracy": _fuzzy_contrib_baseline_accuracy(
            pred_entities, gold_entities
        ),
        "relation_P": rp,
        "relation_R": rr,
        "relation_F1": rf,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def aggregate(evals: list[dict]) -> dict:
    n = len(evals)
    if n == 0:
        return {}

    def avg(key: str) -> float:
        return round(sum(e[key] for e in evals) / n, 4)

    parse_rate = sum(1 for e in evals if e["parse_ok"]) / n

    return {
        "n_chunks": n,
        "parse_rate": round(parse_rate, 4),
        "entity_P": avg("entity_P"),
        "entity_R": avg("entity_R"),
        "entity_F1": avg("entity_F1"),
        "fuzzy_entity_F1": avg("fuzzy_entity_F1"),
        "type_acc": avg("type_accuracy"),
        "cb_acc": avg("cb_accuracy"),
        "fuzzy_cb_acc": avg("fuzzy_cb_accuracy"),
        "relation_F1": avg("relation_F1"),
        "avg_elapsed": avg("elapsed_s"),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "parse_rate": 0.90,
    "entity_F1": 0.60,
    "type_acc": 0.70,
    "cb_acc": 0.80,
}


def _tick(val: float, key: str) -> str:
    threshold = THRESHOLDS.get(key)
    if threshold is None:
        return ""
    return "[PASS]" if val >= threshold else "[FAIL]"


def print_summary(label: str, agg: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  Approach {label}")
    print(f"{'─'*50}")
    print(f"  Chunks evaluated : {agg['n_chunks']}")
    print(
        f"  JSON parse rate  : {agg['parse_rate']*100:5.1f}%  {_tick(agg['parse_rate'], 'parse_rate')}  (threshold >= 90%)"
    )
    print(
        f"  Entity F1 (exact): {agg['entity_F1']*100:5.1f}%  {_tick(agg['entity_F1'],  'entity_F1')}  (threshold >= 60%)"
    )
    print(
        f"  Entity F1 (fuzzy): {agg['fuzzy_entity_F1']*100:5.1f}%  (substring match, no threshold)"
    )
    print(
        f"  Type accuracy    : {agg['type_acc']*100:5.1f}%  {_tick(agg['type_acc'],    'type_acc')}  (threshold >= 70%)"
    )
    print(
        f"  CB acc (exact)   : {agg['cb_acc']*100:5.1f}%  {_tick(agg['cb_acc'],       'cb_acc')}  (threshold >= 80%)"
    )
    print(
        f"  CB acc (fuzzy)   : {agg['fuzzy_cb_acc']*100:5.1f}%  (substring name match)"
    )
    print(f"  Relation F1      : {agg['relation_F1']*100:5.1f}%  (observation only)")
    print(f"  Avg latency      : {agg['avg_elapsed']:.1f}s / chunk")


def print_comparison(results: dict[str, dict]) -> None:
    headers = [
        "Approach",
        "Parse%",
        "Exact F1",
        "Fuzzy F1",
        "Type acc",
        "CB acc",
        "CB(fuz)",
        "Rel F1",
        "Latency",
    ]
    rows = []
    for label, agg in results.items():
        rows.append(
            [
                label,
                f"{agg['parse_rate']*100:.1f}%",
                f"{agg['entity_F1']*100:.1f}%",
                f"{agg['fuzzy_entity_F1']*100:.1f}%",
                f"{agg['type_acc']*100:.1f}%",
                f"{agg['cb_acc']*100:.1f}%",
                f"{agg['fuzzy_cb_acc']*100:.1f}%",
                f"{agg['relation_F1']*100:.1f}%",
                f"{agg['avg_elapsed']:.1f}s",
            ]
        )

    col_w = [
        max(len(h), max(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)
    ]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f" {{:<{w-1}}}" for w in col_w) + "|"

    print(f"\n{'='*60}")
    print("  Comparison Table")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 2 approaches.")
    parser.add_argument(
        "--approach",
        default="all",
        help="Which approach(es) to evaluate: A, B, C, D, or all (default: all)",
    )
    args = parser.parse_args()

    targets = (
        list(APPROACH_FILES.keys())
        if args.approach.lower() == "all"
        else [a.strip().upper() for a in args.approach.split(",")]
    )

    comparison: dict[str, dict] = {}

    for label in targets:
        path = APPROACH_FILES.get(label)
        if path is None:
            print(f"Unknown approach: {label}")
            continue
        if not path.exists():
            print(f"[{label}] results not found: {path}")
            continue

        chunks = json.loads(path.read_text(encoding="utf-8"))
        evals = [evaluate_chunk(c) for c in chunks]
        agg = aggregate(evals)
        comparison[label] = agg
        print_summary(label, agg)

        # save per-chunk eval
        eval_path = path.parent / f"eval_{label.lower()}.json"
        eval_path.write_text(
            json.dumps(evals, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Per-chunk eval → {eval_path}")

    if len(comparison) > 1:
        print_comparison(comparison)


if __name__ == "__main__":
    main()
