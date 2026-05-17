"""Diagnose exact TP/FP/FN for each dedup strategy."""
import json
import re
from itertools import combinations
from pathlib import Path

RESULTS_F = Path(__file__).parent.parent / "entity_extraction" / "approach_f" / "results_f.json"
GT_PATH = Path(__file__).parent / "ground_truth_clusters.json"

ALIAS_MAP = {
    "imagenet 2012": "ImageNet",
    "top-1 error": "top-1 error rate",
    "top-5 error": "top-5 error rate",
    "fisher vectors": "Fisher Vector",
    "convolutional neural network": "convolutional neural networks",
}


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def alias(name: str) -> str:
    return ALIAS_MAP.get(norm(name), name)


def load_entities():
    with open(RESULTS_F, encoding="utf-8") as f:
        chunks = json.load(f)
    entities = []
    for chunk in chunks:
        for e in chunk.get("extracted", {}).get("entities", []):
            entities.append({
                "name": e["name"], "type": e["type"],
                "paper_id": chunk["paper_id"], "section_type": chunk["section_type"],
            })
    return entities


def load_gold_pairs():
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    pairs = set()
    for cluster in gt:
        keys = [(i["name"], i["paper_id"], i["section_type"]) for i in cluster["instances"]]
        for a, b in combinations(keys, 2):
            pairs.add(frozenset([a, b]))
    return pairs


def get_pairs(entities, key_fn):
    cluster_map = {}
    for e in entities:
        k = key_fn(e)
        cluster_map.setdefault(k, []).append(e)
    pairs = set()
    for group in cluster_map.values():
        if len(group) > 1:
            keys = [(e["name"], e["paper_id"], e["section_type"]) for e in group]
            for a, b in combinations(keys, 2):
                pairs.add(frozenset([a, b]))
    return pairs


def show_pair(p):
    a, b = list(p)
    return f'  [{a[1]}/{a[2]}] "{a[0]}"  <->  [{b[1]}/{b[2]}] "{b[0]}"'


def main():
    entities = load_entities()
    gold_pairs = load_gold_pairs()

    pairs_a = get_pairs(entities, lambda e: norm(e["name"]))
    pairs_c = get_pairs(entities, lambda e: norm(alias(e["name"])))

    print("=" * 60)
    print("Strategy A — Exact Match")
    print("=" * 60)
    print(f"Predicted pairs : {len(pairs_a)}")
    print(f"Gold pairs      : {len(gold_pairs)}")
    tp_a = pairs_a & gold_pairs
    fp_a = pairs_a - gold_pairs
    fn_a = gold_pairs - pairs_a
    print(f"TP={len(tp_a)}  FP={len(fp_a)}  FN={len(fn_a)}")

    print("\nTRUE POSITIVES (correctly merged):")
    for p in sorted(tp_a, key=str):
        print(show_pair(p))

    print("\nFALSE NEGATIVES (gold pairs missed by A):")
    for p in sorted(fn_a, key=str):
        print(show_pair(p))

    print()
    print("=" * 60)
    print("Strategy C — Alias Dict + Exact Match")
    print("=" * 60)
    print(f"Predicted pairs : {len(pairs_c)}")
    tp_c = pairs_c & gold_pairs
    fp_c = pairs_c - gold_pairs
    fn_c = gold_pairs - pairs_c
    print(f"TP={len(tp_c)}  FP={len(fp_c)}  FN={len(fn_c)}")

    new_tp = (pairs_c & gold_pairs) - (pairs_a & gold_pairs)
    print(f"\nNEW TRUE POSITIVES added by alias dict ({len(new_tp)}):")
    for p in sorted(new_tp, key=str):
        print(show_pair(p))

    print(f"\nFALSE POSITIVES — wrongly merged by C ({len(fp_c)}):")
    for p in sorted(fp_c, key=str):
        print(show_pair(p))

    print(f"\nFALSE NEGATIVES — still missed by C ({len(fn_c)}):")
    for p in sorted(fn_c, key=str):
        print(show_pair(p))


if __name__ == "__main__":
    main()
