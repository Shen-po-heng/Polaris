"""
Experiment 3 Benchmark — Compare deduplication strategies A / B / C
against ground truth clusters.

Metric: pairwise precision / recall / F1
  - Positive pair: two instances that belong to the same gold cluster
  - A strategy's prediction: two instances that end up in the same predicted cluster
"""

import json
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RESULTS_F = Path(__file__).parent.parent / "entity_extraction" / "approach_f" / "results_f.json"
GT_PATH = Path(__file__).parent / "ground_truth_clusters.json"
EXP_DIR = Path(__file__).parent

THRESHOLDS = [0.80, 0.85, 0.90, 0.92, 0.95]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ALIAS_MAP: dict[str, str] = {
    "imagenet 2012": "ImageNet",
    "imagenet2012": "ImageNet",
    "ilsvrc-2010": "ImageNet",
    "ilsvrc 2010": "ImageNet",
    "top-1 error": "top-1 error rate",
    "top-5 error": "top-5 error rate",
    "training error": "training error rate",
    "validation error": "validation error rate",
    "fisher vectors": "Fisher Vector",
    "residual networks": "ResNet",
    "deep residual networks": "ResNet",
    "deep residual learning": "ResNet",
    "non-line-of-sight": "NLOS",
    "non line of sight": "NLOS",
    "non-los scenario": "NLOS",
    "convolutional neural network": "convolutional neural networks",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _apply_alias(name: str) -> str:
    return ALIAS_MAP.get(_norm(name), name)


def load_entities() -> list[dict]:
    with open(RESULTS_F, encoding="utf-8") as f:
        chunks = json.load(f)
    entities = []
    for chunk in chunks:
        for e in chunk.get("extracted", {}).get("entities", []):
            entities.append(
                {
                    "name": e["name"],
                    "type": e["type"],
                    "paper_id": chunk["paper_id"],
                    "section_type": chunk["section_type"],
                }
            )
    return entities


def load_ground_truth() -> list[set]:
    """Return list of frozensets; each frozenset is a gold merge cluster (instance keys)."""
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    clusters = []
    for cluster in gt:
        keys = frozenset(
            (i["name"], i["paper_id"], i["section_type"]) for i in cluster["instances"]
        )
        if len(keys) > 1:
            clusters.append(keys)
    return clusters


def instance_key(e: dict) -> tuple:
    return (e["name"], e["paper_id"], e["section_type"])


def clusters_to_pairs(clusters: list[list[dict]]) -> set[frozenset]:
    """Convert cluster list → set of positive instance-key pairs."""
    pairs: set[frozenset] = set()
    for cluster in clusters:
        keys = [instance_key(i) for i in cluster]
        for a, b in combinations(keys, 2):
            pairs.add(frozenset([a, b]))
    return pairs


def gt_to_pairs(gt_clusters: list[set]) -> set[frozenset]:
    pairs: set[frozenset] = set()
    for cluster in gt_clusters:
        keys = list(cluster)
        for a, b in combinations(keys, 2):
            pairs.add(frozenset([a, b]))
    return pairs


def prf(pred_pairs: set, gold_pairs: set) -> tuple[float, float, float]:
    if not pred_pairs:
        return 0.0, 0.0, 0.0
    tp = len(pred_pairs & gold_pairs)
    precision = tp / len(pred_pairs)
    recall = tp / len(gold_pairs) if gold_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Strategy implementations (inline for benchmark)
# ---------------------------------------------------------------------------

def strategy_a(entities: list[dict]) -> list[list[dict]]:
    """Exact match on normalized name."""
    cluster_map: dict[str, list] = {}
    for e in entities:
        key = _norm(e["name"])
        cluster_map.setdefault(key, []).append(e)
    return [v for v in cluster_map.values() if len(v) > 1]


def strategy_b(entities: list[dict], threshold: float, embeddings: np.ndarray) -> list[list[dict]]:
    """Exact match first, then embedding similarity merge."""
    cluster_map: dict[str, list] = {}
    for e in entities:
        key = _norm(e["name"])
        cluster_map.setdefault(key, []).append(e)

    keys = list(cluster_map.keys())
    names = [cluster_map[k][0]["name"] for k in keys]

    # embed names
    model = _get_model()
    embs = model.encode(names, normalize_embeddings=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.maximum(norms, 1e-9)
    sim = normed @ normed.T

    n = len(keys)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in combinations(range(n), 2):
        if sim[i, j] >= threshold:
            type_i = cluster_map[keys[i]][0]["type"]
            type_j = cluster_map[keys[j]][0]["type"]
            if type_i == type_j:
                parent[find(i)] = find(j)

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).extend(cluster_map[keys[i]])

    return [v for v in groups.values() if len(v) > 1]


def strategy_c(entities: list[dict]) -> list[list[dict]]:
    """Alias dict + exact match."""
    cluster_map: dict[str, list] = {}
    for e in entities:
        canonical = _apply_alias(e["name"])
        key = _norm(canonical)
        cluster_map.setdefault(key, []).append(e)
    return [v for v in cluster_map.values() if len(v) > 1]


_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    entities = load_entities()
    gt_clusters = load_ground_truth()
    gold_pairs = gt_to_pairs(gt_clusters)

    print("=" * 60)
    print("Experiment 3 — Deduplication Benchmark")
    print("=" * 60)
    print(f"Entity instances : {len(entities)}")
    print(f"Gold clusters    : {len(gt_clusters)}")
    print(f"Gold pairs       : {len(gold_pairs)}")
    print()

    rows = []

    # Strategy A
    clusters_a = strategy_a(entities)
    pred_pairs_a = clusters_to_pairs(clusters_a)
    p, r, f = prf(pred_pairs_a, gold_pairs)
    unique_a = len({_norm(e["name"]) for e in entities})
    rows.append(("A (exact match)", unique_a, p, r, f))
    print(f"Strategy A — unique nodes: {unique_a}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    # Strategy C (alias + exact)
    clusters_c = strategy_c(entities)
    pred_pairs_c = clusters_to_pairs(clusters_c)
    p, r, f = prf(pred_pairs_c, gold_pairs)
    unique_c = len({_norm(_apply_alias(e["name"])) for e in entities})
    rows.append(("C (alias dict)", unique_c, p, r, f))
    print(f"Strategy C — unique nodes: {unique_c}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    # Strategy B (embedding at various thresholds)
    print()
    best_b = None
    model = _get_model()
    all_names = [e["name"] for e in entities]
    embeddings = model.encode(all_names, normalize_embeddings=False)

    for threshold in THRESHOLDS:
        clusters_b = strategy_b(entities, threshold, embeddings)
        pred_pairs_b = clusters_to_pairs(clusters_b)
        p, r, f = prf(pred_pairs_b, gold_pairs)

        cluster_map_b: dict[str, list] = {}
        for e in entities:
            key = _norm(e["name"])
            cluster_map_b.setdefault(key, []).append(e)
        unique_b = len(cluster_map_b)
        for cl in clusters_b:
            names_in_cluster = {_norm(e["name"]) for e in cl}
            if len(names_in_cluster) > 1:
                unique_b -= len(names_in_cluster) - 1

        label = f"B (emb t={threshold:.2f})"
        rows.append((label, unique_b, p, r, f))
        print(f"Strategy {label} — unique nodes: {unique_b}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

        if best_b is None or f > best_b[4]:
            best_b = (label, unique_b, p, r, f)

    print()
    print("=" * 60)
    print("Summary Table")
    print("=" * 60)
    header = f"{'Strategy':<22} {'Nodes':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print(header)
    print("-" * 55)
    for row in rows:
        print(f"{row[0]:<22} {row[1]:>6} {row[2]:>7.3f} {row[3]:>7.3f} {row[4]:>7.3f}")

    print()
    if best_b:
        print(f"Best strategy B: {best_b[0]}  F1={best_b[4]:.3f}")

    # Save summary
    summary = {
        "gold_clusters": len(gt_clusters),
        "gold_pairs": len(gold_pairs),
        "results": [
            {"strategy": r[0], "unique_nodes": r[1], "precision": r[2], "recall": r[3], "f1": r[4]}
            for r in rows
        ],
    }
    out_path = EXP_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
