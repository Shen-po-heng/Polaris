"""
Experiment 3 — Strategy B: Embedding Similarity Deduplication

1. Start from Strategy A (exact match) clusters.
2. Embed canonical names using sentence-transformers.
3. Merge clusters whose cosine similarity exceeds a threshold.
4. Sweep thresholds [0.80, 0.85, 0.90, 0.92, 0.95] to find optimal.
"""

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RESULTS_F = Path(__file__).parent.parent.parent / "entity_extraction" / "approach_f" / "results_f.json"
OUTPUT = Path(__file__).parent / "results_b.json"

THRESHOLDS = [0.80, 0.85, 0.90, 0.92, 0.95]
MODEL_NAME = "all-MiniLM-L6-v2"


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def load_entities(results_path: Path) -> list[dict]:
    with open(results_path, encoding="utf-8") as f:
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
                    "norm_name": _normalize(e["name"]),
                }
            )
    return entities


def exact_match_clusters(entities: list[dict]) -> list[dict]:
    """Strategy A as starting point."""
    cluster_map: dict[str, dict] = {}
    for e in entities:
        key = e["norm_name"]
        if key not in cluster_map:
            cluster_map[key] = {
                "canonical_name": e["name"],
                "canonical_type": e["type"],
                "norm_name": key,
                "instances": [],
            }
        cluster_map[key]["instances"].append(
            {"name": e["name"], "paper_id": e["paper_id"], "section_type": e["section_type"]}
        )
    return list(cluster_map.values())


def merge_by_similarity(clusters: list[dict], threshold: float, embeddings: np.ndarray) -> list[dict]:
    """Union-find merge: combine clusters whose embedding cosine sim >= threshold."""
    n = len(clusters)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    # Compute pairwise similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-9)
    sim_matrix = normed @ normed.T

    for i, j in combinations(range(n), 2):
        if sim_matrix[i, j] >= threshold:
            # Only merge if types are compatible (same type, or both are same category)
            type_i = clusters[i]["canonical_type"]
            type_j = clusters[j]["canonical_type"]
            if type_i == type_j:
                union(i, j)

    # Collect merged groups
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged: list[dict] = []
    for indices in groups.values():
        # Pick the canonical name as the most common / first occurrence
        all_instances = []
        for idx in indices:
            all_instances.extend(clusters[idx]["instances"])
        representative = clusters[indices[0]]
        merged.append(
            {
                "canonical_name": representative["canonical_name"],
                "canonical_type": representative["canonical_type"],
                "instances": all_instances,
                "merged_from": [clusters[idx]["canonical_name"] for idx in indices],
            }
        )
    return merged


def main() -> None:
    entities = load_entities(RESULTS_F)
    clusters = exact_match_clusters(entities)

    print(f"Total entity instances   : {len(entities)}")
    print(f"After exact match (A)    : {len(clusters)} unique nodes")
    print(f"Embedding model          : {MODEL_NAME}")
    print()

    model = SentenceTransformer(MODEL_NAME)
    names = [c["canonical_name"] for c in clusters]
    embeddings = model.encode(names, normalize_embeddings=False)

    threshold_results = []
    for threshold in THRESHOLDS:
        merged = merge_by_similarity(clusters, threshold, embeddings)
        n_merged_clusters = sum(1 for c in merged if len(c["instances"]) > len(set(
            (i["paper_id"], i["section_type"]) for i in c["instances"]
        )) or len(c.get("merged_from", [])) > 1)
        print(f"  threshold={threshold:.2f} → {len(merged)} unique nodes  "
              f"(reduced by {len(clusters) - len(merged)})")

        # Show what was merged beyond Strategy A
        new_merges = [c for c in merged if len(c.get("merged_from", [])) > 1]
        for c in new_merges:
            print(f"    merged: {c['merged_from']} → '{c['canonical_name']}'")

        threshold_results.append(
            {
                "threshold": threshold,
                "unique_nodes": len(merged),
                "reduction_from_A": len(clusters) - len(merged),
                "clusters": [
                    {
                        "canonical_name": c["canonical_name"],
                        "canonical_type": c["canonical_type"],
                        "merged_from": c.get("merged_from", [c["canonical_name"]]),
                        "instances": c["instances"],
                    }
                    for c in merged
                ],
            }
        )
        print()

    result = {
        "strategy": "B_embedding_similarity",
        "model": MODEL_NAME,
        "unique_before_A": len(entities),
        "unique_after_A": len(clusters),
        "threshold_results": threshold_results,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
