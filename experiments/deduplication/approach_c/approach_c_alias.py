"""
Experiment 3 — Strategy C: Alias Dictionary Deduplication

Apply a manually maintained alias dictionary to normalize entity names
before running exact match. Catches known surface-form variants.
"""

import json
import re
from pathlib import Path

RESULTS_F = Path(__file__).parent.parent.parent / "entity_extraction" / "approach_f" / "results_f.json"
OUTPUT = Path(__file__).parent / "results_c.json"

# Alias dictionary: variant → canonical form
# Key: lowercased variant, Value: canonical display name
ALIAS_MAP: dict[str, str] = {
    # Dataset variants
    "imagenet 2012": "ImageNet",
    "imagenet2012": "ImageNet",
    "ilsvrc-2010": "ImageNet",
    "ilsvrc 2010": "ImageNet",
    # Metric variants
    "top-1 error": "top-1 error rate",
    "top-5 error": "top-5 error rate",
    "training error": "training error rate",
    "validation error": "validation error rate",
    # Method variants (plural/singular)
    "fisher vectors": "Fisher Vector",
    "residual networks": "ResNet",
    "deep residual networks": "ResNet",
    "deep residual learning": "ResNet",
    # Abbreviation expansions
    "non-line-of-sight": "NLOS",
    "non line of sight": "NLOS",
    "non-los scenario": "NLOS",
    "recurrent neural network": "RNN",
    "recurrent neural networks": "RNN",
    "long short-term memory": "LSTM",
    "convolutional neural network": "convolutional neural networks",
}


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def apply_alias(name: str) -> str:
    norm = _normalize(name)
    return ALIAS_MAP.get(norm, name)


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
                }
            )
    return entities


def deduplicate_with_alias(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply alias map then exact match. Return (clusters, alias_hits)."""
    cluster_map: dict[str, dict] = {}
    alias_hits = []

    for e in entities:
        canonical = apply_alias(e["name"])
        if canonical != e["name"]:
            alias_hits.append({"original": e["name"], "canonical": canonical})
        key = _normalize(canonical)

        if key not in cluster_map:
            cluster_map[key] = {
                "canonical_name": canonical,
                "canonical_type": e["type"],
                "instances": [],
            }
        cluster_map[key]["instances"].append(
            {"name": e["name"], "paper_id": e["paper_id"], "section_type": e["section_type"]}
        )

    return list(cluster_map.values()), alias_hits


def main() -> None:
    entities = load_entities(RESULTS_F)
    print(f"Total entity instances: {len(entities)}")

    clusters, alias_hits = deduplicate_with_alias(entities)

    unique_before = len(entities)
    unique_after = len(clusters)
    merged_clusters = [c for c in clusters if len(c["instances"]) > 1]

    print(f"Unique nodes before dedup : {unique_before}")
    print(f"Unique nodes after  dedup : {unique_after}")
    print(f"Clusters merged (2+ instances): {len(merged_clusters)}")
    print()

    if alias_hits:
        print("Alias hits (name normalized):")
        for hit in alias_hits:
            print(f"  '{hit['original']}' → '{hit['canonical']}'")
        print()

    print("All merged clusters:")
    for c in sorted(merged_clusters, key=lambda x: x["canonical_name"]):
        instances_summary = [f"{i['paper_id']}/{i['section_type']}" for i in c["instances"]]
        print(f"  '{c['canonical_name']}' ({c['canonical_type']}) ← {instances_summary}")

    result = {
        "strategy": "C_alias_dictionary",
        "alias_hits": alias_hits,
        "unique_before": unique_before,
        "unique_after": unique_after,
        "merged_count": len(merged_clusters),
        "clusters": [
            {
                "canonical_name": c["canonical_name"],
                "canonical_type": c["canonical_type"],
                "instances": c["instances"],
            }
            for c in clusters
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
