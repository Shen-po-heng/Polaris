"""
Experiment 3 — Strategy A: Exact Match Deduplication

Normalize entity names (lowercase + strip) and merge instances with identical names.
No external dependencies. Serves as the baseline.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_F = Path(__file__).parent.parent.parent / "entity_extraction" / "approach_f" / "results_f.json"
OUTPUT = Path(__file__).parent / "results_a.json"


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


def deduplicate(entities: list[dict]) -> dict[str, dict]:
    """Group entities by normalized name. Return canonical node map."""
    clusters: dict[str, dict] = {}
    for e in entities:
        key = e["norm_name"]
        if key not in clusters:
            clusters[key] = {
                "canonical_name": e["name"],
                "canonical_type": e["type"],
                "instances": [],
            }
        clusters[key]["instances"].append(
            {"name": e["name"], "paper_id": e["paper_id"], "section_type": e["section_type"]}
        )
    return clusters


def main() -> None:
    entities = load_entities(RESULTS_F)
    print(f"Total entity instances: {len(entities)}")

    clusters = deduplicate(entities)
    unique_before = len(entities)
    unique_after = len(clusters)

    merged_clusters = {k: v for k, v in clusters.items() if len(v["instances"]) > 1}
    print(f"Unique nodes before dedup : {unique_before}")
    print(f"Unique nodes after  dedup : {unique_after}")
    print(f"Clusters merged (2+ instances): {len(merged_clusters)}")
    print()
    print("Merged clusters:")
    for key, cluster in sorted(merged_clusters.items()):
        instances_summary = [f"{i['paper_id']}/{i['section_type']}" for i in cluster["instances"]]
        print(f"  '{cluster['canonical_name']}' ({cluster['canonical_type']}) ← {instances_summary}")

    result = {
        "strategy": "A_exact_match",
        "unique_before": unique_before,
        "unique_after": unique_after,
        "merged_count": len(merged_clusters),
        "clusters": [
            {
                "canonical_name": v["canonical_name"],
                "canonical_type": v["canonical_type"],
                "instances": v["instances"],
            }
            for v in clusters.values()
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
