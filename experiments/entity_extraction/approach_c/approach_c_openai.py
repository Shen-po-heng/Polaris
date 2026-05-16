"""
Approach C: OpenAI gpt-4o-mini with JSON Object mode.

Usage:
    python experiments/entity_extraction/approach_c/approach_c_openai.py

Requires: OPENAI_API_KEY environment variable (or .env file).
    pip install openai python-dotenv
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR    = SCRIPT_DIR.parent
GT_DIR     = EXP_DIR / "ground_truth"
OUT_FILE   = SCRIPT_DIR / "results_c.json"

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are an academic knowledge graph extractor. Output ONLY valid JSON with keys
"entities" (list of {name, type}) and "relations" (list of {head, tail, type}).
No markdown, no explanation."""

USER_TEMPLATE = """\
Given this text from the **{section_type}** section of an academic paper, \
extract entities and relations.

Entity types: Contribution, Baseline, Concept, Metric, Artifact, Context
Edge types: proposes, outperforms, uses, evaluated_on, measures, related_to

Rules:
- Use canonical/abbreviated names ("Adam" not "Adam optimizer")
- Methods/Our Approach section → novel methods = Contribution
- Related Work/Introduction/Background → cited methods = Baseline
- Do NOT include the paper itself as an entity

Text:
{text}"""


def _load_ground_truth() -> list[dict]:
    chunks = []
    for f in sorted(GT_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("verified"):
            chunks.append(data)
    return chunks


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)
    return key


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed. Run: pip install openai")
        sys.exit(1)

    api_key = _load_api_key()
    client  = OpenAI(api_key=api_key)

    chunks = _load_ground_truth()
    if not chunks:
        print(f"No verified ground truth found in {GT_DIR}")
        sys.exit(1)

    print(f"Approach C — OpenAI {MODEL} (JSON Object mode)")
    print(f"Chunks: {len(chunks)}")
    print("=" * 60)

    results  = []
    parse_ok = 0

    for i, chunk in enumerate(chunks, 1):
        slug = f"{chunk['paper_id']}_{chunk['section_heading']}"
        print(f"[{i:2d}/{len(chunks)}] {slug[:55]}", end=" ... ", flush=True)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_TEMPLATE.format(
                        section_type=chunk["section_type"],
                        text=chunk["text"],
                    )},
                ],
                temperature=0,
            )
            raw     = resp.choices[0].message.content or ""
            elapsed = time.time() - t0
            parsed  = json.loads(raw)
            ok      = "entities" in parsed and "relations" in parsed
        except Exception as e:
            raw     = f"ERROR: {e}"
            parsed  = None
            ok      = False
            elapsed = time.time() - t0

        if ok:
            parse_ok += 1
            print(f"OK  ({elapsed:.1f}s, {len(parsed['entities'])}e {len(parsed['relations'])}r)")
        else:
            print(f"FAIL ({elapsed:.1f}s)")

        results.append({
            "paper_id":        chunk["paper_id"],
            "section_heading": chunk["section_heading"],
            "section_type":    chunk["section_type"],
            "parse_ok":        ok,
            "elapsed_s":       round(elapsed, 2),
            "raw_response":    raw,
            "extracted":       parsed if ok else None,
            "ground_truth": {
                "entities":  chunk["entities"],
                "relations": chunk["relations"],
            },
        })

    print("=" * 60)
    print(f"JSON parse success: {parse_ok}/{len(chunks)} ({100*parse_ok/len(chunks):.1f}%)")
    OUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
