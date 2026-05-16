"""
Approach D: Anthropic Claude Haiku 4.5 with assistant prefill trick.

Claude does not have a JSON Object mode like OpenAI, but prefilling the
assistant turn with "{" forces the model to start directly from the opening
brace, which we prepend back before parsing.

Usage:
    python experiments/entity_extraction/approach_d/approach_d_claude.py

Requires: ANTHROPIC_API_KEY environment variable (or .env file).
    pip install anthropic python-dotenv
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
GT_DIR = EXP_DIR / "ground_truth"
OUT_FILE = SCRIPT_DIR / "results_d.json"

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

USER_TEMPLATE = """\
You are an academic knowledge graph extractor.

Given this text from the **{section_type}** section of an academic paper, \
extract entities and relations as JSON.

Entity types: Contribution, Baseline, Concept, Metric, Artifact, Context
Edge types: proposes, outperforms, uses, evaluated_on, measures, related_to

Rules:
- Use canonical/abbreviated names ("Adam" not "Adam optimizer")
- Methods/Our Approach section → novel methods = Contribution
- Related Work/Introduction/Background → cited methods = Baseline
- Do NOT include the paper itself as an entity
- Output ONLY valid JSON, no markdown fences, no explanation

Text:
{text}

Output JSON:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"head": "...", "tail": "...", "type": "..."}}]}}"""


def _load_ground_truth() -> list[dict]:
    chunks = []
    for f in sorted(GT_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("verified"):
            chunks.append(data)
    return chunks


def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)
    return key


def _parse_prefilled(raw: str) -> dict | None:
    # The model response is everything AFTER the "{" prefill,
    # so we prepend it back.
    full = "{" + raw.strip()
    # strip trailing markdown fence if any
    full = re.sub(r"\s*```$", "", full).strip()
    try:
        return json.loads(full)
    except json.JSONDecodeError:
        # try to find the outermost {...}
        m = re.search(r"\{.*\}", full, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def main() -> None:
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = _load_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    chunks = _load_ground_truth()
    if not chunks:
        print(f"No verified ground truth found in {GT_DIR}")
        sys.exit(1)

    print(f"Approach D — Anthropic {MODEL} (prefill)")
    print(f"Chunks: {len(chunks)}")
    print("=" * 60)

    results = []
    parse_ok = 0

    for i, chunk in enumerate(chunks, 1):
        slug = f"{chunk['paper_id']}_{chunk['section_heading']}"
        print(f"[{i:2d}/{len(chunks)}] {slug[:55]}", end=" ... ", flush=True)

        t0 = time.time()
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            section_type=chunk["section_type"],
                            text=chunk["text"],
                        ),
                    },
                    {"role": "assistant", "content": "{"},  # prefill forces JSON start
                ],
            )
            raw = resp.content[0].text
            elapsed = time.time() - t0
            parsed = _parse_prefilled(raw)
            ok = parsed is not None and "entities" in parsed and "relations" in parsed
        except Exception as e:
            raw = f"ERROR: {e}"
            parsed = None
            ok = False
            elapsed = time.time() - t0

        if ok:
            parse_ok += 1
            print(
                f"OK  ({elapsed:.1f}s, {len(parsed['entities'])}e {len(parsed['relations'])}r)"
            )
        else:
            print(f"FAIL ({elapsed:.1f}s)")

        results.append(
            {
                "paper_id": chunk["paper_id"],
                "section_heading": chunk["section_heading"],
                "section_type": chunk["section_type"],
                "parse_ok": ok,
                "elapsed_s": round(elapsed, 2),
                "raw_response": raw,
                "extracted": parsed if ok else None,
                "ground_truth": {
                    "entities": chunk["entities"],
                    "relations": chunk["relations"],
                },
            }
        )

    print("=" * 60)
    print(
        f"JSON parse success: {parse_ok}/{len(chunks)} ({100*parse_ok/len(chunks):.1f}%)"
    )
    OUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
