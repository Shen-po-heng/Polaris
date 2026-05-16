"""
Approach A: Ollama llama3.2:3b plain prompt entity extraction.

Reads all verified ground truth JSONs, sends the text to Ollama with a
structured output prompt, and saves results to results_a.json.

Usage:
    python experiments/entity_extraction/approach_a/approach_a_ollama.py

Requires: Ollama running at localhost:11434 with llama3.2:3b pulled.
    ollama pull llama3.2:3b
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
GT_DIR = EXP_DIR / "ground_truth"
OUT_FILE = SCRIPT_DIR / "results_a.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

PROMPT_TEMPLATE = """\
You are an academic knowledge graph extractor.

Given this text from the **{section_type}** section of an academic paper, \
extract entities and relations as JSON.

Entity types:
- Contribution: the main method/system proposed by THIS paper
- Baseline: existing methods cited for comparison (appear in Related Work / Introduction)
- Concept: technical concepts or theoretical ideas
- Metric: evaluation metrics (BLEU, accuracy, F1, RMSE, ...)
- Artifact: datasets, hardware, tools, benchmarks (ImageNet, GPU, ...)
- Context: application domains or task types (machine translation, image classification, ...)

Edge types: proposes, outperforms, uses, evaluated_on, measures, related_to

Rules:
- Use the most canonical/abbreviated name (e.g. "Adam" not "Adam optimizer")
- In Methods/Our Approach section: novel methods proposed by this paper → Contribution
- In Related Work/Introduction/Background section: cited existing methods → Baseline
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


def _call_ollama(prompt: str) -> str:
    import urllib.request

    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["response"]


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    # strip markdown fences if model wraps anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # try extracting first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def main() -> None:
    chunks = _load_ground_truth()
    if not chunks:
        print(f"No verified ground truth found in {GT_DIR}")
        sys.exit(1)

    print(f"Approach A — Ollama {MODEL} plain prompt")
    print(f"Chunks: {len(chunks)}")
    print("=" * 60)

    results = []
    parse_ok = 0

    for i, chunk in enumerate(chunks, 1):
        slug = f"{chunk['paper_id']}_{chunk['section_heading']}"
        print(f"[{i:2d}/{len(chunks)}] {slug[:55]}", end=" ... ", flush=True)

        prompt = PROMPT_TEMPLATE.format(
            section_type=chunk["section_type"],
            text=chunk["text"],
        )

        t0 = time.time()
        try:
            raw = _call_ollama(prompt)
            elapsed = time.time() - t0
            parsed = _parse_json(raw)
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
