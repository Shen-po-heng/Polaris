"""
Approach E: Claude Haiku 4.5 + prefill + 3 few-shot examples.

Same as Approach D but adds one example per section type (method / related_work / results)
to clarify the Contribution vs Baseline distinction.

Usage:
    python experiments/entity_extraction/approach_e/approach_e_claude_fewshot.py

Requires: ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR    = SCRIPT_DIR.parent
GT_DIR     = EXP_DIR / "ground_truth"
OUT_FILE   = SCRIPT_DIR / "results_e.json"

MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

FEW_SHOT = """\
=== Example 1 — method section ===
Text: "We propose GAN, a generative adversarial framework in which a generator G and \
discriminator D compete. G tries to fool D; D tries to distinguish real from fake samples. \
We compare against VAE and diffusion models as baselines."
Output:
{"entities": [{"name": "GAN", "type": "Contribution"}, {"name": "generator", "type": "Concept"}, \
{"name": "discriminator", "type": "Concept"}, {"name": "VAE", "type": "Baseline"}, \
{"name": "diffusion models", "type": "Baseline"}], \
"relations": [{"head": "GAN", "tail": "generator", "type": "uses"}, \
{"head": "GAN", "tail": "discriminator", "type": "uses"}, \
{"head": "GAN", "tail": "VAE", "type": "outperforms"}]}

=== Example 2 — related_work section ===
Text: "Word2Vec learns word embeddings via skip-gram. GloVe uses global co-occurrence statistics. \
ELMo generates contextual embeddings with a BiLSTM. Our method, BERT, builds on these by \
introducing bidirectional pre-training with a Transformer."
Output:
{"entities": [{"name": "BERT", "type": "Contribution"}, {"name": "Word2Vec", "type": "Baseline"}, \
{"name": "GloVe", "type": "Baseline"}, {"name": "ELMo", "type": "Baseline"}, \
{"name": "BiLSTM", "type": "Concept"}, {"name": "Transformer", "type": "Concept"}], \
"relations": [{"head": "BERT", "tail": "Word2Vec", "type": "related_to"}, \
{"head": "BERT", "tail": "GloVe", "type": "related_to"}, \
{"head": "BERT", "tail": "ELMo", "type": "related_to"}]}

=== Example 3 — results section ===
Text: "Our model achieves 85.2 BLEU on WMT14 En-De, surpassing the previous best of 82.1 by \
ByteNet and 82.7 by ConvS2S. We also report perplexity on Penn Treebank."
Output:
{"entities": [{"name": "our model", "type": "Contribution"}, {"name": "ByteNet", "type": "Baseline"}, \
{"name": "ConvS2S", "type": "Baseline"}, {"name": "BLEU", "type": "Metric"}, \
{"name": "perplexity", "type": "Metric"}, {"name": "WMT14", "type": "Artifact"}, \
{"name": "Penn Treebank", "type": "Artifact"}], \
"relations": [{"head": "our model", "tail": "ByteNet", "type": "outperforms"}, \
{"head": "our model", "tail": "ConvS2S", "type": "outperforms"}, \
{"head": "our model", "tail": "WMT14", "type": "evaluated_on"}]}

"""

USER_TEMPLATE = """\
You are an academic knowledge graph extractor.

Given this text from the **{section_type}** section of an academic paper, \
extract entities and relations as JSON.

Entity types: Contribution, Baseline, Concept, Metric, Artifact, Context
Edge types: proposes, outperforms, uses, evaluated_on, measures, related_to

Rules:
- Use canonical/abbreviated names ("Adam" not "Adam optimizer")
- Methods/Our Approach section → novel methods proposed by THIS paper = Contribution
- Related Work/Introduction/Background → cited existing methods = Baseline
- If the paper's OWN method appears in Related Work (e.g. "our method, X, builds on..."), \
it is still Contribution, not Baseline
- Do NOT include the paper itself as an entity
- Output ONLY valid JSON, no markdown fences, no explanation

{few_shot}
=== Now extract from this text ===
Section type: {section_type}
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
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    return key


def _parse_prefilled(raw: str) -> dict | None:
    full = "{" + raw.strip()
    full = re.sub(r"\s*```$", "", full).strip()
    try:
        return json.loads(full)
    except json.JSONDecodeError:
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
    client  = anthropic.Anthropic(api_key=api_key)

    chunks = _load_ground_truth()
    if not chunks:
        print(f"No verified ground truth found in {GT_DIR}")
        sys.exit(1)

    print(f"Approach E — Anthropic {MODEL} (prefill + few-shot)")
    print(f"Chunks: {len(chunks)}")
    print("=" * 60)

    results  = []
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
                    {"role": "user", "content": USER_TEMPLATE.format(
                        section_type=chunk["section_type"],
                        few_shot=FEW_SHOT,
                        text=chunk["text"],
                    )},
                    {"role": "assistant", "content": "{"},
                ],
            )
            raw     = resp.content[0].text
            elapsed = time.time() - t0
            parsed  = _parse_prefilled(raw)
            ok      = parsed is not None and "entities" in parsed and "relations" in parsed
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
    print(f"Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
