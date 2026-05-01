# Experiment 1 — Approach A Plan: GROBID + SciPDF Parser

**Status**: Planned (pending Docker setup)
**Motivated by**: Approach B failure on Adam v9 (drop cap + no-space PDF encoding) and font-size blind spots in ResNet and BERT.

---

## Why GROBID?

Approach B revealed two fundamental failure modes that regex + font heuristics cannot overcome:

| Failure mode | Example paper | Approach B result | Why GROBID handles it |
|---|---|---|---|
| Drop cap rendering | Adam v9 | 0% F1 | CRF model reads visual layout, not raw char stream |
| Headings = body font size | ResNet, BERT | 50% recall | CRF uses position, spacing, text features — not font size alone |
| No word-space tokens | Adam v9 | 0% F1 | GROBID uses its own tokeniser, not pdfplumber's char join |
| Ligature glyphs (ﬁ, ﬂ) | AlexNet | Needed fix | GROBID normalises ligatures internally |

GROBID is a **machine-learning based** document parser trained on hundreds of thousands of scientific papers. It uses a cascading CRF (Conditional Random Field) model that jointly segments and labels document structure at multiple levels:

```
PDF bytes → PDFAlto (visual layout extraction)
         → Segmentation model (header / body / references / annex)
         → Full-text model (title / abstract / section / figure / table)
         → Reference model (citation parsing)
```

**References**:
- GROBID GitHub: https://github.com/kermitt2/grobid
- GROBID documentation: https://grobid.readthedocs.io/en/latest/Principles/
- SciPDF Parser (Python wrapper): https://github.com/titipata/scipdf_parser
- Original paper: López, P. (2009). GROBID: Combining automatic bibliographic data recognition and term extraction for scholarship publications. ECDL.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │  GROBID Service (Docker, port 8070)  │
                    │  ┌────────────────────────────────┐  │
PDF file ──POST──►  │  │  PDFAlto layout extraction    │  │
                    │  │  → Segmentation CRF model     │  │
                    │  │  → Full-text CRF model        │  │
                    │  └────────────────────────────────┘  │
                    │          returns TEI-XML              │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  SciPDF Parser (Python)              │
                    │  parse_pdf_to_dict()                 │
                    │  → { title, abstract,                │
                    │      sections: [{heading, text}],    │
                    │      references }                    │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  Our infer_section_type()            │
                    │  heading text → type label           │
                    │  (same regex as Approach B,          │
                    │   but now applied to CLEAN text)     │
                    └──────────────────────────────────────┘
```

---

## Setup

### Step 1 — Start GROBID service

```bash
# Pull the image (one-time, ~500 MB)
docker pull lfoppiano/grobid:0.8.0

# Start service (leave running in background)
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

Verify it's running:
```
http://localhost:8070  →  GROBID web console
```

### Step 2 — Install Python wrapper

```bash
# Add to experiments/section_detection/requirements_exp.txt
pip install scipdf-parser
```

> Note: `scipdf-parser` requires `grobid` to be running at localhost:8070 before calling any parse function.

---

## Code Plan: `approach_a_grobid.py`

```python
"""
Approach A: GROBID + SciPDF Parser for academic section detection.
Requires GROBID service running at localhost:8070 (see setup above).
"""
import scipdf
from dataclasses import dataclass, field

GROBID_URL = "http://localhost:8070"

# Reuse the same type-inference logic as Approach B
# (imported from approach_b_heuristic to keep comparison fair)
from approach_b_heuristic import _normalize_heading, _match_section_type


@dataclass
class DetectedSection:
    heading: str
    type: str
    text: str = ""


def detect_sections(pdf_path: str) -> list[DetectedSection]:
    """
    Detect top-level sections using GROBID.
    Falls back to empty list if GROBID is unavailable.
    """
    try:
        article = scipdf.parse_pdf_to_dict(str(pdf_path), grobid_url=GROBID_URL)
    except Exception as e:
        raise RuntimeError(f"GROBID unavailable: {e}. Start with: docker run -p 8070:8070 lfoppiano/grobid:0.8.0")

    sections = []

    # Abstract is always returned as a top-level field
    if article.get("abstract"):
        sections.append(DetectedSection(
            heading="Abstract",
            type="abstract",
            text=article["abstract"],
        ))

    for sec in article.get("sections", []):
        heading = sec.get("heading") or ""
        if not heading.strip():
            continue
        sec_type = _match_section_type(heading) or "other"
        sections.append(DetectedSection(
            heading=heading,
            type=sec_type,
            text=sec.get("text", ""),
        ))

    return sections
```

**Important design note**: We reuse `_match_section_type()` from Approach B deliberately. This makes the comparison fair — the only variable is the text extraction backend (GROBID vs pdfplumber), not the classification logic.

---

## What We Expect vs Approach B

| Aspect | Approach B (pdfplumber) | Approach A (GROBID) |
|--------|------------------------|---------------------|
| Adam v9 drop cap | 0% (completely fails) | Expected: recover sections |
| Font-size blind spots | Recall ~50% | Expected: higher recall |
| Clean arXiv PDFs (Transformer, AlexNet) | Already 94% | Expected: similar or slightly lower (GROBID adds overhead) |
| Text quality | No spaces, ligatures | Normalised, clean |
| Speed | < 100ms / paper | 2–5s / paper (service call) |
| Dependencies | pdfplumber only | Docker + GROBID + scipdf-parser |
| Offline use | Yes | Yes (GROBID runs locally) |

---

## Benchmark Integration

`benchmark.py` will be extended to run both approaches on the same ground truth and produce a side-by-side comparison:

```
Paper              | B: F1  | A: F1  | Winner | Notes
─────────────────────────────────────────────────────────
Adam               |  0.0%  |  ?     |  A     | drop cap rescue
ResNet             | 66.7%  |  ?     |  ?     |
Transformer        | 94.1%  |  ?     |  ?     |
BERT               | 66.7%  |  ?     |  ?     |
AlexNet (2-col)    | 94.1%  |  ?     |  ?     |
─────────────────────────────────────────────────────────
MEAN               | 64.3%  |  ?     |        |
```

The updated benchmark will import `detect_sections` from either `approach_b_heuristic` or `approach_a_grobid` based on a `--approach` argument.

---

## Known Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GROBID returns empty sections for some PDFs | Low | Log + fallback to Approach B |
| SciPDF Parser heading extraction incomplete | Medium | Inspect raw TEI-XML if needed |
| Docker not available on target machine | Possible | Document as prerequisite; Approach B as fallback |
| GROBID section type labels don't map cleanly to our schema | Medium | Our own `infer_section_type()` sits on top, decoupled from GROBID |
| Two-column handling: GROBID may merge columns incorrectly | Low (GROBID handles this well) | Verify with AlexNet |

---

## Decision Criteria After Running Approach A

| Outcome | Decision |
|---------|----------|
| Approach A mean F1 ≥ 80% AND fixes Adam | Adopt GROBID as primary, Approach B as fast fallback |
| Approach A mean F1 similar to B, no improvement | Approach B sufficient; avoid Docker dependency |
| Approach A slower but significantly more accurate | Offer as `--high-quality` flag in production |
| Both approaches < 80% on multi-domain papers | Proceed to Approach C (CRF) |
