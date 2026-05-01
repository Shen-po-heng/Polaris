# Ground Truth Annotation Guide

Each JSON file contains a pre-filled section list based on prior knowledge of each paper.
**Your job**: open the paper PDF and verify / correct the JSON. This should take 5-10 minutes per paper.

## How to annotate

### Step 1 — Open the PDF
Open the corresponding PDF in `ref_papers/` side-by-side with the JSON file.

### Step 2 — Check each section heading
Compare what's in `"heading"` against the actual text in the paper.
- Heading text must match **exactly** as it appears in the PDF (including the number, e.g., "3 Model Architecture" not "Model Architecture")
- If a section is missing from the JSON, add it in the correct order
- If a section in the JSON doesn't exist in the PDF, delete that entry

### Step 3 — Check the type label
Allowed types (pick the closest one):

| type | When to use |
|------|-------------|
| `abstract` | The Abstract section |
| `introduction` | Introduction, Background, Motivation, Overview |
| `related_work` | Related Work, Prior Work, Literature Review |
| `method` | Methods, Methodology, Approach, Architecture, Proposed ..., System Design, Training |
| `results` | Results, Experiments, Evaluation, Performance, Ablation |
| `discussion` | Discussion, Analysis |
| `conclusion` | Conclusion, Future Work, Summary |
| `references` | References, Bibliography |
| `other` | Appendix, Acknowledgements, or anything that doesn't fit |

### Step 4 — Mark as verified
Change `"verified": false` to `"verified": true` when done.

## Example of a corrected entry

Before (pre-filled, possibly wrong):
```json
{"heading": "3 Model Architecture", "type": "method"}
```

After (corrected to match exact PDF text):
```json
{"heading": "3 Model Architecture", "type": "method"}
```

Or if the paper actually says "3. Architecture":
```json
{"heading": "3. Architecture", "type": "method"}
```

## What "heading" means here
Only top-level sections (1, 2, 3...). Do NOT include subsections (1.1, 2.3, etc.) unless they are clearly the same level as the main sections in that paper.
