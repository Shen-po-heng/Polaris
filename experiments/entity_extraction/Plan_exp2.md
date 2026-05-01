# Experiment 2 — Entity Extraction Quality Benchmark

**Branch**: `feat/exp2-entity-extraction` (to be branched from `feat/phase-3-graphrag`)
**Date planned**: 2026-05-01
**Status**: 計畫中

---

## 目標

確認哪個 LLM（本地 / 雲端）能穩定從學術論文 section 中輸出格式正確、分類正確的 entity JSON，
作為 Phase 3c `entity_extractor.py` 的實作決策依據。

---

## 核心問題

1. **JSON parse 成功率**：模型能不能穩定輸出有效 JSON？（3B 模型容易格式歪）
2. **Entity type 正確率**：Contribution / Baseline / Concept / Metric / Artifact / Context 分對了嗎？
3. **Contribution vs Baseline 區分**：section context（Methods vs Related Work）有沒有起作用？
4. **Relation 正確率**：proposes / outperforms / uses / evaluated_on / measures / related_to 正不正確？

---

## Entity Schema（Phase 3 通用）

```python
ENTITY_TYPES = {
    "Contribution": "這篇論文提出的主要方法/系統/架構",
    "Baseline":     "被拿來比較的既有方法（Related Work / Introduction 中出現）",
    "Concept":      "技術概念、理論基礎（low-rank decomposition、attention mechanism）",
    "Metric":       "評估指標（RMSE、accuracy、BLEU、perplexity）",
    "Artifact":     "資料集、硬體、工具、benchmark（ImageNet、NVIDIA V100、KITTI）",
    "Context":      "應用場景、任務（text classification、urban NLOS、machine translation）",
}

EDGE_TYPES = {
    "proposes",       # Paper → Contribution
    "outperforms",    # Contribution → Baseline
    "uses",           # Contribution → Concept / Artifact
    "evaluated_on",   # Contribution → Context / Artifact(dataset)
    "measures",       # Experiment → Metric
    "related_to",     # 兜底
}
```

---

## Prompt Template

```
You are an academic knowledge graph extractor.

Given this text from the **{section_type}** section of an academic paper,
extract entities and relations as JSON.

Entity types: Contribution, Baseline, Concept, Metric, Artifact, Context
Edge types: proposes, outperforms, uses, evaluated_on, measures, related_to

Rules:
- Always use the most canonical/abbreviated form of entity names (e.g. "Adam" not "Adam optimizer")
- In Methods / Our Approach section: novel methods proposed by this paper → Contribution
- In Related Work / Introduction section: cited existing methods → Baseline
- Do NOT include the paper itself as an entity
- Output ONLY valid JSON, no markdown fences, no explanation

Text:
{section_text}

Output JSON:
{"entities": [{"name": str, "type": str}], "relations": [{"head": str, "tail": str, "type": str}]}
```

---

## Test Corpus

同 Experiment 1，使用同 5 篇論文的選定 sections：

| Paper | 取用 Sections | 理由 |
|-------|-------------|------|
| Adam (1412.6980) | 2 ALGORITHM, 6 EXPERIMENTS, 5 RELATED WORK | method / results / related_work 各一 |
| ResNet (1512.03385) | 3 Deep Residual Learning, 4 Experiments, Related Work | 同上 |
| Transformer (1706.03762) | 3 Model Architecture, 6 Results, 2 Background | 同上 |
| BERT (1810.04805) | 3 Pre-training, 4 Experiments, Related Work | 同上 |
| AlexNet (NIPS-2012) | 3 The Architecture, 5 Results, Introduction | 同上 |

每篇取 3 個 section → 總計 **15 個 chunks** 作為 benchmark input。

---

## 四種方法

| 方法 | 資料夾 | LLM | 需要 | 備注 |
|------|--------|-----|------|------|
| A | `approach_a/` | Ollama llama3.2:3b（plain） | Ollama 本地 | 基準線 |
| B | `approach_b/` | Ollama llama3.2:3b + JSON mode | Ollama 本地 | `format: json` 參數 |
| C | `approach_c/` | OpenAI gpt-4o-mini | OpenAI API key | 雲端對照組 |
| D | `approach_d/` | Anthropic Claude Haiku 4.5 | Anthropic API key | 雲端對照組 |

**執行順序**：先跑 A → B（不需 API），再視結果決定要不要跑 C / D。

---

## Ground Truth 格式

`ground_truth/{paper_id}_{section_slug}.json`

```json
{
  "paper_id": "1706.03762",
  "section_heading": "3 Model Architecture",
  "section_type": "method",
  "text_excerpt": "The Transformer follows an encoder-decoder structure...",
  "verified": true,
  "entities": [
    {"name": "Transformer", "type": "Contribution"},
    {"name": "encoder-decoder", "type": "Concept"},
    {"name": "multi-head attention", "type": "Concept"},
    {"name": "BLEU", "type": "Metric"}
  ],
  "relations": [
    {"head": "Transformer", "tail": "encoder-decoder", "type": "uses"},
    {"head": "Transformer", "tail": "multi-head attention", "type": "uses"},
    {"head": "Transformer", "tail": "BLEU", "type": "measures"}
  ]
}
```

---

## 評估指標

### Entity-level

| 指標 | 計算方式 | 門檻 |
|------|---------|------|
| JSON parse 成功率 | 成功 parse 的 chunk 數 / 15 | > 90% |
| Entity precision | TP / (TP + FP) | > 60% |
| Entity recall | TP / (TP + FN) | > 60% |
| Entity F1 | harmonic mean | > 60% |
| Type accuracy | 正確 type 的 entity / TP | > 70% |

### Section-context-level（最重要）

| 指標 | 計算方式 | 門檻 |
|------|---------|------|
| Contribution vs Baseline 區分準確率 | Methods section 的 method entity 標為 Contribution 的比例 + Related Work 的標為 Baseline 的比例 | > 80% |

### Relation-level

| 指標 | 計算方式 | 門檻 |
|------|---------|------|
| Relation F1 | 同 entity F1，以 (head, tail, type) 為 key | 觀察用，無硬性門檻 |

---

## 失敗降級策略

| 失敗情況 | 處理方式 |
|---------|---------|
| A JSON parse 成功率 < 90% | 改用 B（JSON mode）|
| B 仍 < 90% | 必須用 C 或 D（雲端）|
| Type accuracy < 70% | 改寫 prompt，加更多 few-shot examples |
| Contribution vs Baseline < 80% | 在 prompt 加強 section role instruction |

---

## 產出物規劃

| 檔案 | 說明 |
|------|------|
| `approach_a/approach_a_ollama.py` | Ollama plain 抽取 |
| `approach_b/approach_b_ollama_json.py` | Ollama JSON mode 抽取 |
| `approach_c/approach_c_openai.py` | OpenAI gpt-4o-mini 抽取 |
| `approach_d/approach_d_claude.py` | Claude Haiku 4.5 抽取 |
| `benchmark.py` | 統一 runner，支援 `--approach A/B/C/D/all` |
| `ground_truth/*.json` | 15 個 chunk 的人工標注（需手動建立後驗證） |
| `generate_ground_truth.py` | 自動從 A2 section text 提取 chunk，生成草稿 |
| `results_a.json` / `results_b.json` / ... | 各方法完整輸出 |
| `Report_exp2.md` | 四方比較報告（實驗結束後寫） |

---

## 執行計畫

```
Step 1: 建立 ground_truth（先做，再跑模型）
  - 從 approach_a2_grobid.detect_sections() 取 15 個 sections 的 text
  - 人工標注 entities + relations，設 verified=true

Step 2: 實作 approach_a（Ollama plain）
  - 呼叫 Ollama REST API（http://localhost:11434）
  - 跑 benchmark，看 JSON parse 成功率

Step 3: 實作 approach_b（Ollama JSON mode）
  - 加 format: "json" 參數
  - 比較 A vs B parse 成功率

Step 4: 視 A/B 結果決定
  - 若 A/B 通過門檻 → 選較好的進 Phase 3c，跳過 C/D
  - 若 A/B 不通過 → 實作 C / D（需申請 API key）

Step 5: 寫 Report_exp2.md
```

---

## 決策矩陣（實驗後填入）

| 方法 | JSON parse% | Type acc% | Contrib/Base% | 成本 | 推薦用於 |
|------|------------|----------|--------------|------|---------|
| A: llama3.2 plain | — | — | — | 免費 | — |
| B: llama3.2 JSON mode | — | — | — | 免費 | — |
| C: gpt-4o-mini | — | — | — | ~$0.001/chunk | — |
| D: claude-haiku-4-5 | — | — | — | ~$0.001/chunk | — |

---

## 參考

- Approach A2 section text 來源：`experiments/section_detection/approach_a/approach_a2_grobid.py`
- Entity Schema 定義：`Ref/project-plan-v2.md` → Phase 3 通用 Entity Schema
- LiteLLM 統一介面：`core/llm_provider.py`
