# Ground Truth — Experiment 2: Entity Extraction

每個 JSON 對應一個 section chunk（paper × section）。

## 命名規則

```
{paper_id}_{section_slug}.json
```

例：`1706.03762_model_architecture.json`

## 格式

```json
{
  "paper_id": "1706.03762",
  "section_heading": "3 Model Architecture",
  "section_type": "method",
  "text_excerpt": "...",
  "verified": false,
  "entities": [
    {"name": "Transformer", "type": "Contribution"},
    {"name": "multi-head attention", "type": "Concept"},
    {"name": "BLEU", "type": "Metric"}
  ],
  "relations": [
    {"head": "Transformer", "tail": "multi-head attention", "type": "uses"},
    {"head": "Transformer", "tail": "BLEU", "type": "measures"}
  ]
}
```

## 目標清單（15 chunks）

| # | 檔案 | Section (實際 GROBID heading) | section_type | Chars | verified |
|---|------|------------------------------|-------------|-------|---------|
| 1 | adam_method.json | 2.1 ADAM'S UPDATE RULE | method | 2028 | ☐ |
| 2 | adam_results.json | 6 EXPERIMENTS | results | 575 | ☐ |
| 3 | adam_related.json | 5 RELATED WORK | related_work | 2481 | ☐ |
| 4 | resnet_method.json | 3. Deep Residual Learning (含 3.1-3.4) | method | 6038 | ☐ |
| 5 | resnet_results.json | 4.1. ImageNet Classification | results | 6498 | ☐ |
| 6 | resnet_related.json | 2. Related Work | related_work | 2455 | ☐ |
| 7 | transformer_method.json | 3 Model Architecture | method | 711 | ☐ |
| 8 | transformer_results.json | 6.1 Machine Translation | results | 1675 | ☐ |
| 9 | transformer_related.json | 2 Background | related_work | 1817 | ☐ |
| 10 | bert_method.json | 3.1 Pre-training BERT | method | 3537 | ☐ |
| 11 | bert_results.json | 4.1 GLUE | results | 1992 | ☐ |
| 12 | bert_related.json | 2 Related Work (含 2.1-2.3) | related_work | 3851 | ☐ |
| 13 | alexnet_method.json | 3 The Architecture (含 3.1-3.5) | method | 8003 | ☐ |
| 14 | alexnet_results.json | 6 Results | results | 561 | ☐ |
| 15 | alexnet_intro.json | 1 Introduction | introduction | 4150 | ☐ |

## Entity Type 參考

| Type | 說明 | 例子 |
|------|------|------|
| Contribution | 這篇論文提出的方法/系統 | Transformer, Adam optimizer, BERT |
| Baseline | 被比較的既有方法 | SGD, RNNs, ELMo |
| Concept | 技術概念、理論 | attention mechanism, dropout, batch normalization |
| Metric | 評估指標 | BLEU, accuracy, RMSE, perplexity |
| Artifact | 資料集、硬體、工具 | ImageNet, WMT 2014, NVIDIA P100 |
| Context | 應用場景、任務 | machine translation, image classification |
