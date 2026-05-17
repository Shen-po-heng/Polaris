# Experiment 3：Entity Deduplication 策略驗證

## 狀態：🔄 進行中

---

## 問題

Entity extraction（Exp 2）之後，同一個概念可能以不同形式出現在圖譜裡：

```
"ImageNet"（AlexNet 論文）  ←→  "ImageNet 2012"（ResNet 論文）
"Fisher Vectors"（AlexNet）  ←→  "Fisher Vector"（ResNet）
"top-1 error"               ←→  "top-1 error rate"
"Adam"（method section）    ←→  "Adam"（results section）← 同一篇，應合併
```

不 dedup → 知識圖譜破碎，同一個概念有多個 node → 查詢品質下降。

---

## 實驗設計

### 輸入
`experiments/entity_extraction/approach_f/results_f.json`
- 5 篇論文 × 3 chunks = 15 chunks
- Approach F（Claude Haiku 4.5 + prefill + few-shot）抽取的 entities

### Ground Truth
`experiments/deduplication/ground_truth_clusters.json`
- 手工標注哪些 entity instances 應該合併成一個 knowledge graph node
- 格式：merge cluster（一組應該是同一個 node 的 instances）

### 三種策略

| 策略 | 方法 | 外部依賴 |
|------|------|---------|
| A: Exact match | 大小寫正規化後完全匹配 | 無 |
| B: Embedding similarity | sentence-transformers cosine similarity | sentence-transformers |
| C: Alias dictionary | 手工維護常見變體字典 | 無 |

### 評估指標

以 merge cluster 為單位，測量各策略的 precision / recall / F1：
- **Precision**：策略合併的 pair 中，有多少是正確的（gold truth 也要合併）
- **Recall**：gold truth 要合併的 pair 中，有多少被找到
- **F1**：Precision 和 Recall 的調和平均
- **Unique nodes**：dedup 前後的圖譜節點數

---

## 驗收標準

- Recall ≥ 0.80（不漏合太多真正重複的 entity）
- Precision ≥ 0.85（不錯誤合併不同的 entity）
- 至少一種策略的 F1 ≥ 0.80

---

## 執行方式

```bash
# Strategy A
python experiments/deduplication/approach_a/approach_a_exact.py

# Strategy B
python experiments/deduplication/approach_b/approach_b_embedding.py

# Strategy C
python experiments/deduplication/approach_c/approach_c_alias.py

# 綜合 benchmark
python experiments/deduplication/benchmark.py
```

---

## 預期產出

- `Report_exp3.md`：三種策略的比較分析
- 選出進入 Phase 3c 的 dedup 策略
- 決定 Embedding dedup 的最佳 threshold（Strategy B）
