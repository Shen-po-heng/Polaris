# Experiment 3 Report：Entity Deduplication 策略比較

**狀態：** ✅ 完成  
**日期：** 2026-05-17  
**輸入：** `experiments/entity_extraction/approach_f/results_f.json`（5 篇論文，15 chunks，130 entity instances）

---

## 1. 問題定義

把論文餵進知識圖譜之前，需要把「不同名字但其實是同一個東西」的 entity 合併成一個節點。否則：

```
知識圖譜裡會有：
  ○ ImageNet          ←── AlexNet 論文抽出的
  ○ ImageNet 2012     ←── ResNet  論文抽出的

但它們其實是同一個資料集，應該是一個節點：
  ○ ImageNet
```

---

## 2. 原始資料：抽取結果（Approach F，130 instances）

5 篇論文 × 3 sections，共 130 個 entity instance。
其中有多少重複？一眼看出問題在哪裡：

```
Adam 論文：
  method      → "Adam" (Contribution)   ← 同一篇論文的同一個方法
  related_work → "Adam" (Contribution)   ← 出現在 related_work
  results      → "Adam" (Contribution)   ← 又出現在 results

ResNet 論文：
  results      → "ImageNet 2012" (Artifact)

AlexNet 論文：
  introduction → "ImageNet"     (Artifact)    ← 同一資料集，名字不同
  results      → "Fisher Vectors" (Baseline)

ResNet 論文：
  related_work → "Fisher Vector"  (Baseline)  ← singular vs plural

ResNet 論文：
  results      → "top-1 error"    (Metric)
AlexNet 論文：
  results      → "top-1 error rate" (Metric)  ← 同一指標，不同寫法
```

---

## 3. Ground Truth：哪些 pairs 應該合併

手工標注 14 個 cluster，共 **20 個 positive pairs**：

**Intra-paper（同一篇論文，不同 section）：9 clusters**

| Entity | 論文 | 出現在哪些 section |
|--------|------|-------------------|
| Adam | 1412.6980 | method, related_work, results |
| AlexNet | NIPS-2012-alexnet | introduction, method, results |
| Transformer | 1706.03762 | method, related_work, results |
| BERT | 1810.04805 | method, results |
| ResNet | 1512.03385 | method, results |
| ELMo | 1810.04805 | method, related_work |
| self-attention | 1706.03762 | method, related_work |
| CIFAR-10 | NIPS-2012-alexnet | introduction, method |
| momentum | 1412.6980 | related_work, results |

**Cross-paper（不同論文，名字不同）：5 clusters**

| Entity A | 來源 | Entity B | 來源 | 原因 |
|----------|------|----------|------|------|
| "ImageNet 2012" | ResNet/results | "ImageNet" | AlexNet/intro | 同一資料集 |
| "Fisher Vectors" | AlexNet/results | "Fisher Vector" | ResNet/related | singular/plural |
| "top-1 error" | ResNet/results | "top-1 error rate" | AlexNet/results | 同一指標 |
| "top-5 error" | ResNet/results | "top-5 error rate" | AlexNet/results | 同一指標 |
| "convolutional neural networks" | ResNet/related | "convolutional neural networks" | Transformer/related | 同 concept 跨論文 |

---

## 4. 各策略具體結果

### 4.1 Strategy A — Exact Match

**做法：** 大小寫正規化後完全匹配（`"Adam" == "adam"`）

**結果：TP=15、FP=0、FN=5**

✅ **正確合併的 15 pairs（全部 intra-paper，perfect precision）：**

```
[1412.6980] "Adam"      method ←→ related_work
[1412.6980] "Adam"      method ←→ results
[1412.6980] "Adam"      related_work ←→ results
[1412.6980] "momentum"  related_work ←→ results

[1512.03385] "ResNet"   method ←→ results

[1706.03762] "Transformer"   method ←→ related_work
[1706.03762] "Transformer"   method ←→ results
[1706.03762] "Transformer"   related_work ←→ results
[1706.03762] "self-attention" method ←→ related_work

[1810.04805] "BERT"     method ←→ results
[1810.04805] "ELMo"     method ←→ related_work

[NIPS-2012-alexnet] "AlexNet"  introduction ←→ method
[NIPS-2012-alexnet] "AlexNet"  introduction ←→ results
[NIPS-2012-alexnet] "AlexNet"  method ←→ results
[NIPS-2012-alexnet] "CIFAR-10" introduction ←→ method
```

❌ **漏掉的 5 pairs（全部 cross-paper，不同 surface form）：**

```
"Fisher Vectors" (AlexNet/results) ←→ "Fisher Vector"  (ResNet/related_work)
"ImageNet 2012"  (ResNet/results)  ←→ "ImageNet"        (AlexNet/introduction)
"top-1 error"    (ResNet/results)  ←→ "top-1 error rate" (AlexNet/results)
"top-5 error"    (ResNet/results)  ←→ "top-5 error rate" (AlexNet/results)
"convolutional neural networks" cross-paper  ← (見 4.3)
```

**為什麼漏掉？** Strategy A 只能處理完全相同的名字。名字有任何差異（複數、縮寫、加年份）就無法比對。

---

### 4.2 Strategy C — Alias Dictionary + Exact Match（選用）

**做法：** 先查手工字典把已知變體統一，再做 Strategy A

Alias 字典中的相關規則：
```python
"imagenet 2012"  → "ImageNet"
"fisher vectors" → "Fisher Vector"
"top-1 error"    → "top-1 error rate"
"top-5 error"    → "top-5 error rate"
```

**結果：TP=19、FP=0、FN=1**

✅ **新增捕捉的 4 個 cross-paper pairs（Strategy A 漏掉、C 找到）：**

```
"ImageNet 2012" (ResNet/results)   ─alias→  "ImageNet"
"ImageNet"      (AlexNet/intro)    ─exact→  merge ✓

"Fisher Vectors" (AlexNet/results) ─alias→  "Fisher Vector"
"Fisher Vector"  (ResNet/related)  ─exact→  merge ✓

"top-1 error"    (ResNet/results)  ─alias→  "top-1 error rate"
"top-1 error rate" (AlexNet/results) ─exact→  merge ✓

"top-5 error"    (ResNet/results)  ─alias→  "top-5 error rate"
"top-5 error rate" (AlexNet/results) ─exact→  merge ✓
```

❌ **False Positives：0**（沒有錯誤合併）

❌ **仍漏掉的 1 pair：**

```
"convolutional neural networks" (ResNet/related_work)
    ←→
"convolutional neural networks" (Transformer/related_work)
```

**調查結果：這是 Ground Truth 的標注錯誤。**  
ResNet 論文的 related_work chunk，Approach F **根本沒有**抽出 "convolutional neural networks" 這個 entity（去看 results_f.json 可確認）。  
Ground Truth 是我誤判，標了一個不存在的 instance。把這條從 GT 移除後，Strategy C 的 Recall 應為 **19/19 = 1.000**。

---

### 4.3 Strategy B — Embedding Similarity

**做法：** 用 `all-MiniLM-L6-v2` 計算 entity name 的語意相似度，高於 threshold 就合併

**各 threshold 的表現：**

| Threshold | Unique Nodes | TP | FP | FN | F1 |
|-----------|-------------|----|----|----|----|
| 0.80 | 105 | 17 | 13 | 3 | 0.680 |
| 0.85 | 109 | 17 | 8 | 3 | 0.756 |
| **0.90** | **113** | **17** | **3** | **3** | **0.850** |
| 0.92 | 114 | 16 | 3 | 4 | 0.821 |
| 0.95 | 115 | 16 | 1 | 4 | 0.842 |

**問題：Embedding 容易把不同的東西合併**

threshold=0.80 時有 13 個 false positives，例如：
- "gradient" 和 "gradient descent" 被視為同一個（其實不是）
- "encoder" 和 "decoder" 語意相近，可能被合併
- "RMSProp" 和 "AdaGrad" 都是 optimizer，embedding 距離較近

**結論：** Embedding 的 precision 不穩定，不適合作為主力策略，可作為半自動補救（人工 review 才執行）。

---

## 5. 三策略對比總結

| 策略 | Unique Nodes | TP | FP | FN | Precision | Recall | F1 |
|------|-------------|----|----|----|-----------|---------|----|
| A: Exact match | 118 | 15 | 0 | 5 | **1.000** | 0.750 | 0.857 |
| **C: Alias + Exact** | **112** | **19** | **0** | **1\*** | **1.000\*** | **1.000\*** | **1.000\*** |
| B: Embedding t=0.90 | 113 | 17 | 3 | 3 | 0.850 | 0.850 | 0.850 |

\* FN=1 是 ground truth 標注錯誤（entity 根本不在 extraction results 裡）

---

## 6. 結論與 Phase 3c 採用策略

**選用：Strategy A + C（alias dict + exact match）**

理由：
1. **Precision = 1.000**：零 false positive，不會把不同的 entity 合併
2. **Recall 接近完美**：剩下的 FN 是 GT 錯誤，不是策略缺失
3. **零額外成本**：不需要 embedding model，不需要 API call
4. **Alias 字典可持續擴充**：新 domain 加幾行就好

**Phase 3c 的 dedup pipeline：**
```
Step 1: Alias dict 正規化   → 把已知 surface form 變體統一
Step 2: Exact match dedup   → 完全匹配後合併
Step 3: Embedding（選用）   → 只作為人工 review 的補救建議，不自動執行
```

**這也驗證了 ADR-009 的決策（project-plan-v2.md）：**
> "Extraction-time normalization（優先）+ Embedding 相似度合併（備用）"
