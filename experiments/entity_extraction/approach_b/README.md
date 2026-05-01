# Approach B: Ollama llama3.2:3b（JSON Mode）

**LLM**: `ollama/llama3.2:3b`
**Mode**: `format: "json"` 參數，強制輸出 JSON
**需要**: Ollama 在本地 localhost:11434 運行
**成本**: 免費

## 與 A 的差異

Ollama 支援 `format: "json"` 參數，讓模型在 token 層級被約束只輸出有效 JSON，
不會有 markdown fence 或多餘說明文字。

```python
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "format": "json",   # ← 這一行
        "stream": False,
    }
)
```

## 預期改善

- JSON parse 成功率從 A 的不確定 → 理論上接近 100%
- 但 JSON 內容（entity 分類）仍取決於模型能力

## 輸出

`results_b.json`
