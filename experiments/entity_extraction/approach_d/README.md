# Approach D: Anthropic Claude Haiku 4.5

**LLM**: `anthropic/claude-haiku-4-5-20251001`
**Mode**: Messages API，prompt 要求輸出 JSON，加 `prefill: "{"` 強制 JSON 開頭
**需要**: `ANTHROPIC_API_KEY` 環境變數
**成本**: ~$0.00025 / 1K input tokens，15 chunks ≈ < $0.05

## 設定

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

## Prefill 技巧

Claude 不支援 JSON Object mode（OpenAI 那種），但可以用 prefill：
```python
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": "{"},  # 強制從 { 開始
]
```
然後把 `"{"` 加回到回應開頭再 parse。

## 輸出

`results_d.json`
