# Approach C: OpenAI gpt-4o-mini

**LLM**: `openai/gpt-4o-mini`
**Mode**: Chat completion with `response_format: {"type": "json_object"}`
**需要**: `OPENAI_API_KEY` 環境變數
**成本**: ~$0.00015 / 1K input tokens，15 chunks ≈ < $0.05

## 設定

```bash
# .env
OPENAI_API_KEY=sk-...
```

## 為什麼用這個

- gpt-4o-mini 有原生 JSON Object mode，parse 成功率接近 100%
- 推理能力遠高於 3B 本地模型，entity type 正確率預期明顯提升
- 作為雲端對照組，確認 3B 本地模型的上限差距

## 輸出

`results_c.json`
