# Approach A: Ollama llama3.2:3b（Plain Prompt）

**LLM**: `ollama/llama3.2:3b`
**Mode**: 一般 chat completion，plain text prompt
**需要**: Ollama 在本地 localhost:11434 運行
**成本**: 免費

## 設定

```bash
# 確認 Ollama 有跑
ollama list   # 應看到 llama3.2:3b
```

## 預期問題

- 3B 模型有時輸出 markdown fence（```json ... ```）而非純 JSON
- 有時在 JSON 後面加說明文字
- 需要 strip + parse，parse 失敗要記錄

## 輸出

`results_a.json`
