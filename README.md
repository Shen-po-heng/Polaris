# Polaris — Lightweight Local GraphRAG Personal Knowledge Base

[![CI](https://github.com/Shen-po-heng/Polaris/actions/workflows/test.yml/badge.svg)](https://github.com/Shen-po-heng/Polaris/actions/workflows/test.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Simple · Lightweight · Local · Private**
>
> A fully local, privacy-first document assistant built for researchers and graduate students.
> All data stays on your machine — no cloud, no API keys required.

---

## Why Polaris?

| Feature | NotebookLM | Polaris |
|---------|-----------|---------|
| Document limit | 50 sources | Unlimited |
| Privacy | Google cloud | 100% local |
| Cost | Free (limited) | Free forever |
| Offline | No | Yes |
| Customizable | No | Full source |

---

## Features (Phase 2.5)

- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- **100% local LLM** — Ollama (llama3.2, mistral, qwen2.5) — no API key needed
- **Swap models via `.env`** — switch to OpenAI / Anthropic without changing code
- **Persistent vector store** — ChromaDB on disk, survives restarts
- **Source citations** — every answer includes filename + page number
- **Conversational UI** — two-panel Gradio interface with multi-turn chat
- **Per-document summaries** — summarise any indexed document on demand
- **Persistent chat history** — conversation saved to disk, restored on restart
- **Document library** — view, filter, and delete indexed sources from the UI
- **Source-level retrieval filter** — choose which documents to query per conversation
- **Secure uploads** — path traversal protection, 100 MB limit, extension whitelist

---

## Quick Start

### Prerequisites

1. Install [Ollama](https://ollama.com/) and pull the required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Clone and install:
```bash
git clone https://github.com/Shen-po-heng/Polaris.git
cd Polaris
pip install -r requirements.txt
```

3. Configure (optional — defaults work out of the box):
```bash
cp .env.example .env
# edit .env if you want to change models or settings
```

4. Run:
```bash
# Make sure Ollama is running first
ollama serve

# Start Polaris
python app.py
```

Open `http://localhost:7860` in your browser.

---

## Docker

```bash
# CPU
docker compose up

# GPU (NVIDIA)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

## Architecture

```
polaris/
├── app.py                      # Entry point
├── config/
│   └── settings.py             # pydantic-settings, reads from .env
├── core/
│   ├── llm_provider.py         # LiteLLM wrapper (Ollama / OpenAI / Anthropic)
│   ├── embedder.py             # Embeddings via LiteLLM
│   ├── document_loader.py      # PDF / DOCX / TXT / MD loader
│   ├── chunker.py              # Text splitting
│   ├── vector_store.py         # ChromaDB persistent store
│   └── exceptions.py           # Domain exception hierarchy
├── services/
│   ├── rag_service.py          # RAG orchestration + source-filtered retrieval
│   ├── summarizer.py           # Per-document LLM summarisation
│   ├── chat_history.py         # JSON-backed conversation persistence
│   └── task_queue.py           # Async indexing queue
├── interfaces/
│   └── gradio_interface.py     # Two-tab Gradio UI (Chat + Library)
└── utils/
    ├── logger.py               # Structured logging
    └── security.py             # File validation
```

---

## Switching Models

Everything is controlled by `.env` — zero code changes needed:

```env
# Local (free, no API key)
LLM_MODEL=ollama/llama3.2
EMBEDDING_MODEL=ollama/nomic-embed-text

# OpenAI
LLM_MODEL=openai/gpt-4o-mini
OPENAI_API_KEY=sk-...

# Anthropic
LLM_MODEL=anthropic/claude-3-haiku-20240307
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Development

```bash
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Lint + format check
ruff check .
black --check .
```

---

## Roadmap

- [x] Phase 0 — Repository cleanup
- [x] Phase 1 — Core refactor (bug fixes, security, CI)
- [x] Phase 2 — Local-first foundation (Ollama, multi-format, persistent ChromaDB)
- [x] Phase 2.5 — Conversational UI (per-document summary, multi-turn chat, document library)
- [ ] Phase 3 — GraphRAG core (knowledge graph, entity extraction, pyvis visualisation)
- [ ] Phase 4 — Document CRUD + SQLite metadata
- [ ] Phase 5 — REST API + MCP server
- [ ] Phase 6 — RAGAS benchmark

---

## License

MIT License — see [LICENSE](LICENSE)

## Contributing

Issues and PRs welcome. If you find this useful, consider giving it a ⭐
