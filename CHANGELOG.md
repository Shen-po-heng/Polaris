# Changelog

All notable changes to Polaris will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- GraphRAG pipeline with hybrid entity extraction (spaCy + LLM)
- Local model support via Ollama
- Knowledge graph visualization (Pyvis)
- FastAPI REST API layer
- MCP Server for Claude Desktop / Cursor integration
- Document CRUD interface
- arXiv / DOI import
- Answer validation pipeline

---

## [0.1.0] - 2025-03-17

### Added
- Multi-PDF upload and ingestion
- RAG-based question answering with source citations (filename + page)
- Vector store with ChromaDB
- Sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- TinyLlama local LLM inference
- Gradio web interface
- Docker support

### Changed
- Refactor: modular architecture (models/, services/, interfaces/, utils/)

### Removed
- Legacy monolithic implementations (`Legacy/` folder archived)
- Stale branches: `add-summarizer`, `Add-Multiple-PDF-support`
