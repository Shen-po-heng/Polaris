"""Centralised settings loaded from environment variables / .env file.

Usage:
    from config.settings import settings
    print(settings.chunk_size)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ────────────────────────────────────────────────────────────────
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 2048
    repetition_penalty: float = 1.15

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ───────────────────────────────────────────────────────────
    search_k: int = 5

    # ── Security ────────────────────────────────────────────────────────────
    max_file_size_bytes: int = 100 * 1024 * 1024  # 100 MB

    # ── Server ─────────────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 7860

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"


settings = Settings()
