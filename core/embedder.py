"""Embedder — wraps LiteLLM embedding endpoint.

Uses nomic-embed-text via Ollama by default (274 MB, runs fully local).
Switch to OpenAI text-embedding-3-small by changing EMBEDDING_MODEL in .env.

Usage:
    from core.embedder import Embedder
    embedder = Embedder()
    vectors = embedder.embed_documents(["chunk1", "chunk2"])
    query_vec = embedder.embed_query("What is GPS?")
"""

from __future__ import annotations

import litellm
from langchain_core.embeddings import Embeddings

from config.settings import settings
from core.exceptions import IndexingError
from utils.logger import get_logger

logger = get_logger(__name__)

litellm.suppress_debug_info = True


class Embedder(Embeddings):
    """LangChain-compatible embeddings class backed by LiteLLM.

    Implements the ``Embeddings`` interface so it can be dropped directly
    into any LangChain vector store (Chroma, FAISS, …).
    """

    def __init__(self) -> None:
        self.model: str = settings.embedding_model
        self._api_base: str | None = (
            settings.ollama_base_url if self.model.startswith("ollama/") else None
        )
        logger.info("Embedder initialised — model: %s", self.model)

    # ── LangChain Embeddings interface ──────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks.

        Args:
            texts: Raw text strings (chunks) to embed.

        Returns:
            List of embedding vectors (one per text).

        Raises:
            IndexingError: If the embedding call fails.
        """
        if not texts:
            return []
        try:
            response = litellm.embedding(
                model=self.model,
                input=texts,
                api_base=self._api_base,
            )
            vectors = [item["embedding"] for item in response.data]
            logger.debug("Embedded %d chunks", len(vectors))
            return vectors
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise IndexingError(f"Embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The user's query.

        Returns:
            Embedding vector.

        Raises:
            IndexingError: If the embedding call fails.
        """
        results = self.embed_documents([text])
        return results[0]
