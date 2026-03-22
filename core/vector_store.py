"""Vector store — persistent ChromaDB singleton.

Chunks are stored on disk under ``data/chroma/`` so indexing survives
process restarts.  The same collection is reused across queries instead of
being rebuilt from scratch every time (Phase 1 limitation).

Usage:
    from core.vector_store import VectorStore
    vs = VectorStore()
    vs.add(chunks, embedder)
    retriever = vs.as_retriever()
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from core.exceptions import IndexingError
from utils.logger import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "polaris_docs"
_PERSIST_DIR = Path("data") / "chroma"


class VectorStore:
    """Persistent ChromaDB-backed vector store.

    A single ``chromadb.PersistentClient`` is created per process and
    reused across all ``VectorStore`` instances (module-level singleton),
    preventing repeated disk I/O on every query.
    """

    _client: chromadb.ClientAPI | None = None  # module-level singleton

    def __init__(self, embedder: Embeddings | None = None) -> None:
        self._embedder = embedder  # can be set later via add()
        self._chroma: Chroma | None = None
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def add(self, chunks: list[Document], embedder: Embeddings) -> None:
        """Embed chunks and persist them to ChromaDB.

        Args:
            chunks: Document chunks to index.
            embedder: Embeddings implementation (e.g., ``Embedder()``).

        Raises:
            IndexingError: If ChromaDB write fails.
        """
        if not chunks:
            logger.warning("add() called with empty chunk list — nothing indexed")
            return

        self._embedder = embedder
        try:
            self._chroma = Chroma.from_documents(
                documents=chunks,
                embedding=embedder,
                client=self._get_client(),
                collection_name=_COLLECTION_NAME,
            )
            logger.info("Indexed %d chunks into ChromaDB", len(chunks))
        except Exception as exc:
            logger.error("ChromaDB write failed: %s", exc)
            raise IndexingError(f"ChromaDB write failed: {exc}") from exc

    def as_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        """Return a LangChain retriever over the persisted collection.

        Args:
            k: Number of nearest-neighbour results to return.
               Defaults to ``settings.search_k``.

        Returns:
            A LangChain ``VectorStoreRetriever``.

        Raises:
            IndexingError: If no documents have been indexed yet.
        """
        chroma = self._get_or_load_chroma()
        return chroma.as_retriever(search_kwargs={"k": k or settings.search_k})

    def clear(self) -> None:
        """Delete all documents from the collection (for testing / reset)."""
        try:
            client = self._get_client()
            client.delete_collection(_COLLECTION_NAME)
            self._chroma = None
            logger.info("ChromaDB collection '%s' cleared", _COLLECTION_NAME)
        except Exception as exc:
            logger.warning("Failed to clear collection: %s", exc)

    # ── Private helpers ─────────────────────────────────────────────────────

    @classmethod
    def _get_client(cls) -> chromadb.ClientAPI:
        if cls._client is None:
            cls._client = chromadb.PersistentClient(path=str(_PERSIST_DIR))
            logger.debug("ChromaDB PersistentClient created at %s", _PERSIST_DIR)
        return cls._client

    def _get_or_load_chroma(self) -> Chroma:
        """Return the in-memory Chroma wrapper, loading from disk if needed."""
        if self._chroma is None:
            if self._embedder is None:
                raise IndexingError(
                    "No documents indexed yet. Call add() before as_retriever()."
                )
            self._chroma = Chroma(
                client=self._get_client(),
                collection_name=_COLLECTION_NAME,
                embedding_function=self._embedder,
            )
        return self._chroma
