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

    def as_retriever(
        self,
        k: int | None = None,
        sources: list[str] | None = None,
    ) -> VectorStoreRetriever:
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
        search_kwargs: dict = {"k": k or settings.search_k}
        if sources:
            search_kwargs["filter"] = {"source": {"$in": sources}}
        return chroma.as_retriever(search_kwargs=search_kwargs)

    def get_source_text(self, source: str, max_chunks: int = 20) -> str:
        """Return concatenated text of stored chunks for *source*.

        Args:
            source: Filename as stored in metadata (e.g. ``"paper.pdf"``).
            max_chunks: Maximum number of chunks to include (avoids LLM overflow).

        Returns:
            Concatenated chunk text, or empty string if not found.
        """
        try:
            client = self._get_client()
            collection = client.get_collection(_COLLECTION_NAME)
            results = collection.get(
                where={"source": source},
                include=["documents"],
                limit=max_chunks,
            )
            return "\n\n".join(results["documents"])
        except Exception as exc:
            logger.warning("get_source_text(%s) failed: %s", source, exc)
            return ""

    def list_sources(self) -> list[str]:
        """Return sorted list of unique source filenames in the collection."""
        try:
            client = self._get_client()
            collection = client.get_collection(_COLLECTION_NAME)
            results = collection.get(include=["metadatas"])
            sources = {
                meta["source"]
                for meta in results["metadatas"]
                if meta and "source" in meta
            }
            return sorted(sources)
        except Exception:
            return []

    def delete_by_sources(self, sources: list[str]) -> int:
        """Delete all chunks whose source is in *sources*.

        Returns:
            Number of chunks deleted.

        Raises:
            IndexingError: If the deletion fails.
        """
        if not sources:
            return 0
        try:
            client = self._get_client()
            collection = client.get_collection(_COLLECTION_NAME)
            results = collection.get(
                where={"source": {"$in": sources}},
                include=[],
            )
            ids = results["ids"]
            if ids:
                collection.delete(ids=ids)
            self._chroma = None  # invalidate cached Chroma wrapper
            logger.info("Deleted %d chunks for sources: %s", len(ids), sources)
            return len(ids)
        except Exception as exc:
            logger.error("delete_by_sources failed: %s", exc)
            raise IndexingError(f"Delete failed: {exc}") from exc

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
