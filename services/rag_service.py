"""RAG service — document indexing and question answering.

Phase 2.5: index and query are now separate operations.
  - index_documents() — load/chunk/store to ChromaDB, return source names
  - query()           — retrieve + LLM call with optional source filter + history

Backward-compat wrappers (process_document, answer_query, answer_with_history)
are kept for existing tests and callers.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from core.chunker import Chunker
from core.document_loader import DocumentLoader
from core.embedder import Embedder
from core.exceptions import DocumentLoadError, IndexingError, QueryError
from core.llm_provider import LLMProvider
from core.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

_RAG_PROMPT = """\
Use the following document excerpts to answer the question.
If the answer is not in the excerpts, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

_RAG_HISTORY_PROMPT = """\
Use the following document excerpts to answer the question.
If the answer is not in the excerpts, say "I don't have enough information."

Context:
{context}

Conversation so far:
{history}

Question: {question}

Answer:"""


class RAGService:
    """Orchestrates document ingestion and retrieval-augmented generation."""

    def __init__(self) -> None:
        self._llm = LLMProvider()
        self._embedder = Embedder()
        self._chunker = Chunker()
        self._vector_store = VectorStore(embedder=self._embedder)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_documents(self, file_paths: list[str]) -> list[str]:
        """Load, chunk, and index documents into the persistent vector store.

        Args:
            file_paths: Absolute paths to documents (PDF / DOCX / TXT / MD).

        Returns:
            List of source filenames that were successfully indexed.

        Raises:
            IndexingError: If loading or indexing fails.
        """
        try:
            all_docs: list[Document] = []
            for raw_path in file_paths:
                docs = DocumentLoader.load(raw_path)
                all_docs.extend(docs)

            chunks = self._chunker.split(all_docs)
            self._vector_store.add(chunks, self._embedder)
            source_names = sorted({Path(p).name for p in file_paths})
            logger.info("Indexed %d sources: %s", len(source_names), source_names)
            return source_names

        except (DocumentLoadError, IndexingError):
            raise
        except Exception as exc:
            logger.exception("Unexpected error during document indexing")
            raise IndexingError(f"Indexing failed: {exc}") from exc

    # ── Querying ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        history: list[list[str]] | None = None,
        selected_sources: list[str] | None = None,
    ) -> str:
        """Retrieve relevant chunks and generate an answer.

        Args:
            question: The user's question.
            history: List of [user_msg, assistant_msg] pairs (last N turns).
            selected_sources: If provided, only search within these sources.

        Returns:
            Answer text with source citations appended.

        Raises:
            QueryError: On unexpected failures.
        """
        try:
            retriever = self._vector_store.as_retriever(
                sources=selected_sources or None
            )
            source_docs: list[Document] = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in source_docs)

            history_text = "\n".join(
                f"User: {u}\nAssistant: {a}" for u, a in (history or [])
            )

            if history_text:
                prompt = _RAG_HISTORY_PROMPT.format(
                    context=context,
                    history=history_text,
                    question=question,
                )
            else:
                prompt = _RAG_PROMPT.format(context=context, question=question)

            answer = self._llm.chat(prompt)

            source_info: list[str] = []
            for doc in source_docs:
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "unknown")
                source_info.append(f"{source} (Page {page})")

            citation_text = (
                "\n\nSources:\n" + "\n".join(sorted(set(source_info)))
                if source_info
                else ""
            )
            return answer + citation_text

        except Exception as exc:
            logger.exception("Error answering query")
            raise QueryError(str(exc)) from exc

    def list_sources(self) -> list[str]:
        """Return all unique source filenames currently in the vector store."""
        return self._vector_store.list_sources()

    def get_source_text(self, source: str) -> str:
        """Return stored chunk text for *source* (for summarisation)."""
        return self._vector_store.get_source_text(source)

    def delete_sources(self, sources: list[str]) -> int:
        """Delete all chunks for the given sources. Returns chunk count deleted."""
        return self._vector_store.delete_by_sources(sources)

    # ── Backward-compat wrappers ──────────────────────────────────────────────

    def process_document(self, file_paths: list[str]) -> object:
        """[Compat] Index documents and return a retriever."""
        self.index_documents(file_paths)
        return self._vector_store.as_retriever()

    def answer_query(self, file_objs: list[str], question: str) -> str:
        """[Compat] Index files then answer query."""
        try:
            retriever = self.process_document(file_objs)
            source_docs: list[Document] = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            answer = self._llm.chat(
                _RAG_PROMPT.format(context=context, question=question)
            )
            source_info = [
                f"{d.metadata.get('source','?')} (Page {d.metadata.get('page','?')})"
                for d in source_docs
            ]
            citation = (
                "\n\nSources:\n" + "\n".join(sorted(set(source_info)))
                if source_info
                else ""
            )
            return answer + citation
        except (IndexingError, DocumentLoadError):
            raise
        except Exception as exc:
            raise QueryError(str(exc)) from exc

    def answer_with_history(
        self,
        file_objs: list[str],
        question: str,
        history: list[tuple[str, str]],
    ) -> str:
        """[Compat] Index files then answer query with history."""
        try:
            retriever = self.process_document(file_objs)
            source_docs: list[Document] = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            history_text = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
            prompt = (
                _RAG_HISTORY_PROMPT.format(
                    context=context, history=history_text, question=question
                )
                if history_text
                else _RAG_PROMPT.format(context=context, question=question)
            )
            answer = self._llm.chat(prompt)
            source_info = [
                f"{d.metadata.get('source','?')} (Page {d.metadata.get('page','?')})"
                for d in source_docs
            ]
            citation = (
                "\n\nSources:\n" + "\n".join(sorted(set(source_info)))
                if source_info
                else ""
            )
            return answer + citation
        except (IndexingError, DocumentLoadError):
            raise
        except Exception as exc:
            raise QueryError(str(exc)) from exc
