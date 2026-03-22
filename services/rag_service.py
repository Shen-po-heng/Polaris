"""RAG service — document indexing and question answering.

Phase 2 rewrite: uses LiteLLM directly (no LangChain LLM wrapper needed),
LCEL-style retrieval, and persistent ChromaDB.

Responsibilities:
  - Accept file paths, delegate loading → chunking → indexing to core/.
  - Answer user queries with source citations.
"""

from __future__ import annotations

from langchain_core.documents import Document

from core.chunker import Chunker
from core.document_loader import DocumentLoader
from core.embedder import Embedder
from core.exceptions import DocumentLoadError, IndexingError, QueryError
from core.llm_provider import LLMProvider
from core.vector_store import VectorStore
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_RAG_PROMPT = """\
Use the following document excerpts to answer the question.
If the answer is not in the excerpts, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    """Orchestrates document ingestion and retrieval-augmented generation."""

    def __init__(self) -> None:
        self._llm = LLMProvider()
        self._embedder = Embedder()
        self._chunker = Chunker()
        self._vector_store = VectorStore(embedder=self._embedder)

    # ── Public API ──────────────────────────────────────────────────────────

    def process_document(self, file_paths: list[str]) -> object:
        """Load, chunk, and index documents into the persistent vector store.

        Args:
            file_paths: Absolute paths to documents (PDF / DOCX / TXT / MD).

        Returns:
            A LangChain retriever ready for similarity search.

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
            return self._vector_store.as_retriever()

        except (DocumentLoadError, IndexingError):
            raise
        except Exception as exc:
            logger.exception("Unexpected error during document processing")
            raise IndexingError(f"Document processing failed: {exc}") from exc

    def answer_query(self, file_objs: list[str], query: str) -> str:
        """Answer a query using RAG over the provided documents.

        Args:
            file_objs: File paths (from Gradio upload or direct call).
            query: The user's question.

        Returns:
            Answer text with source citations appended.

        Raises:
            QueryError: If the answer step fails unexpectedly.
        """
        try:
            retriever = self.process_document(file_objs)

            # Retrieve relevant chunks
            source_docs: list[Document] = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in source_docs)

            # Call LiteLLM directly — no LangChain chain needed
            prompt = _RAG_PROMPT.format(context=context, question=query)
            answer = self._llm.chat(prompt)

            # Build citation list
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

        except (IndexingError, DocumentLoadError):
            raise
        except Exception as exc:
            logger.exception("Error answering query")
            raise QueryError(str(exc)) from exc
