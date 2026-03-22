"""Chunker — splits LangChain Documents into overlapping text chunks.

Extracted from RAGService so the splitting logic can be tested and reused
independently (e.g., by the indexing pipeline in Phase 3).

Usage:
    from core.chunker import Chunker
    chunks = Chunker().split(documents)
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Chunker:
    """Wraps RecursiveCharacterTextSplitter with project-level defaults."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks, preserving metadata.

        Args:
            documents: Raw LangChain Document objects (one per page / file).

        Returns:
            List of smaller Document chunks with the same metadata.
        """
        chunks = self._splitter.split_documents(documents)
        logger.debug(
            "Split %d document(s) into %d chunk(s)",
            len(documents),
            len(chunks),
        )
        return chunks
