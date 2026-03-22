"""Unit tests for VectorStore.

ChromaDB client and Chroma wrapper are mocked so tests run without
a running database or embedding model.
"""

import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.exceptions import IndexingError
from core.vector_store import VectorStore


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(content="chunk text"):
    doc = SimpleNamespace()
    doc.page_content = content
    doc.metadata = {"source": "paper.pdf", "page": 1}
    return doc


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAdd:
    @patch("core.vector_store.chromadb.PersistentClient")
    @patch("core.vector_store.Chroma")
    def test_adds_chunks_successfully(self, MockChroma, MockClient):
        VectorStore._client = None  # reset singleton for test isolation
        vs = VectorStore()
        embedder = MagicMock()
        chunks = [_make_chunk()]

        vs.add(chunks, embedder)

        MockChroma.from_documents.assert_called_once()

    @patch("core.vector_store.chromadb.PersistentClient")
    @patch("core.vector_store.Chroma")
    def test_raises_on_chroma_failure(self, MockChroma, MockClient):
        VectorStore._client = None
        MockChroma.from_documents.side_effect = RuntimeError("disk full")

        vs = VectorStore()
        with pytest.raises(IndexingError, match="ChromaDB write failed"):
            vs.add([_make_chunk()], MagicMock())

    @patch("core.vector_store.chromadb.PersistentClient")
    def test_skips_empty_chunks(self, MockClient):
        VectorStore._client = None
        vs = VectorStore()
        # Should not raise and should not call ChromaDB
        vs.add([], MagicMock())


class TestAsRetriever:
    @patch("core.vector_store.chromadb.PersistentClient")
    @patch("core.vector_store.Chroma")
    def test_returns_retriever_after_add(self, MockChroma, MockClient):
        VectorStore._client = None
        mock_retriever = MagicMock()
        MockChroma.from_documents.return_value.as_retriever.return_value = mock_retriever

        vs = VectorStore()
        vs.add([_make_chunk()], MagicMock())
        retriever = vs.as_retriever()

        assert retriever is mock_retriever

    def test_raises_if_nothing_indexed(self):
        VectorStore._client = None
        vs = VectorStore()  # no embedder set, no add() called
        with pytest.raises(IndexingError, match="No documents indexed"):
            vs.as_retriever()
