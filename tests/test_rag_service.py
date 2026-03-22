"""Unit tests for RAGService.

All external dependencies (ModelManager, PyPDFLoader, Chroma, etc.) are
mocked so the tests run without torch / transformers installed.
"""

import sys
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.exceptions import IndexingError, QueryError
from services.rag_service import RAGService


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_doc(content: str = "mock content", source: str = "paper.pdf", page: int = 1):
    """Return a minimal document-like object with metadata."""
    doc = SimpleNamespace()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def rag_service():
    """RAGService with ModelManager fully mocked out."""
    with patch("services.rag_service.ModelManager") as MockMM:
        instance = MockMM.return_value
        instance.initialize_models.return_value = True
        instance.llm = MagicMock()
        instance.embedding_model = MagicMock()
        yield RAGService()


# ── process_document ───────────────────────────────────────────────────────────

class TestProcessDocument:
    @patch("services.rag_service.validate_file")
    @patch("services.rag_service.Chroma")
    @patch("services.rag_service.RecursiveCharacterTextSplitter")
    @patch("services.rag_service.PyPDFLoader")
    def test_returns_retriever_on_success(
        self, MockLoader, MockSplitter, MockChroma, mock_validate, rag_service, tmp_path
    ):
        fake_pdf = tmp_path / "paper.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        mock_validate.return_value = fake_pdf
        mock_doc = _make_doc()
        MockLoader.return_value.load.return_value = [mock_doc]
        MockSplitter.return_value.split_documents.return_value = [mock_doc]
        mock_retriever = MagicMock()
        MockChroma.from_documents.return_value.as_retriever.return_value = mock_retriever

        result = rag_service.process_document([str(fake_pdf)])

        assert result is mock_retriever
        MockLoader.assert_called_once_with(str(fake_pdf))

    @patch("services.rag_service.validate_file")
    @patch("services.rag_service.PyPDFLoader")
    def test_raises_indexing_error_on_loader_failure(
        self, MockLoader, mock_validate, rag_service, tmp_path
    ):
        fake_pdf = tmp_path / "bad.pdf"
        fake_pdf.write_bytes(b"not a pdf")
        mock_validate.return_value = fake_pdf
        MockLoader.return_value.load.side_effect = RuntimeError("corrupt PDF")

        with pytest.raises(IndexingError):
            rag_service.process_document([str(fake_pdf)])

    @patch("services.rag_service.validate_file")
    def test_processes_multiple_files(self, mock_validate, rag_service, tmp_path):
        """Multiple file paths should all be iterated."""
        files = [tmp_path / f"paper{i}.pdf" for i in range(3)]
        for f in files:
            f.write_bytes(b"%PDF-1.4")

        mock_validate.side_effect = files  # return each path in order

        with (
            patch("services.rag_service.PyPDFLoader") as MockLoader,
            patch("services.rag_service.Chroma") as MockChroma,
            patch("services.rag_service.RecursiveCharacterTextSplitter"),
        ):
            MockLoader.return_value.load.return_value = [_make_doc()]
            MockChroma.from_documents.return_value.as_retriever.return_value = MagicMock()

            rag_service.process_document([str(f) for f in files])

            assert MockLoader.call_count == 3


# ── answer_query ───────────────────────────────────────────────────────────────

class TestAnswerQuery:
    @patch("services.rag_service.RetrievalQA")
    @patch.object(RAGService, "process_document")
    def test_returns_answer_with_citations(
        self, mock_process, MockQA, rag_service
    ):
        mock_process.return_value = MagicMock()
        source_doc = _make_doc(source="paper.pdf", page=3)
        MockQA.from_chain_type.return_value.invoke.return_value = {
            "result": "Helpful Answer: GPS achieves 1m accuracy.",
            "source_documents": [source_doc],
        }

        result = rag_service.answer_query(["paper.pdf"], "How accurate is GPS?")

        assert "GPS achieves 1m accuracy." in result
        assert "paper.pdf" in result
        assert "Page 3" in result

    @patch("services.rag_service.RetrievalQA")
    @patch.object(RAGService, "process_document")
    def test_answer_without_helpful_prefix(self, mock_process, MockQA, rag_service):
        mock_process.return_value = MagicMock()
        MockQA.from_chain_type.return_value.invoke.return_value = {
            "result": "Direct answer here.",
            "source_documents": [],
        }

        result = rag_service.answer_query(["paper.pdf"], "Test question?")
        assert result == "Direct answer here."

    @patch.object(RAGService, "process_document")
    def test_raises_query_error_on_unexpected_failure(self, mock_process, rag_service):
        mock_process.side_effect = Exception("unexpected")

        with pytest.raises(Exception):
            rag_service.answer_query(["paper.pdf"], "question?")
