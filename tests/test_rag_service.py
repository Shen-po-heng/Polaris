"""Unit tests for RAGService.

All external dependencies (LLMProvider, DocumentLoader, Embedder, VectorStore)
are mocked so tests run without Ollama or any ML packages installed.
"""

import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.exceptions import IndexingError, QueryError
from services.rag_service import RAGService


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_doc(content: str = "mock content", source: str = "paper.pdf", page: int = 1):
    doc = SimpleNamespace()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def rag_service():
    """RAGService with all core dependencies mocked."""
    with (
        patch("services.rag_service.LLMProvider") as MockLLM,
        patch("services.rag_service.Embedder"),
        patch("services.rag_service.Chunker"),
        patch("services.rag_service.VectorStore"),
    ):
        MockLLM.return_value.chat.return_value = "Mocked answer."
        yield RAGService()


# ── process_document ───────────────────────────────────────────────────────────

class TestProcessDocument:
    def test_returns_retriever_on_success(self, rag_service, tmp_path):
        fake_pdf = tmp_path / "paper.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        mock_retriever = MagicMock()
        rag_service._vector_store.as_retriever.return_value = mock_retriever

        with patch("services.rag_service.DocumentLoader.load") as mock_load:
            mock_load.return_value = [_make_doc()]
            rag_service._chunker.split.return_value = [_make_doc()]

            result = rag_service.process_document([str(fake_pdf)])

        assert result is mock_retriever

    def test_raises_indexing_error_on_loader_failure(self, rag_service, tmp_path):
        with patch("services.rag_service.DocumentLoader.load") as mock_load:
            mock_load.side_effect = IndexingError("corrupt PDF")

            with pytest.raises(IndexingError):
                rag_service.process_document(["bad.pdf"])

    def test_processes_multiple_files(self, rag_service, tmp_path):
        files = [tmp_path / f"paper{i}.pdf" for i in range(3)]
        for f in files:
            f.write_bytes(b"%PDF-1.4")

        with patch("services.rag_service.DocumentLoader.load") as mock_load:
            mock_load.return_value = [_make_doc()]
            rag_service._chunker.split.return_value = [_make_doc()]

            rag_service.process_document([str(f) for f in files])

            assert mock_load.call_count == 3


# ── answer_query ───────────────────────────────────────────────────────────────

class TestAnswerQuery:
    def test_returns_answer_with_citations(self, rag_service):
        source_doc = _make_doc(source="paper.pdf", page=3)
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [source_doc]

        with patch.object(rag_service, "process_document", return_value=mock_retriever):
            rag_service._llm.chat.return_value = "GPS achieves 1m accuracy."
            result = rag_service.answer_query(["paper.pdf"], "How accurate is GPS?")

        assert "GPS achieves 1m accuracy." in result
        assert "paper.pdf" in result
        assert "Page 3" in result

    def test_answer_without_sources(self, rag_service):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        with patch.object(rag_service, "process_document", return_value=mock_retriever):
            rag_service._llm.chat.return_value = "Direct answer."
            result = rag_service.answer_query(["paper.pdf"], "Test question?")

        assert result == "Direct answer."
        assert "Sources" not in result

    def test_raises_query_error_on_unexpected_failure(self, rag_service):
        with patch.object(rag_service, "process_document") as mock_process:
            mock_process.side_effect = Exception("unexpected")

            with pytest.raises(QueryError):
                rag_service.answer_query(["paper.pdf"], "question?")


# ── answer_with_history ────────────────────────────────────────────────────────

class TestAnswerWithHistory:
    def test_includes_history_in_prompt(self, rag_service):
        source_doc = _make_doc(source="paper.pdf", page=1)
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [source_doc]

        history = [("What is GPS?", "GPS is a navigation system.")]

        with patch.object(rag_service, "process_document", return_value=mock_retriever):
            rag_service._llm.chat.return_value = "Follow-up answer."
            result = rag_service.answer_with_history(
                ["paper.pdf"], "How accurate is it?", history
            )

        assert "Follow-up answer." in result
        # Verify that the LLM was called with history context
        prompt_used = rag_service._llm.chat.call_args[0][0]
        assert "What is GPS?" in prompt_used
        assert "GPS is a navigation system." in prompt_used

    def test_no_history_uses_simple_prompt(self, rag_service):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        with patch.object(rag_service, "process_document", return_value=mock_retriever):
            rag_service._llm.chat.return_value = "Answer."
            result = rag_service.answer_with_history(["paper.pdf"], "Question?", [])

        assert result == "Answer."
        prompt_used = rag_service._llm.chat.call_args[0][0]
        assert "Conversation so far" not in prompt_used

    def test_raises_query_error_on_failure(self, rag_service):
        with patch.object(rag_service, "process_document") as mock_process:
            mock_process.side_effect = Exception("boom")

            with pytest.raises(QueryError):
                rag_service.answer_with_history(["paper.pdf"], "q?", [])
