"""Unit tests for Summarizer service."""

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.summarizer import Summarizer
from core.exceptions import DocumentLoadError


@pytest.fixture()
def summarizer():
    with patch("services.summarizer.LLMProvider") as MockLLM:
        MockLLM.return_value.chat.return_value = "Mocked summary."
        yield Summarizer()


class TestSummarizer:
    def test_returns_summary_for_each_file(self, summarizer, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch("services.summarizer.DocumentLoader.load") as mock_load:
            from types import SimpleNamespace
            doc = SimpleNamespace(page_content="Some text about GPS.")
            mock_load.return_value = [doc]
            result = summarizer.summarise([str(pdf)])

        assert "paper.pdf" in result
        assert result["paper.pdf"] == "Mocked summary."

    def test_multiple_files_all_summarised(self, summarizer, tmp_path):
        files = [tmp_path / f"doc{i}.pdf" for i in range(3)]
        for f in files:
            f.write_bytes(b"%PDF-1.4")

        with patch("services.summarizer.DocumentLoader.load") as mock_load:
            from types import SimpleNamespace
            mock_load.return_value = [SimpleNamespace(page_content="text")]
            result = summarizer.summarise([str(f) for f in files])

        assert len(result) == 3

    def test_error_stored_in_result(self, summarizer, tmp_path):
        with patch("services.summarizer.DocumentLoader.load") as mock_load:
            mock_load.side_effect = DocumentLoadError("corrupt")
            result = summarizer.summarise(["bad.pdf"])

        assert "bad.pdf" in result
        assert "[Error:" in result["bad.pdf"]

    def test_text_truncated_to_max_chars(self, summarizer, tmp_path):
        """Summarizer must not pass more than _MAX_CHARS to the LLM."""
        long_text = "x" * 20_000
        with patch("services.summarizer.DocumentLoader.load") as mock_load:
            from types import SimpleNamespace
            mock_load.return_value = [SimpleNamespace(page_content=long_text)]
            summarizer.summarise(["doc.txt"])

        called_prompt = summarizer._llm.chat.call_args[0][0]
        # The prompt contains the text sample — verify it's bounded
        assert len(called_prompt) < 15_000
