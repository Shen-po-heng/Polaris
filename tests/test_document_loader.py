"""Unit tests for DocumentLoader.

All file-system and library interactions are mocked where needed.
"""

import sys
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.document_loader import DocumentLoader
from core.exceptions import DocumentLoadError


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_doc(content="hello", source="paper.pdf", page=0):
    doc = SimpleNamespace()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


# ── PDF ───────────────────────────────────────────────────────────────────────

class TestLoadPDF:
    @patch("core.document_loader.validate_file")
    @patch("core.document_loader.PyPDFLoader")
    def test_returns_documents(self, MockLoader, mock_validate, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        mock_validate.return_value = pdf
        MockLoader.return_value.load.return_value = [_make_doc(source="paper.pdf")]

        docs = DocumentLoader.load(str(pdf))
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "paper.pdf"

    @patch("core.document_loader.validate_file")
    @patch("core.document_loader.PyPDFLoader")
    def test_raises_on_corrupt_pdf(self, MockLoader, mock_validate, tmp_path):
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"not a pdf")
        mock_validate.return_value = pdf
        MockLoader.return_value.load.side_effect = RuntimeError("corrupt")

        with pytest.raises(DocumentLoadError):
            DocumentLoader.load(str(pdf))


# ── TXT / MD ─────────────────────────────────────────────────────────────────

class TestLoadPlainText:
    @patch("core.document_loader.validate_file")
    def test_loads_txt(self, mock_validate, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("Hello from txt", encoding="utf-8")
        mock_validate.return_value = txt

        docs = DocumentLoader.load(str(txt))
        assert len(docs) == 1
        assert "Hello from txt" in docs[0].page_content
        assert docs[0].metadata["source"] == "notes.txt"

    @patch("core.document_loader.validate_file")
    def test_loads_md(self, mock_validate, tmp_path):
        md = tmp_path / "readme.md"
        md.write_text("# Title\nContent", encoding="utf-8")
        mock_validate.return_value = md

        docs = DocumentLoader.load(str(md))
        assert "Title" in docs[0].page_content


# ── DOCX ─────────────────────────────────────────────────────────────────────

class TestLoadDOCX:
    @patch("core.document_loader.validate_file")
    def test_raises_if_python_docx_missing(self, mock_validate, tmp_path):
        docx = tmp_path / "report.docx"
        docx.write_bytes(b"fake")
        mock_validate.return_value = docx

        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(DocumentLoadError, match="python-docx"):
                DocumentLoader.load(str(docx))


# ── Unsupported format ───────────────────────────────────────────────────────

class TestUnsupportedFormat:
    @patch("core.document_loader.validate_file")
    def test_raises_on_unknown_extension(self, mock_validate, tmp_path):
        xlsx = tmp_path / "data.xlsx"
        xlsx.write_bytes(b"fake")
        mock_validate.return_value = xlsx

        with pytest.raises(DocumentLoadError, match="Unsupported"):
            DocumentLoader.load(str(xlsx))
