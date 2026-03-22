"""Per-document summarisation service.

Loads each file independently, concatenates a representative sample of its
text, then asks the LLM for a concise summary in the same language as the
document.
"""

from __future__ import annotations

from core.document_loader import DocumentLoader
from core.llm_provider import LLMProvider
from core.exceptions import PolarisError
from utils.logger import get_logger

logger = get_logger(__name__)

_SUMMARY_PROMPT = """\
You are a research assistant.
Summarise the following document excerpt in 3–5 sentences.
Focus on the main topic, key findings, and conclusions.
Write in the same language as the document.

Document ({filename}):
{text}

Summary:"""

# Maximum characters fed to the LLM (avoids context overflow)
_MAX_CHARS = 8_000


class Summarizer:
    """Generate a short summary for each uploaded file."""

    def __init__(self) -> None:
        self._llm = LLMProvider()

    def summarise(self, file_paths: list[str]) -> dict[str, str]:
        """Return a mapping of filename → summary for each path.

        Args:
            file_paths: Absolute paths to documents.

        Returns:
            ``{"report.pdf": "This paper explores …", …}``
        """
        results: dict[str, str] = {}
        for path in file_paths:
            try:
                docs = DocumentLoader.load(path)
                full_text = "\n\n".join(d.page_content for d in docs)
                sample = full_text[:_MAX_CHARS]
                from pathlib import Path
                filename = Path(path).name
                prompt = _SUMMARY_PROMPT.format(filename=filename, text=sample)
                summary = self._llm.chat(prompt)
                results[filename] = summary
                logger.info("Summarised %s", filename)
            except PolarisError as exc:
                logger.error("Could not summarise %s: %s", path, exc)
                results[path] = f"[Error: {exc}]"
            except Exception as exc:
                logger.exception("Unexpected error summarising %s", path)
                results[path] = f"[Error: {exc}]"
        return results
