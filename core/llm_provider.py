"""LLM provider — thin LiteLLM wrapper with Ollama health-check.

Supports:
  - Ollama (local, free): ollama/llama3.2
  - OpenAI:               openai/gpt-4o-mini
  - Anthropic:            anthropic/claude-3-haiku-20240307

Switch models by changing LLM_MODEL in .env — no code changes needed.

Usage:
    from core.llm_provider import LLMProvider
    provider = LLMProvider()
    response = provider.chat("Explain GPS in one sentence.")
"""

from __future__ import annotations

import urllib.request
import urllib.error
from typing import Iterator

import litellm

from config.settings import settings
from core.exceptions import PolarisError
from utils.logger import get_logger

logger = get_logger(__name__)

# Silence LiteLLM's verbose output; Polaris controls its own logging.
litellm.suppress_debug_info = True


class LLMProviderError(PolarisError):
    """Raised when the LLM backend is unavailable or returns an error."""


class LLMProvider:
    """Stateless wrapper around LiteLLM.

    All configuration is read from ``config.settings`` at instantiation time,
    so swapping models only requires changing the .env file.
    """

    def __init__(self) -> None:
        self.model: str = settings.llm_model
        self.api_base: str | None = (
            settings.ollama_base_url if self._is_ollama() else None
        )
        self._check_backend()

    # ── Public API ──────────────────────────────────────────────────────────

    def chat(self, prompt: str, *, system: str = "") -> str:
        """Single-turn completion.

        Args:
            prompt: The user message.
            system: Optional system prompt.

        Returns:
            The model's reply as a plain string.

        Raises:
            LLMProviderError: If the API call fails.
        """
        messages = self._build_messages(prompt, system)
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_base=self.api_base,
                timeout=settings.llm_timeout_seconds,
            )
            content: str = response.choices[0].message.content or ""
            logger.debug("LLM response (%d chars)", len(content))
            return content
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise LLMProviderError(f"LLM call failed: {exc}") from exc

    def stream(self, prompt: str, *, system: str = "") -> Iterator[str]:
        """Streaming completion — yields text chunks as they arrive.

        Args:
            prompt: The user message.
            system: Optional system prompt.

        Yields:
            Incremental text chunks from the model.

        Raises:
            LLMProviderError: If the API call fails.
        """
        messages = self._build_messages(prompt, system)
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_base=self.api_base,
                timeout=settings.llm_timeout_seconds,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("LLM stream failed: %s", exc)
            raise LLMProviderError(f"LLM stream failed: {exc}") from exc

    # ── Private helpers ─────────────────────────────────────────────────────

    def _is_ollama(self) -> bool:
        return self.model.startswith("ollama/")

    def _check_backend(self) -> None:
        """Verify the backend is reachable at startup.

        Gives a clear, actionable error message instead of a cryptic timeout
        later during the first query.
        """
        if not self._is_ollama():
            logger.info("LLM backend: %s (API key required)", self.model)
            return

        url = f"{settings.ollama_base_url}/api/tags"
        try:
            urllib.request.urlopen(url, timeout=3)  # noqa: S310
            logger.info(
                "Ollama is running at %s — model: %s",
                settings.ollama_base_url,
                self.model,
            )
        except urllib.error.URLError as exc:
            msg = (
                f"Cannot reach Ollama at {settings.ollama_base_url}.\n"
                "  → Start Ollama: run 'ollama serve' in a terminal\n"
                f"  → Then pull the model: ollama pull {self.model.removeprefix('ollama/')}\n"
                f"  Original error: {exc}"
            )
            logger.error(msg)
            raise LLMProviderError(msg) from exc

    @staticmethod
    def _build_messages(prompt: str, system: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
