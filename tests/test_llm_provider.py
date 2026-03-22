"""Unit tests for LLMProvider.

All network calls (litellm.completion, urllib.request.urlopen) are mocked
so tests run without Ollama running.
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_provider import LLMProvider, LLMProviderError

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_litellm_response(text: str) -> MagicMock:
    """Build a mock litellm completion response."""
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def provider():
    """LLMProvider with Ollama health-check bypassed."""
    with patch("core.llm_provider.urllib.request.urlopen"):
        yield LLMProvider()


# ── Tests ────────────────────────────────────────────────────────────────────


class TestLLMProviderInit:
    def test_raises_when_ollama_unreachable(self):
        import urllib.error

        with patch("core.llm_provider.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("connection refused")
            with pytest.raises(LLMProviderError, match="Cannot reach Ollama"):
                LLMProvider()

    def test_no_health_check_for_api_models(self):
        """Non-ollama models skip the Ollama reachability check."""
        with patch("core.llm_provider.urllib.request.urlopen") as mock_open:
            with patch("core.llm_provider.settings") as mock_settings:
                mock_settings.llm_model = "openai/gpt-4o-mini"
                mock_settings.ollama_base_url = "http://localhost:11434"
                mock_settings.llm_timeout_seconds = 30
                LLMProvider()
            mock_open.assert_not_called()


class TestChat:
    @patch("core.llm_provider.litellm.completion")
    def test_returns_model_reply(self, mock_completion, provider):
        mock_completion.return_value = _make_litellm_response("42 is the answer.")
        result = provider.chat("What is the answer?")
        assert result == "42 is the answer."

    @patch("core.llm_provider.litellm.completion")
    def test_includes_system_message(self, mock_completion, provider):
        mock_completion.return_value = _make_litellm_response("OK")
        provider.chat("Hello", system="You are a helpful assistant.")
        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("core.llm_provider.litellm.completion")
    def test_raises_on_api_failure(self, mock_completion, provider):
        mock_completion.side_effect = RuntimeError("timeout")
        with pytest.raises(LLMProviderError, match="LLM call failed"):
            provider.chat("question?")


class TestStream:
    @patch("core.llm_provider.litellm.completion")
    def test_yields_chunks(self, mock_completion, provider):
        def _make_chunk(text):
            c = MagicMock()
            c.choices[0].delta.content = text
            return c

        mock_completion.return_value = iter(
            [_make_chunk("Hello"), _make_chunk(" world")]
        )
        chunks = list(provider.stream("Say hello"))
        assert chunks == ["Hello", " world"]

    @patch("core.llm_provider.litellm.completion")
    def test_raises_on_stream_failure(self, mock_completion, provider):
        mock_completion.side_effect = RuntimeError("stream error")
        with pytest.raises(LLMProviderError, match="LLM stream failed"):
            list(provider.stream("question?"))
