"""Persistent chat history stored as JSON on disk.

History is a list of [user_msg, assistant_msg] pairs (Gradio Chatbot format).
Saved to ``data/chat_history.json`` after every message.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_HISTORY_FILE = Path("data") / "chat_history.json"


class ChatHistoryManager:
    """Load and save conversation history from/to disk."""

    @staticmethod
    def load() -> list[list[str]]:
        """Return saved history, or [] if none exists."""
        if not _HISTORY_FILE.exists():
            return []
        try:
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Could not load chat history: %s", exc)
        return []

    @staticmethod
    def save(history: list[list[str]]) -> None:
        """Persist history to disk (silent on failure)."""
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _HISTORY_FILE.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save chat history: %s", exc)

    @staticmethod
    def clear() -> None:
        """Delete the history file."""
        try:
            if _HISTORY_FILE.exists():
                _HISTORY_FILE.unlink()
        except Exception as exc:
            logger.warning("Could not clear chat history: %s", exc)
