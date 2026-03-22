"""Backwards-compatible config exports.

Existing code that does ``from config import CHUNK_SIZE`` continues to work
while new code should import directly from ``config.settings``.
"""

from config.settings import settings

MODEL_ID = settings.model_id
EMBEDDING_MODEL_NAME = settings.embedding_model_name
MAX_LENGTH = settings.max_length
REPETITION_PENALTY = settings.repetition_penalty
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
SEARCH_K = settings.search_k
