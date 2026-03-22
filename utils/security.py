"""Security utilities for file validation.

Validates uploaded files before they enter the indexing pipeline,
protecting against oversized uploads and unexpected file types.
"""

from pathlib import Path

from config.settings import settings
from core.exceptions import SecurityError

# Phase 1: PDF only. Expanded in Phase 2 (DOCX, TXT, MD).
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf"})


def validate_file(file_path: str | Path) -> Path:
    """Validate a file before ingestion.

    Args:
        file_path: Path to the uploaded file.

    Returns:
        Resolved Path object if valid.

    Raises:
        SecurityError: If the file fails any validation check.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise SecurityError(f"File not found: {path.name}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise SecurityError(
            f"Unsupported file type '{path.suffix}'. Allowed: {allowed}"
        )

    file_size = path.stat().st_size
    if file_size > settings.max_file_size_bytes:
        limit_mb = settings.max_file_size_bytes // (1024 * 1024)
        raise SecurityError(
            f"File '{path.name}' exceeds the {limit_mb} MB size limit."
        )

    if file_size == 0:
        raise SecurityError(f"File '{path.name}' is empty.")

    return path
