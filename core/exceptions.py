"""Polaris custom exception hierarchy.

All domain errors inherit from PolarisError so callers can catch
the base class when they don't care about the specific failure type.
"""


class PolarisError(Exception):
    """Base exception for all Polaris errors."""


class DocumentLoadError(PolarisError):
    """Raised when a document cannot be loaded or parsed."""


class IndexingError(PolarisError):
    """Raised when document indexing (chunking / embedding) fails."""


class QueryError(PolarisError):
    """Raised when a user query cannot be answered."""


class SecurityError(PolarisError):
    """Raised when a file fails security validation."""
