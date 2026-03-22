"""Structured logging configuration for Polaris.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Document indexed", extra={"document_id": doc_id, "chunk_count": 42})
"""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the given module name.

    Log level is controlled by the LOG_LEVEL environment variable
    (default: INFO). Outputs structured plain-text lines to stdout.
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    return logging.getLogger(name)
