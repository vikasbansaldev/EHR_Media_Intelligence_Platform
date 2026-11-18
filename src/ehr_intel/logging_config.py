"""Logging setup shared by the CLI and the API server."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once; safe to call repeatedly."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S")
    else:
        root.setLevel(level)
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
