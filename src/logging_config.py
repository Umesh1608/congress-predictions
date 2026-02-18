"""Structured JSON logging configuration with correlation IDs."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter as _JsonFormatter

# Correlation ID propagated through async context
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class CorrelationFilter(logging.Filter):
    """Inject correlation_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get("")  # type: ignore[attr-defined]
        return True


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return uuid.uuid4().hex[:16]


def setup_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging for the application."""
    handler = logging.StreamHandler()
    formatter = _JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(message)s %(correlation_id)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addFilter(CorrelationFilter())

    # Quiet noisy third-party loggers
    for name in ("uvicorn.access", "sqlalchemy.engine", "neo4j"):
        logging.getLogger(name).setLevel(logging.WARNING)
