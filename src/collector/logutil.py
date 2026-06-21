"""Structured logging for the collector.

Logs go to *both* stdout (so ``nohup`` capture works) and a rotating file. We
emit single-line ``key=value`` records so counts (fetched / skipped / backoff)
are greppable without pulling in a JSON-logging dependency.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class _KVFormatter(logging.Formatter):
    """Render ``logger.info("event", extra={"kv": {...}})`` as ``k=v`` pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} {record.levelname:<5} {record.getMessage()}"
        kv = getattr(record, "kv", None)
        if kv:
            pairs = " ".join(f"{k}={_fmt(v)}" for k, v in kv.items())
            base = f"{base} {pairs}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def _fmt(v: Any) -> str:
    s = str(v)
    return f'"{s}"' if " " in s else s


def get_logger(name: str = "collector", log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stdout and (optionally) a file."""
    logger = logging.getLogger(name)
    if getattr(logger, "_collector_configured", False):
        return logger
    logger.setLevel(level)
    fmt = _KVFormatter()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    logger._collector_configured = True  # type: ignore[attr-defined]
    return logger


def log_kv(logger: logging.Logger, msg: str, level: int = logging.INFO, **kv: Any) -> None:
    """Convenience wrapper for ``logger.log(level, msg, extra={'kv': kv})``."""
    logger.log(level, msg, extra={"kv": kv})
