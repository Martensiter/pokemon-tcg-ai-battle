"""Production replay collector for the Pokemon TCG AI Battle Challenge.

This package turns the one-shot ``tools/`` episode scanners into a long-running,
idempotent, resumable collector that mines real Kaggle ladder replays and emits
records the *existing* value-net training pipeline (``selfplay/train_value.py``)
can consume unchanged.

Design split (see ``AGENTS.md`` / ``CLAUDE.md``):
  * collection runs on a small ARM device, 24/7, numpy-only, no torch;
  * training stays offline on a separate machine.

Nothing here imports torch or the engine binary, so it runs on aarch64 with only
``numpy`` available.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
