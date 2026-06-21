"""Convert parsed episodes into records the existing training loop consumes.

The value net (``selfplay/train_value.py``) trains on ``data*.npz`` files with two
arrays:
  * ``X``: float32 ``(N, FEATURE_DIM)`` from ``agent.features.extract(state, me)``
  * ``y``: float32 ``(N,)`` label = 1.0 win / 0.5 draw / 0.0 loss for the
    to-move player ``me`` at that state.

``selfplay/gen_data.py`` builds exactly this from live self-play at MAIN
decisions. We reproduce that labelling from *downloaded* replays so real ladder
games flow into the same pipeline. The only heavy import is ``numpy``; we reuse
``agent.features`` directly (numpy-only, no torch, no engine binary), so this runs
on the ARM collector device.

A thin adapter is unnecessary -- output is byte-compatible with what
``selfplay/merge_data.py`` and ``train_value.py`` already read.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

# Make the repo-root ``agent`` package importable from the collector package.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from agent.features import extract, FEATURE_DIM  # noqa: E402  (numpy-only)

from .parse import ParsedEpisode, iter_main_decisions  # noqa: E402


def label_for(winner: int, me: int) -> float:
    """1.0 if ``me`` won, 0.5 draw, 0.0 loss -- matches gen_data.py."""
    if winner == 2:
        return 0.5
    return 1.0 if winner == me else 0.0


@dataclass
class ValueRecords:
    """Accumulator of ``(X, y)`` rows, flushable to an npz-compatible payload."""

    X: list[np.ndarray] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def add(self, feat: np.ndarray, label: float) -> None:
        self.X.append(feat)
        self.y.append(label)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` as the exact dtypes/shapes train_value.py expects."""
        if not self.y:
            return (np.zeros((0, FEATURE_DIM), np.float32),
                    np.zeros((0,), np.float32))
        X = np.stack(self.X).astype(np.float32)
        y = np.asarray(self.y, dtype=np.float32)
        return X, y


def episode_to_records(ep: ParsedEpisode, records: ValueRecords) -> int:
    """Append value-net rows from one episode. Returns rows added.

    Episodes whose winner is unknown are skipped for *value* records (we cannot
    label them) but still produce metadata via :func:`episode_metadata`.
    """
    if ep.winner == -1:
        return 0
    added = 0
    for state, me in iter_main_decisions(ep):
        try:
            feat = extract(state, me)
        except Exception:  # noqa: BLE001  (defensive: skip malformed state)
            continue
        if feat is None or getattr(feat, "shape", (0,))[0] != FEATURE_DIM:
            continue
        records.add(feat, label_for(ep.winner, me))
        added += 1
    return added


def episode_metadata(ep: ParsedEpisode) -> dict[str, Any]:
    """Small, repo-safe derived summary (no raw board data) for indexing."""
    return {
        "episode_id": ep.episode_id,
        "agents": ep.agents,
        "rewards": ep.rewards,
        "winner": ep.winner,
        "reason": ep.reason,
        "turns": ep.turns,
        "n_frames": ep.n_frames,
        "decks": {str(k): v for k, v in ep.decks.items()},
        "ok": ep.ok,
    }
