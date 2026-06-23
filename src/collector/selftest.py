"""Offline, no-credentials self-test of the full collector pipeline.

This lets a collaborator acceptance-test (UAT) the collector end-to-end without
Kaggle credentials, network, or the engine binary. It feeds a built-in synthetic
replay through the *real* :class:`~collector.collector.Collector` +
:class:`~collector.sink.LocalSink` into a temp directory, then verifies a
training chunk was produced with the exact shape the value net expects.

Run via ``python -m collector --self-test``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from agent.features import FEATURE_DIM

from .collector import Collector
from .config import CollectorConfig
from .manifest import Manifest
from .sink import LocalSink


# --- synthetic replay builders (self-contained; no test deps) -------------
def _player(hp: int = 100) -> dict[str, Any]:
    return {
        "active": [{"id": 100, "hp": hp, "maxHp": 120, "energies": [0, 1]}],
        "bench": [{"id": 101, "hp": 70, "maxHp": 70, "energies": [0]}],
        "benchMax": 5, "deckCount": 40,
        "discard": [{"id": 1}, {"id": 1}],
        "prize": [{"id": 0}] * 3, "handCount": 5, "hand": None,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }


def _state(turn: int, your_index: int, result: int = -1) -> dict[str, Any]:
    return {
        "turn": turn, "turnActionCount": 1, "yourIndex": your_index,
        "firstPlayer": 0, "supporterPlayed": False, "stadiumPlayed": False,
        "energyAttached": False, "retreated": False, "result": result,
        "stadium": [], "looking": None,
        "players": [_player(), _player(80)],
    }


def _synthetic_replay(winner: int = 0) -> dict[str, Any]:
    """Synthetic replay in the REAL Kaggle env shape (board state in each
    steps[i][seat].observation.current/select), so the self-test exercises the
    same extraction path production uses."""
    deck = list(range(1, 61))
    sel = {"context": 0, "type": 0, "minCount": 1, "maxCount": 1, "option": [{"type": 14}]}
    # deck-selection step (no live state yet)
    steps = [[
        {"action": deck, "status": "ACTIVE",
         "observation": {"current": None, "select": None, "logs": []}},
        {"action": deck, "status": "INACTIVE",
         "observation": {"current": None, "select": None, "logs": []}},
    ]]
    for i in range(4):  # 4 MAIN decisions, alternating seats
        me = i % 2
        active = {"status": "ACTIVE",
                  "observation": {"current": _state(i + 1, me), "select": sel, "logs": []}}
        inactive = {"status": "INACTIVE",
                    "observation": {"current": None, "select": None, "logs": []}}
        steps.append([active, inactive] if me == 0 else [inactive, active])
    rewards = [1, 0] if winner == 0 else ([0, 1] if winner == 1 else [0, 0])
    return {
        "info": {"Agents": [{"Name": "selftest_a"}, {"Name": "selftest_b"}]},
        "rewards": rewards,
        "steps": steps,
    }


class _OfflineClient:
    """A KaggleClient stand-in returning synthetic data (no network)."""

    def __init__(self, n_subs: int = 2, n_eps: int = 3):
        self._subs = [{"submissionId": str(900 + i), "teamName": f"Self{i}"}
                      for i in range(n_subs)]
        # numeric episode ids (Kaggle's `replay` requires an int)
        self._eps = {s["submissionId"]: [f"{s['submissionId']}{j:02d}" for j in range(n_eps)]
                     for s in self._subs}

    def leaderboard(self): return list(self._subs)
    def submissions(self): return list(self._subs)

    def episodes(self, submission_id):
        return [{"episodeId": e} for e in self._eps.get(str(submission_id), [])]

    def replay(self, episode_id):
        # winner derived from the trailing digit for a bit of label variety
        winner = int(str(episode_id)[-1]) % 2
        return {"episode_id": episode_id, "replay": _synthetic_replay(winner)}


def run_selftest(workdir: str | None = None, n_subs: int = 2, n_eps: int = 3) -> dict[str, Any]:
    """Run one fully-offline collection pass and validate the output.

    Returns a summary dict. Raises ``AssertionError`` if the produced chunk does
    not match the value-net contract.
    """
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="collector_selftest_"))
    cfg = CollectorConfig(data_dir=tmp / "data", state_dir=tmp / "state",
                          rps=0.0, chunk_size=1000, sink="local")
    client = _OfflineClient(n_subs=n_subs, n_eps=n_eps)
    col = Collector(cfg, client=client, sink=LocalSink(cfg.data_dir),
                    manifest=Manifest(cfg.state_dir / "manifest.jsonl"))
    stats = col.run_once()

    chunks = sorted((cfg.data_dir / "value").glob("data_collected_*.npz"))
    assert chunks, "self-test produced no value chunk"
    d = np.load(chunks[0])
    assert "X" in d and "y" in d, "chunk missing X/y arrays"
    assert d["X"].shape[1] == FEATURE_DIM, f"feature dim {d['X'].shape[1]} != {FEATURE_DIM}"
    assert d["X"].shape[0] == len(d["y"]) == stats.converted_rows, "row count mismatch"
    assert set(np.unique(d["y"])).issubset({0.0, 0.5, 1.0}), "labels out of range"

    expected_rows = n_subs * n_eps * 4  # 4 MAIN frames per synthetic episode
    assert stats.converted_rows == expected_rows, \
        f"expected {expected_rows} rows, got {stats.converted_rows}"

    return {
        "workdir": str(tmp),
        "submissions": stats.submissions,
        "episodes": stats.episodes_listed,
        "rows": stats.converted_rows,
        "feature_dim": int(d["X"].shape[1]),
        "chunk": str(chunks[0]),
        "mean_label": round(float(d["y"].mean()), 4),
    }
