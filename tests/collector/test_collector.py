"""End-to-end collector pass with a mock Kaggle client + LocalSink."""
from __future__ import annotations

import numpy as np
import conftest as cf

from collector.collector import Collector
from collector.config import CollectorConfig
from collector.manifest import Manifest
from collector.sink import LocalSink


class FakeClient:
    """Stand-in for KaggleClient: deterministic, no network."""

    def __init__(self, n_subs=2, n_eps=3, winner=0):
        self._subs = [{"submissionId": str(100 + i), "teamName": f"Team{i}"}
                      for i in range(n_subs)]
        self._eps = {s["submissionId"]: [f"{s['submissionId']}-e{j}" for j in range(n_eps)]
                     for s in self._subs}
        self.winner = winner
        self.replay_calls = 0

    def leaderboard(self):
        return list(self._subs)

    def submissions(self):
        return list(self._subs)

    def episodes(self, submission_id):
        return [{"episodeId": e} for e in self._eps.get(str(submission_id), [])]

    def replay(self, episode_id):
        self.replay_calls += 1
        return {"episode_id": episode_id,
                "replay": cf.make_episode_steps(winner=self.winner)}


def _cfg(tmp_path, **over):
    base = dict(data_dir=tmp_path / "data", state_dir=tmp_path / "state",
                rps=0.0, chunk_size=100, sink="local")
    base.update(over)
    return CollectorConfig(**base)


def test_run_once_produces_loadable_chunk(tmp_path):
    cfg = _cfg(tmp_path)
    client = FakeClient(n_subs=2, n_eps=3, winner=0)
    sink = LocalSink(cfg.data_dir)
    manifest = Manifest(cfg.state_dir / "manifest.jsonl")
    col = Collector(cfg, client=client, sink=sink, manifest=manifest)

    stats = col.run_once()
    assert stats.submissions == 2
    assert stats.episodes_listed == 6
    assert client.replay_calls == 6
    # 6 episodes * 4 MAIN frames each = 24 rows
    assert stats.converted_rows == 24

    chunks = list((cfg.data_dir / "value").glob("data_collected_*.npz"))
    assert len(chunks) == 1
    d = np.load(chunks[0])
    assert d["X"].shape[0] == 24
    assert d["X"].shape[1] == 32
    assert set(np.unique(d["y"])).issubset({0.0, 0.5, 1.0})


def test_idempotent_second_pass_skips(tmp_path):
    cfg = _cfg(tmp_path)
    client = FakeClient()
    sink = LocalSink(cfg.data_dir)
    manifest = Manifest(cfg.state_dir / "manifest.jsonl")
    col = Collector(cfg, client=client, sink=sink, manifest=manifest)

    col.run_once()
    first_calls = client.replay_calls
    # Second pass: everything already in manifest -> no new replay fetches.
    stats2 = col.run_once()
    assert client.replay_calls == first_calls
    assert stats2.skipped == 6
    assert stats2.converted_rows == 0


def test_resume_after_restart(tmp_path):
    cfg = _cfg(tmp_path)
    client = FakeClient()
    # First collector instance does a pass.
    col1 = Collector(cfg, client=client, sink=LocalSink(cfg.data_dir),
                     manifest=Manifest(cfg.state_dir / "manifest.jsonl"))
    col1.run_once()
    calls_after_first = client.replay_calls
    # Fresh instance (simulated restart) reloads manifest and skips all.
    col2 = Collector(cfg, client=FakeClient(), sink=LocalSink(cfg.data_dir),
                     manifest=Manifest(cfg.state_dir / "manifest.jsonl"))
    stats = col2.run_once()
    assert stats.skipped == 6
    assert stats.converted_rows == 0
    assert calls_after_first == 6


def test_metadata_written(tmp_path):
    cfg = _cfg(tmp_path)
    col = Collector(cfg, client=FakeClient(), sink=LocalSink(cfg.data_dir),
                    manifest=Manifest(cfg.state_dir / "manifest.jsonl"))
    col.run_once()
    metas = list((cfg.data_dir / "meta").glob("episodes_*.jsonl"))
    assert metas
    lines = metas[0].read_text().strip().splitlines()
    assert len(lines) == 6


def test_status_file_written(tmp_path):
    import json
    cfg = _cfg(tmp_path)
    col = Collector(cfg, client=FakeClient(), sink=LocalSink(cfg.data_dir),
                    manifest=Manifest(cfg.state_dir / "manifest.jsonl"))
    col.run_once()
    status = json.loads((cfg.state_dir / "status.json").read_text())
    assert status["seen"] == 6
    assert "last_pass" in status


def test_keep_raw_optin(tmp_path):
    cfg = _cfg(tmp_path, keep_raw=True)
    col = Collector(cfg, client=FakeClient(n_subs=1, n_eps=1), sink=LocalSink(cfg.data_dir),
                    manifest=Manifest(cfg.state_dir / "manifest.jsonl"))
    col.run_once()
    raws = list((cfg.data_dir / "raw").glob("*.json"))
    assert len(raws) == 1
