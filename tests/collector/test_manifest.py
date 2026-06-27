"""Manifest idempotency + crash-resume tests."""
from __future__ import annotations

from collector.manifest import Manifest


def test_record_and_has(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    assert not m.has("e1")
    m.record("e1", "converted", rows=10)
    assert m.has("e1")
    assert m.counts()["converted"] == 1
    assert m.seen_count == 1


def test_idempotent_rerecord(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    m.record("e1", "converted", rows=10)
    m.record("e1", "failed", rows=0)  # ignored: already seen
    assert m.counts().get("failed") is None
    assert m.counts()["converted"] == 1


def test_resume_reloads_seen(tmp_path):
    path = tmp_path / "manifest.jsonl"
    m1 = Manifest(path)
    m1.record("e1", "converted", rows=5)
    m1.record("e2", "empty")
    # simulate restart
    m2 = Manifest(path)
    assert m2.has("e1") and m2.has("e2")
    assert m2.seen_count == 2
    assert m2.counts()["converted"] == 1
    assert m2.counts()["empty"] == 1


def test_tolerates_torn_final_line(tmp_path):
    path = tmp_path / "manifest.jsonl"
    m = Manifest(path)
    m.record("e1", "converted")
    # append a half-written line as a kill -9 might leave
    with path.open("a", encoding="utf-8") as f:
        f.write('{"episode_id": "e2", "stat')
    m2 = Manifest(path)
    assert m2.has("e1")
    assert not m2.has("e2")  # torn line dropped


def test_summary_written(tmp_path):
    import json
    m = Manifest(tmp_path / "manifest.jsonl")
    m.record("e1", "converted", rows=7)
    out = tmp_path / "status.json"
    m.write_summary(out, extra={"foo": "bar"})
    data = json.loads(out.read_text())
    assert data["seen"] == 1
    assert data["counts"]["converted"] == 1
    assert data["foo"] == "bar"


def test_summary_stamps_timestamp(tmp_path):
    """status.json carries a unix `ts` heartbeat for external staleness checks."""
    import json
    import time
    m = Manifest(tmp_path / "manifest.jsonl")
    before = int(time.time())
    out = tmp_path / "status.json"
    m.write_summary(out)
    data = json.loads(out.read_text())
    assert isinstance(data["ts"], int)
    assert data["ts"] >= before
