"""Sink tests: local persistence + Kaggle publish (mock runner)."""
from __future__ import annotations

import numpy as np

from collector.config import CollectorConfig
from collector.sink import KaggleDatasetSink, LocalSink, build_sink


def _xy(n=3, dim=32):
    return np.ones((n, dim), np.float32), np.array([1.0, 0.5, 0.0], np.float32)


def test_local_sink_writes_npz(tmp_path):
    sink = LocalSink(tmp_path)
    X, y = _xy()
    path = sink.write_value_chunk("data_collected_x", X, y)
    d = np.load(path)
    assert d["X"].shape == (3, 32)
    assert list(d["y"]) == [1.0, 0.5, 0.0]


def test_local_sink_metadata_appends(tmp_path):
    sink = LocalSink(tmp_path)
    sink.write_metadata("episodes_a", [{"episode_id": "1"}])
    sink.write_metadata("episodes_a", [{"episode_id": "2"}])
    lines = (tmp_path / "meta" / "episodes_a.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_local_sink_raw_optin(tmp_path):
    sink = LocalSink(tmp_path)
    p = sink.write_raw("ep1", {"hello": "world"})
    assert p and (tmp_path / "raw" / "ep1.json").exists()


def test_zip_value_chunks(tmp_path):
    import zipfile
    sink = LocalSink(tmp_path)
    X, y = _xy()
    sink.write_value_chunk("data_collected_1", X, y)
    sink.write_value_chunk("data_collected_2", X, y)
    z = sink.zip_value_chunks()
    assert z is not None
    with zipfile.ZipFile(z) as zf:
        assert sorted(zf.namelist()) == ["data_collected_1.npz", "data_collected_2.npz"]


def test_kaggle_sink_publish_invokes_cli(tmp_path):
    calls = []

    def runner(args, timeout):
        calls.append(args)
        return 0, "uploaded", ""

    local = LocalSink(tmp_path)
    sink = KaggleDatasetSink(local, "owner/ds", runner=runner)
    X, y = _xy()
    sink.write_value_chunk("data_collected_1", X, y)
    assert sink.publish("test msg") is True
    assert calls and calls[0][:3] == ["kaggle", "datasets", "version"]
    # metadata template auto-created
    assert (tmp_path / "dataset-metadata.json").exists()


def test_kaggle_sink_publish_handles_failure(tmp_path):
    def runner(args, timeout):
        return 1, "", "403 forbidden"

    sink = KaggleDatasetSink(LocalSink(tmp_path), "owner/ds", runner=runner)
    assert sink.publish("msg") is False


def test_kaggle_sink_no_slug_noop(tmp_path):
    sink = KaggleDatasetSink(LocalSink(tmp_path), "", runner=lambda a, t: (0, "", ""))
    assert sink.publish("msg") is False


def test_build_sink_factory(tmp_path):
    cfg = CollectorConfig(data_dir=tmp_path, sink="local")
    assert isinstance(build_sink(cfg), LocalSink)
    cfg2 = CollectorConfig(data_dir=tmp_path, sink="kaggle", dataset_slug="o/d")
    assert isinstance(build_sink(cfg2), KaggleDatasetSink)
