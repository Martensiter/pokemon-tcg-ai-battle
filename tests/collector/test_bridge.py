"""Bridge test: collector chunks merge into one train_value-ready dataset.

End-to-end proof of the "collected data feeds existing training" claim, using
only numpy (no engine binary, no torch): run the offline self-test to produce a
real collector chunk, then merge chunks with tools/merge_collected.py and verify
the output matches the value-net data contract (X=(N, FEATURE_DIM) float32, y in
{0,0.5,1}).
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.merge_collected import find_chunks, merge  # noqa: E402
from agent.features import FEATURE_DIM  # noqa: E402


def _write_chunk(path, n, dim=FEATURE_DIM, label=1.0):
    X = np.ones((n, dim), np.float32)
    y = np.full((n,), label, np.float32)
    np.savez_compressed(path, X=X, y=y)


def test_find_chunks_dir_glob_file(tmp_path):
    (tmp_path / "value").mkdir()
    _write_chunk(tmp_path / "value" / "data_collected_1.npz", 2)
    _write_chunk(tmp_path / "value" / "data_collected_2.npz", 3)
    _write_chunk(tmp_path / "other.npz", 1)
    # directory + pattern
    found = find_chunks([str(tmp_path / "value")], "data_collected_*.npz")
    assert len(found) == 2
    # explicit file
    found2 = find_chunks([str(tmp_path / "other.npz")], "data_collected_*.npz")
    assert len(found2) == 1


def test_merge_concatenates(tmp_path):
    _write_chunk(tmp_path / "data_collected_a.npz", 4, label=1.0)
    _write_chunk(tmp_path / "data_collected_b.npz", 6, label=0.0)
    files = find_chunks([str(tmp_path)], "data_collected_*.npz")
    X, y = merge(files, verbose=False)
    assert X.shape == (10, FEATURE_DIM)
    assert y.shape == (10,)
    assert X.dtype == np.float32 and y.dtype == np.float32
    assert y.sum() == 4.0  # four 1.0 labels


def test_merge_skips_empty_and_mismatched(tmp_path):
    _write_chunk(tmp_path / "data_collected_good.npz", 5)
    np.savez_compressed(tmp_path / "data_collected_empty.npz",
                        X=np.zeros((0, FEATURE_DIM), np.float32), y=np.zeros((0,), np.float32))
    np.savez_compressed(tmp_path / "data_collected_baddim.npz",
                        X=np.ones((3, FEATURE_DIM + 1), np.float32), y=np.ones((3,), np.float32))
    files = find_chunks([str(tmp_path)], "data_collected_*.npz")
    X, y = merge(files, verbose=False)
    assert X.shape == (5, FEATURE_DIM)  # only the good chunk survives


def test_full_collector_to_merge_pipeline(tmp_path):
    """selftest -> real chunk -> merge -> train-ready npz."""
    from collector.selftest import run_selftest
    summary = run_selftest(workdir=tmp_path, n_subs=2, n_eps=2)
    value_dir = os.path.join(summary["workdir"], "data", "value")
    files = find_chunks([value_dir], "data_collected_*.npz")
    assert files, "self-test wrote no chunk"
    X, y = merge(files, verbose=False)
    assert X.shape[1] == FEATURE_DIM
    assert X.shape[0] == len(y) == summary["rows"]
    assert set(np.unique(y)).issubset({0.0, 0.5, 1.0})

    out = tmp_path / "data_collected_all.npz"
    np.savez_compressed(out, X=X, y=y)
    # reload exactly as selfplay/train_value.py does: d["X"], d["y"]
    d = np.load(out)
    assert d["X"].shape == X.shape and len(d["y"]) == len(y)
