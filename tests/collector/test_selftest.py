"""The offline self-test (UAT helper) must pass with no network/creds/engine."""
from __future__ import annotations

import numpy as np

from collector.selftest import run_selftest


def test_selftest_runs_and_validates(tmp_path):
    summary = run_selftest(workdir=tmp_path, n_subs=2, n_eps=3)
    assert summary["rows"] == 2 * 3 * 4
    assert summary["feature_dim"] == 32
    assert 0.0 <= summary["mean_label"] <= 1.0
    d = np.load(summary["chunk"])
    assert d["X"].shape == (24, 32)


def test_selftest_single(tmp_path):
    summary = run_selftest(workdir=tmp_path, n_subs=1, n_eps=1)
    assert summary["rows"] == 4
