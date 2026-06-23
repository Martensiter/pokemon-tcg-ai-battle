"""Daily pipeline: merge collector chunks -> numpy retrain -> candidate weights.

Pure numpy, no engine/torch/network. Uses the offline self-test to produce real
collector chunks, then runs the pipeline and checks the candidate weights match
the value-net layout.
"""
from __future__ import annotations

import json

import numpy as np

from collector.selftest import run_selftest
from tools.daily_pipeline import run_pipeline

FEATURE_DIM = 32


def test_pipeline_trains_and_writes_candidate(tmp_path):
    # produce real chunks via the offline collector self-test
    summary = run_selftest(workdir=tmp_path, n_subs=2, n_eps=3)
    data_dir = tmp_path / "data"

    out = run_pipeline(data_dir, hidden=[64, 64], epochs=5, min_rows=1,
                       publish=False, dataset_slug="")
    assert out["trained"] is True
    assert out["rows"] == summary["rows"]
    assert out["published"] is False

    wpath = data_dir / "weights" / "weights_candidate.npz"
    d = np.load(wpath)
    assert d["W1"].shape[0] == FEATURE_DIM
    assert [k for k in d.files] == ["W1", "b1", "W2", "b2", "W3", "b3"]

    report = json.loads((data_dir / "weights" / "train_report.json").read_text())
    assert report["rows"] == summary["rows"] and "val_acc" in report


def test_pipeline_skips_when_too_few_rows(tmp_path):
    run_selftest(workdir=tmp_path, n_subs=1, n_eps=1)   # ~4 rows
    out = run_pipeline(tmp_path / "data", hidden=[64, 64], epochs=3,
                       min_rows=10_000, publish=False, dataset_slug="")
    assert out["trained"] is False
    assert not (tmp_path / "data" / "weights").exists()


def test_pipeline_publish_noop_without_slug(tmp_path):
    run_selftest(workdir=tmp_path, n_subs=2, n_eps=2)
    out = run_pipeline(tmp_path / "data", hidden=[64, 64], epochs=3, min_rows=1,
                       publish=True, dataset_slug="")    # publish requested, no slug
    assert out["trained"] is True and out["published"] is False
