"""Daily pipeline: merge collector chunks -> numpy retrain -> candidate weights.

Pure numpy, no engine/torch/network. Uses the offline self-test to produce real
collector chunks, then runs the pipeline and checks the candidate weights match
the value-net layout.
"""
from __future__ import annotations

import json

import numpy as np

from agent.features import FEATURE_DIM
from collector.selftest import run_selftest
from tools.daily_pipeline import _stage_state, run_pipeline


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


def test_stage_state_copies_manifest_into_upload_dir(tmp_path):
    """Disaster-recovery: the manifest is staged under data/state/ so it rides in
    the published Kaggle Dataset version (restore -> no duplicate-collection)."""
    import logging
    run_selftest(workdir=tmp_path, n_subs=2, n_eps=2)   # writes state/manifest.jsonl
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    assert (state_dir / "manifest.jsonl").exists()

    staged = _stage_state(data_dir, state_dir, logging.getLogger("t"))
    assert staged is True
    copied = data_dir / "state" / "manifest.jsonl"
    assert copied.exists()
    # content matches the source manifest exactly
    assert copied.read_text() == (state_dir / "manifest.jsonl").read_text()


def test_stage_state_noop_without_state_dir(tmp_path):
    import logging
    run_selftest(workdir=tmp_path, n_subs=1, n_eps=1)
    assert _stage_state(tmp_path / "data", None, logging.getLogger("t")) is False
