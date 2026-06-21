"""Conversion tests: output is byte-compatible with the existing training data.

These assert the collector emits exactly what ``selfplay/train_value.py`` reads:
``X`` of shape ``(N, FEATURE_DIM)`` float32 and ``y`` in {0.0, 0.5, 1.0}, with the
*same* labelling convention as ``selfplay/gen_data.py``.
"""
from __future__ import annotations

import numpy as np
import conftest as cf

from collector.convert import (
    ValueRecords, episode_to_records, episode_metadata, label_for,
)
from collector.parse import parse_episode
from agent.features import FEATURE_DIM, extract


def test_label_convention_matches_gen_data():
    # gen_data.py: 0.5 draw, 1.0 if winner==player, else 0.0
    assert label_for(2, 0) == 0.5
    assert label_for(0, 0) == 1.0
    assert label_for(0, 1) == 0.0
    assert label_for(1, 1) == 1.0


def test_records_shape_and_dtype():
    ep = parse_episode(cf.make_episode_steps(winner=0))
    rec = ValueRecords()
    added = episode_to_records(ep, rec)
    assert added == 4
    X, y = rec.arrays()
    assert X.shape == (4, FEATURE_DIM)
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    assert set(np.unique(y)).issubset({0.0, 0.5, 1.0})


def test_labels_track_winner():
    ep = parse_episode(cf.make_episode_steps(winner=0))
    rec = ValueRecords()
    episode_to_records(ep, rec)
    _, y = rec.arrays()
    # Frames alternate seats 0,1,0,1 -> seat0 won -> labels 1,0,1,0
    assert list(y) == [1.0, 0.0, 1.0, 0.0]


def test_features_match_direct_extract():
    """A collected MAIN state yields the identical vector agent.features would."""
    ep = parse_episode(cf.make_episode_steps(winner=0))
    rec = ValueRecords()
    episode_to_records(ep, rec)
    X, _ = rec.arrays()
    first_state = ep.frames[0]["current"]
    direct = extract(first_state, first_state["yourIndex"])
    assert np.allclose(X[0], direct)


def test_unknown_winner_skipped_for_value():
    ep = parse_episode(cf.make_episode_steps(winner=0))
    ep.winner = -1  # force unknown
    rec = ValueRecords()
    assert episode_to_records(ep, rec) == 0
    assert len(rec) == 0


def test_empty_records_arrays():
    rec = ValueRecords()
    X, y = rec.arrays()
    assert X.shape == (0, FEATURE_DIM)
    assert y.shape == (0,)


def test_metadata_has_no_raw_board():
    ep = parse_episode(cf.make_episode_steps(winner=0))
    meta = episode_metadata(ep)
    assert meta["agents"] == ["alice", "bob"]
    assert meta["winner"] == 0
    assert "0" in meta["decks"]
    assert "frames" not in meta  # no heavy board data leaks into metadata


def test_chunk_roundtrips_through_numpy_load(tmp_path):
    """Saved chunk reloads with X/y keys exactly like merge_data.py expects."""
    ep = parse_episode(cf.make_episode_steps(winner=0))
    rec = ValueRecords()
    episode_to_records(ep, rec)
    X, y = rec.arrays()
    path = tmp_path / "data_collected_test.npz"
    np.savez_compressed(path, X=X, y=y)
    d = np.load(path)
    assert "X" in d and "y" in d
    assert d["X"].shape == (4, FEATURE_DIM)
    assert len(d["y"]) == 4
