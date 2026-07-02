"""The torch-free numpy trainer must learn and export weights.npz in the exact
layout agent/value_net.py reads (W{i} (in,out) + b{i}, ReLU stack + sigmoid)."""
from __future__ import annotations

import numpy as np

from selfplay.train_value_np import train_mlp, _sigmoid

FEATURE_DIM = 32


def _separable_data(n=2000, d=FEATURE_DIM, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d)
    y = (X @ w + 0.1 * rng.standard_normal(n) > 0).astype(np.float32)
    return X, y


def _forward(weights, X):
    """Replicate agent/value_net.ValueNet.forward (ReLU between, sigmoid last)."""
    L = sum(1 for k in weights if k.startswith("W"))
    h = X
    for i in range(1, L + 1):
        h = h @ weights[f"W{i}"] + weights[f"b{i}"]
        h = _sigmoid(h) if i == L else np.maximum(h, 0.0)
    return h


def test_export_layout_matches_value_net():
    X, y = _separable_data(n=500)
    weights, _ = train_mlp(X, y, hidden=[64, 64], epochs=5, verbose=False)
    assert [k for k in weights] == ["W1", "b1", "W2", "b2", "W3", "b3"]
    assert weights["W1"].shape == (FEATURE_DIM, 64)
    assert weights["W2"].shape == (64, 64)
    assert weights["W3"].shape == (64, 1)
    assert weights["b3"].shape == (1,)
    for v in weights.values():
        assert v.dtype == np.float32


def test_learns_separable_problem():
    X, y = _separable_data(n=3000)
    weights, metrics = train_mlp(X, y, hidden=[64, 64], epochs=40, verbose=False)
    assert metrics["val_acc"] > 0.85          # clearly learned
    probs = _forward(weights, X[:50])
    assert probs.min() >= 0.0 and probs.max() <= 1.0   # valid sigmoid outputs


def test_roundtrips_through_npz(tmp_path):
    X, y = _separable_data(n=400)
    weights, _ = train_mlp(X, y, hidden=[64, 64], epochs=3, verbose=False)
    p = tmp_path / "weights.npz"
    np.savez(p, **weights)
    d = np.load(p)
    # mimic value_net.maybe_load's contract check
    assert d["W1"].shape[0] == FEATURE_DIM
    i = 1
    while f"W{i}" in d:
        assert f"b{i}" in d
        i += 1
    assert i - 1 == 3


def test_early_stopping_stops_and_exports_best():
    X, y = _separable_data(n=1500)
    weights, metrics = train_mlp(X, y, hidden=[64, 64], epochs=300, patience=3,
                                 verbose=False)
    assert metrics["epochs_run"] < 300                 # actually stopped early
    assert 1 <= metrics["best_epoch"] <= metrics["epochs_run"]
    assert metrics["val_acc"] > 0.85                   # still learned the task
    # export layout unchanged by the rollback to the best epoch
    assert [k for k in weights] == ["W1", "b1", "W2", "b2", "W3", "b3"]
    for v in weights.values():
        assert v.dtype == np.float32


def test_patience_zero_keeps_fixed_epochs():
    X, y = _separable_data(n=500)
    _, metrics = train_mlp(X, y, hidden=[64, 64], epochs=7, patience=0, verbose=False)
    assert metrics["epochs_run"] == 7
    assert metrics["best_epoch"] == 7                  # final weights exported
