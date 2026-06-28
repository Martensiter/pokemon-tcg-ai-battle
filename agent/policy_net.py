"""Policy network (behavioral-cloning) — numpy inference for the agent.

Loaded from ``policy.npz`` (trained by ``selfplay/train_policy_np.py``). Given a
decision's state and its options, it scores each option and returns a normalized
prior over them. The MCTS uses this prior PUCT-style to bias exploration toward
the moves strong agents make. Like the value net, it is OPTIONAL: if the file is
absent (or disabled via config) the agent runs exactly as before.

Inference is numpy-only and engine/CSV-free (option features come from
``agent.policy_features``), so it ships in the submission and is testable in CI.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .features import extract
from .policy_features import OPT_FEAT_DIM, STATE_DIM, featurize_option


def _relu(x):
    return np.maximum(x, 0.0)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    return e / s if s else np.full_like(e, 1.0 / len(e))


class PolicyNet:
    """Option-scoring MLP: input [state | option] -> scalar score; softmax = prior."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = layers  # (W, b); ReLU between layers, linear at the top

    @classmethod
    def maybe_load(cls, path: str) -> "PolicyNet | None":
        if not path or not os.path.exists(path):
            return None
        try:
            data = np.load(path)
            layers = []
            i = 1
            while f"W{i}" in data:
                layers.append((data[f"W{i}"].astype(np.float32),
                               data[f"b{i}"].astype(np.float32)))
                i += 1
            if not layers or layers[0][0].shape[0] != STATE_DIM + OPT_FEAT_DIM:
                return None
            return cls(layers)
        except Exception:
            return None

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Per-row scalar scores for input rows ``X (M, STATE_DIM+OPT_FEAT_DIM)``."""
        h = X
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if k < len(self.layers) - 1:
                h = _relu(h)
        return h.reshape(-1)

    def priors(self, state: dict, me: int, options: list) -> np.ndarray | None:
        """Normalized prior over ``options`` (sums to 1), or None if not applicable."""
        if not options:
            return None
        try:
            feat = extract(state, me)
            if feat is None or feat.shape[0] != STATE_DIM:
                return None
            n = len(options)
            opt = np.stack([featurize_option(o, i, n, me, state) for i, o in enumerate(options)])
            state_rep = np.repeat(feat.reshape(1, -1), n, axis=0)
            X = np.concatenate([state_rep, opt], axis=1).astype(np.float32)
            return _softmax(self.scores(X))
        except Exception:
            return None


def puct_select(visits, values, priors, c: float) -> int:
    """PUCT arm choice: argmax_i  Q_i + c * P_i * sqrt(1+total) / (1 + visits_i).

    Q_i is the mean value of arm i (0 when unvisited). With c <= 0 this is unused
    (the caller stays on plain UCB1). Pure/numpy-free so it is unit-testable.
    """
    n = len(visits)
    total = sum(visits)
    norm = math.sqrt(1.0 + total)
    best_s, best_i = -1e18, 0
    for i in range(n):
        q = (values[i] / visits[i]) if visits[i] > 0 else 0.0
        p = priors[i] if priors is not None and i < len(priors) else (1.0 / n)
        u = c * p * norm / (1.0 + visits[i])
        s = q + u
        if s > best_s:
            best_s, best_i = s, i
    return best_i
