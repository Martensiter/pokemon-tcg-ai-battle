"""Value network: trained offline with torch, executed here in pure numpy.

Keeping inference dependency-free (numpy only) matters because the Kaggle
submission runs sandboxed with no network and a minimal environment. Weights are
exported by selfplay/train_value.py to weights.npz as W1/b1/.../Wk/bk. If the
file is missing the agent transparently falls back to the heuristic evaluation,
so it is always runnable.
"""
from __future__ import annotations

import os
import numpy as np

from . import config as C
from .features import extract, FEATURE_DIM
from .evaluate import evaluate as heuristic_evaluate


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ValueNet:
    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        # layers: list of (W, b); all but the last use ReLU, the last uses sigmoid
        self.layers = layers

    @classmethod
    def maybe_load(cls, path: str) -> "ValueNet | None":
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path)
            layers = []
            i = 1
            while f"W{i}" in data:
                layers.append((data[f"W{i}"].astype(np.float32),
                               data[f"b{i}"].astype(np.float32)))
                i += 1
            if not layers or layers[0][0].shape[0] != FEATURE_DIM:
                return None
            return cls(layers)
        except Exception:
            return None

    def forward(self, x: np.ndarray) -> float:
        h = x
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            h = _sigmoid(h) if k == len(self.layers) - 1 else _relu(h)
        return float(h.reshape(-1)[0])

    def value(self, state: dict, me: int) -> float:
        """Win-probability mapped to [-1, 1] for player `me`."""
        if state is None:
            return 0.0
        p = self.forward(extract(state, me))
        return 2.0 * p - 1.0


def make_leaf_evaluator(cfg=C):
    """Return eval_fn(state, me) -> [-1, 1], blending heuristic + value net."""
    net = ValueNet.maybe_load(cfg.WEIGHTS_PATH)
    w = cfg.VALUE_NET_WEIGHT if net is not None else 0.0
    # L2 gate resolved from cfg so tools/sweep_config.py can vary it per agent;
    # None (attr absent) defers to eval_params.L2_W (env-overridable, default 0).
    l2_w = getattr(cfg, "L2_W", None)

    def eval_fn(state: dict, me: int) -> float:
        h = heuristic_evaluate(state, me, l2_w=l2_w)
        if net is None:
            return h
        return (1.0 - w) * h + w * net.value(state, me)

    eval_fn.has_net = net is not None  # type: ignore[attr-defined]
    return eval_fn
