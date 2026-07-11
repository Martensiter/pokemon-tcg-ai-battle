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
from .features import (
    extract, extract_v2, extract_l1only, extract_all,
    extract_tempo, extract_race, extract_mob,
    FEATURE_DIM, FEATURE_DIM_V2, FEATURE_DIM_L1, FEATURE_DIM_ALL,
    FEATURE_DIM_TEMPO, FEATURE_DIM_RACE, FEATURE_DIM_MOB,
)
from .evaluate import evaluate as heuristic_evaluate


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Known feature versions: the input dim of W1 selects the extractor, so v1 and
# v2 weight files are interchangeable drop-ins for the same agent code.
_EXTRACTORS = {FEATURE_DIM: extract, FEATURE_DIM_L1: extract_l1only,
               FEATURE_DIM_V2: extract_v2, FEATURE_DIM_ALL: extract_all,
               FEATURE_DIM_TEMPO: extract_tempo, FEATURE_DIM_RACE: extract_race,
               FEATURE_DIM_MOB: extract_mob}


class ValueNet:
    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]],
                 extractor=extract):
        # layers: list of (W, b); all but the last use ReLU, the last uses sigmoid
        self.layers = layers
        self._extract = extractor

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
            if not layers:
                return None
            extractor = _EXTRACTORS.get(layers[0][0].shape[0])
            if extractor is None:
                return None
            return cls(layers, extractor)
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
        p = self.forward(self._extract(state, me))
        return 2.0 * p - 1.0


def make_leaf_evaluator(cfg=C):
    """Return eval_fn(state, me) -> [-1, 1], blending heuristic + value net."""
    net = ValueNet.maybe_load(cfg.WEIGHTS_PATH)
    w = cfg.VALUE_NET_WEIGHT if net is not None else 0.0
    # L2 gate resolved from cfg so tools/sweep_config.py can vary it per agent;
    # None (attr absent) defers to eval_params.L2_W (env-overridable, default 0).
    l2_w = getattr(cfg, "L2_W", None)
    hand_w = getattr(cfg, "HAND_VALUE_W", None)
    hand_cw = getattr(cfg, "HAND_COUNT_W", None)
    hand_bw = getattr(cfg, "HAND_BASIC_W", None)
    hand_ew = getattr(cfg, "HAND_EVO_W", None)
    role_w = getattr(cfg, "ROLE_W", None)
    anti_w = getattr(cfg, "ANTI_STALL_W", None)
    bad_w = getattr(cfg, "BAD_SHAPE_W", None)

    def eval_fn(state: dict, me: int) -> float:
        h = heuristic_evaluate(state, me, l2_w=l2_w, hand_value_w=hand_w,
                               hand_count_w=hand_cw, hand_basic_w=hand_bw,
                               hand_evo_w=hand_ew, role_w=role_w,
                               anti_stall_w=anti_w, bad_shape_w=bad_w)
        if net is None:
            return h
        return (1.0 - w) * h + w * net.value(state, me)

    eval_fn.has_net = net is not None  # type: ignore[attr-defined]
    return eval_fn
