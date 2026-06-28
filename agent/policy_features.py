"""Option featurization shared by policy-data extraction and agent inference.

This lives in ``agent/`` (not the collector) because the agent SHIPS in the
Kaggle submission and must featurize options at inference time without importing
the collector package. Engine/CSV-free: an option becomes its ``type`` one-hot
plus a few structural flags, so it works in the sandbox and in CI.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_DIM

STATE_DIM = FEATURE_DIM
# One-hot of the option ``type`` (OptionType has ~54 members and may grow mid-
# season, so bucket generously and clamp overflow into the last bucket).
OPTION_TYPE_DIM = 64
# + flags: has-area, has-attackId, has-index, playerIndex==me, normalized position
OPT_FEAT_DIM = OPTION_TYPE_DIM + 5


def featurize_option(o: dict, idx: int, n: int, me: int) -> np.ndarray:
    """Engine/DB-free feature vector for one option (varies across a group)."""
    v = np.zeros(OPT_FEAT_DIM, dtype=np.float32)
    if not isinstance(o, dict):
        o = {}
    t = o.get("type")
    if isinstance(t, int):
        v[min(max(t, 0), OPTION_TYPE_DIM - 1)] = 1.0
    base = OPTION_TYPE_DIM
    v[base + 0] = 1.0 if o.get("area") is not None else 0.0
    v[base + 1] = 1.0 if o.get("attackId") is not None else 0.0
    v[base + 2] = 1.0 if o.get("index") is not None else 0.0
    pi = o.get("playerIndex")
    v[base + 3] = 1.0 if (pi is not None and pi == me) else 0.0
    v[base + 4] = (idx / n) if n else 0.0
    return v
