"""Option featurization shared by policy-data extraction and agent inference.

This lives in ``agent/`` (not the collector) because the agent SHIPS in the
Kaggle submission and must featurize options at inference time without importing
the collector package. It is engine/CSV-free (no card DB), so it works in the
sandbox and in CI.

An option becomes:
  * its ``type`` one-hot (attack / play / retreat / end / ...),
  * a few structural flags, and
  * an IDENTITY signature -- a hashed one-hot of the option's ``attackId`` and of
    the card it refers to (resolved from the observation, DB-free). Without the
    identity, two options of the SAME type (e.g. two different attacks, or two
    cards to play) were near-identical vectors and the policy could only tell
    them apart by list position -- the real accuracy ceiling. The card id is read
    straight from the player's own hand/bench/discard arrays in the state, so no
    card database is needed (the id is depended on only when the replay carries
    those arrays; otherwise that block is simply zero).
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_DIM

STATE_DIM = FEATURE_DIM
# One-hot of the option ``type`` (OptionType has ~54 members and may grow mid-
# season, so bucket generously and clamp overflow into the last bucket).
OPTION_TYPE_DIM = 64
_N_FLAGS = 5                         # has-area, has-attackId, has-index, mine, pos
# Hashed one-hot of the option IDENTITY (breaks the same-type collapse, DB-free).
ATTACK_HASH_DIM = 32
CARD_HASH_DIM = 64
OPT_FEAT_DIM = OPTION_TYPE_DIM + _N_FLAGS + ATTACK_HASH_DIM + CARD_HASH_DIM


# AreaType: 2=HAND 3=DISCARD 5=BENCH (the public/own areas whose card ids the
# acting seat can see). Mirrors agent.policy._card_id_from_option but WITHOUT
# importing it (that module pulls the engine) -- pure dict access only.
_AREA_KEY = {2: "hand", 3: "discard", 5: "bench"}


def _resolve_card_id(o: dict, state) -> int | None:
    """The card id an option refers to, read from the state. None if not locatable."""
    if not isinstance(state, dict):
        return None
    area, idx, pi = o.get("area"), o.get("index"), o.get("playerIndex")
    key = _AREA_KEY.get(area)
    if key is None or idx is None:
        return None
    try:
        player = state["players"][pi if pi is not None else state.get("yourIndex", 0)]
        arr = player.get(key) or []
    except (KeyError, IndexError, TypeError):
        return None
    if 0 <= idx < len(arr) and isinstance(arr[idx], dict):
        return arr[idx].get("id")
    return None


def _hash_onehot(value, dim: int, out: np.ndarray, base: int) -> None:
    """Set one bucket high for ``value`` (feature hashing; no vocab/DB needed)."""
    if value is None:
        return
    try:
        out[base + (abs(int(value)) % dim)] = 1.0
    except (TypeError, ValueError):
        return


def featurize_option(o: dict, idx: int, n: int, me: int, state=None) -> np.ndarray:
    """Engine/DB-free feature vector for one option (now distinguishes identity)."""
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
    # identity: which attack, and which card (so same-type options differ)
    idf = base + _N_FLAGS
    _hash_onehot(o.get("attackId"), ATTACK_HASH_DIM, v, idf)
    _hash_onehot(_resolve_card_id(o, state), CARD_HASH_DIM, v, idf + ATTACK_HASH_DIM)
    return v
