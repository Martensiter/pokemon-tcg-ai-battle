"""Behavioral-cloning data: (state, options, chosen) from top-agent replays.

For *top-agent distillation* we learn a POLICY -- which option a strong agent
picks -- not a value. The collector already pulls leaderboard-top submissions'
games and keeps the raw replays, so the expert data is already flowing; this
turns those raw replays into policy-training records.

From each MAIN single-select decision we emit:
  * the to-move state's features (``agent.features.extract``, 32-dim, engine-free),
  * a per-option feature vector (engine-free: option ``type`` one-hot + a few
    flags -- NO card DB, so it runs on the collector device and in CI), and
  * which option index was chosen.

By default only the WINNER's decisions are kept (learn from winning play); the
collector already biases toward top agents, so winners in those games are strong.
Filtering to a *specific* top agent by name is a later refinement.

Records pack into an npz the policy trainer reads:
  ``state (G, 32)`` | ``opt (M, OPT_FEAT_DIM)`` | ``group (G,)`` sizes (sum=M) |
  ``chosen (G,)`` index-within-group.

Richer, card-aware option features (attack damage, card type) need the card DB +
engine and are intentionally out of scope for this engine-free foundation.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from agent.features import extract  # noqa: E402  (numpy-only, no engine)
# Featurization lives in agent/ so it ships with the submission; re-exported here
# (STATE_DIM/OPT_FEAT_DIM/OPTION_TYPE_DIM/featurize_option) for the data pipeline.
from agent.policy_features import (  # noqa: E402,F401
    OPT_FEAT_DIM, OPTION_TYPE_DIM, STATE_DIM, featurize_option,
)

from .parse import (  # noqa: E402
    MAIN_CONTEXT, _as_dict, _as_list, _as_int, extract_rewards, winner_from_rewards,
)


def _single_index(action: Any, n: int) -> int | None:
    """Chosen option index for a single-select action, else None.

    Replays store the action as the list of chosen option indices; the strategic
    MAIN decisions the MCTS agent makes are single-select (one index in range).
    """
    if not isinstance(action, list) or len(action) != 1:
        return None
    i = _as_int(action[0], None)
    if i is None or not (0 <= i < n):
        return None
    return i


def iter_policy_decisions(payload: Any) -> Iterator[tuple[dict, int, list, int]]:
    """Yield ``(state, me, options, chosen_idx)`` for MAIN single-select decisions.

    Walks the Kaggle env timeline directly (the value path drops the action, which
    we need here). Never raises on malformed input -- bad entries are skipped.
    """
    steps = _as_list(_as_dict(payload).get("steps"))
    for step in steps:
        for entry in _as_list(step):
            e = _as_dict(entry)
            if str(e.get("status", "")).upper() == "INACTIVE":
                continue
            obs = _as_dict(e.get("observation"))
            cur = obs.get("current")
            sel = obs.get("select")
            if not isinstance(cur, dict) or not isinstance(sel, dict):
                continue
            if _as_int(sel.get("context"), -1) != MAIN_CONTEXT:
                continue
            options = _as_list(sel.get("option"))
            if len(options) < 2:                     # single/zero option = no choice
                continue
            me = _as_int(cur.get("yourIndex"), -1)
            if me not in (0, 1):
                continue
            chosen = _single_index(e.get("action"), len(options))
            if chosen is None:                       # multi-select / invalid: skip
                continue
            yield cur, me, options, chosen


@dataclass
class PolicyRecords:
    """Accumulator of grouped (state, options, chosen) decisions."""

    states: list[np.ndarray] = field(default_factory=list)
    opts: list[np.ndarray] = field(default_factory=list)
    groups: list[int] = field(default_factory=list)
    chosen: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.groups)

    def add_decision(self, state_feat: np.ndarray, opt_feats: list[np.ndarray],
                     chosen_idx: int) -> None:
        self.states.append(state_feat)
        self.opts.extend(opt_feats)
        self.groups.append(len(opt_feats))
        self.chosen.append(chosen_idx)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.groups:
            return {"state": np.zeros((0, STATE_DIM), np.float32),
                    "opt": np.zeros((0, OPT_FEAT_DIM), np.float32),
                    "group": np.zeros((0,), np.int32),
                    "chosen": np.zeros((0,), np.int32)}
        return {"state": np.stack(self.states).astype(np.float32),
                "opt": np.stack(self.opts).astype(np.float32),
                "group": np.asarray(self.groups, np.int32),
                "chosen": np.asarray(self.chosen, np.int32)}


def episode_to_policy_records(payload: Any, records: PolicyRecords,
                              winners_only: bool = True) -> int:
    """Append policy rows from one raw replay. Returns decisions added.

    With ``winners_only`` (default), only the winning seat's decisions are kept;
    if the winner is unknown the episode contributes nothing (we can't tell which
    seat played well).
    """
    rewards = extract_rewards(payload)
    winner = winner_from_rewards(rewards)
    if winners_only and winner in (-1, 2):
        return 0
    added = 0
    for cur, me, options, chosen in iter_policy_decisions(payload):
        if winners_only and me != winner:
            continue
        try:
            feat = extract(cur, me)
        except Exception:  # noqa: BLE001  (defensive: skip malformed state)
            continue
        if feat is None or getattr(feat, "shape", (0,))[0] != STATE_DIM:
            continue
        opt_feats = [featurize_option(o, i, len(options), me)
                     for i, o in enumerate(options)]
        records.add_decision(feat, opt_feats, chosen)
        added += 1
    return added
