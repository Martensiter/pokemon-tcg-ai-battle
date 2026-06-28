"""Policy (behavioral-cloning) extraction + numpy trainer -- mock-only, no engine.

Builds a synthetic Kaggle-env replay with MAIN multi-option decisions + chosen
actions, extracts grouped (state, options, chosen) records, and smoke-trains the
numpy policy net. All engine/CSV/torch-free, so it runs in CI.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from collections import Counter

from collector.policy_extract import (
    OPT_FEAT_DIM, OPTION_TYPE_DIM, STATE_DIM, PolicyRecords, _single_index,
    episode_to_policy_records, featurize_option,
)
from agent.policy_features import ATTACK_HASH_DIM, CARD_HASH_DIM, _N_FLAGS
from collector.selftest import _state  # valid state dict (players) for features.extract

OPTS = [
    {"type": 13, "attackId": 5},                                  # ATTACK
    {"type": 14},                                                  # END
    {"type": 7, "area": 2, "index": 0, "playerIndex": 0},          # PLAY from hand
]


def _decision(me: int, chosen: int, options=OPTS):
    sel = {"context": 0, "type": 0, "minCount": 1, "maxCount": 1, "option": options}
    active = {"status": "ACTIVE", "action": [chosen],
              "observation": {"current": _state(1, me), "select": sel, "logs": []}}
    inactive = {"status": "INACTIVE", "action": [],
                "observation": {"current": None, "select": None, "logs": []}}
    return [active, inactive] if me == 0 else [inactive, active]


def _replay(winner=0):
    # seat 0 makes two decisions, seat 1 one decision
    steps = [_decision(0, 1), _decision(1, 0), _decision(0, 2)]
    rewards = [1, 0] if winner == 0 else [0, 1]
    return {"info": {"Agents": [{"Name": "a"}, {"Name": "b"}]},
            "rewards": rewards, "steps": steps}


def test_single_index():
    assert _single_index([2], 3) == 2
    assert _single_index([0], 3) == 0
    assert _single_index([3], 3) is None      # out of range
    assert _single_index([0, 1], 3) is None   # multi-select
    assert _single_index([], 3) is None
    assert _single_index("x", 3) is None


def test_featurize_option_type_and_flags():
    v = featurize_option(OPTS[0], idx=0, n=3, me=0)
    assert v.shape == (OPT_FEAT_DIM,)
    assert v[13] == 1.0                                   # type 13 one-hot
    assert v[OPTION_TYPE_DIM + 1] == 1.0                  # has attackId
    assert v[OPTION_TYPE_DIM + 0] == 0.0                  # no area
    play = featurize_option(OPTS[2], idx=2, n=3, me=0)
    assert play[OPTION_TYPE_DIM + 0] == 1.0               # has area
    assert play[OPTION_TYPE_DIM + 3] == 1.0               # playerIndex == me
    assert abs(play[OPTION_TYPE_DIM + 4] - 2 / 3) < 1e-6  # normalized position


def test_identity_distinguishes_same_type_options():
    # two different attacks (same type, same position) were IDENTICAL before;
    # now the attackId identity hash separates them.
    a = featurize_option({"type": 13, "attackId": 5}, 0, 2, 0)
    b = featurize_option({"type": 13, "attackId": 99}, 0, 2, 0)
    assert not np.array_equal(a, b)
    af = OPTION_TYPE_DIM + _N_FLAGS
    assert not np.array_equal(a[af:af + ATTACK_HASH_DIM], b[af:af + ATTACK_HASH_DIM])


def test_card_identity_resolved_from_state_db_free():
    st = _state(1, 0)  # players[0].bench == [{"id": 101, ...}]
    v = featurize_option({"type": 7, "area": 5, "index": 0, "playerIndex": 0}, 0, 2, 0, st)
    cf = OPTION_TYPE_DIM + _N_FLAGS + ATTACK_HASH_DIM
    assert v[cf:cf + CARD_HASH_DIM].sum() == 1.0   # exactly one card-id bucket set


def test_default_keeps_both_seats():
    rec = PolicyRecords()
    added = episode_to_policy_records(_replay(winner=0), rec)   # default flipped to all-seats
    assert added == 3                                           # both seats (was 2 winners-only)


def test_skip_counts_breakdown():
    c = Counter()
    rec = PolicyRecords()
    episode_to_policy_records(_replay(winner=0), rec, winners_only=True, counts=c)
    assert c["kept"] == 2 and c["skip_loser_seat"] == 1


def test_extract_winner_only():
    rec = PolicyRecords()
    added = episode_to_policy_records(_replay(winner=0), rec, winners_only=True)
    assert added == 2                              # only seat 0's two decisions
    arr = rec.arrays()
    assert arr["state"].shape == (2, STATE_DIM)
    assert arr["group"].tolist() == [3, 3]         # 3 options each
    assert arr["opt"].shape == (6, OPT_FEAT_DIM)
    assert arr["chosen"].tolist() == [1, 2]        # the chosen indices, in order


def test_extract_all_seats_and_unknown_winner():
    rec = PolicyRecords()
    assert episode_to_policy_records(_replay(winner=1), rec, winners_only=False) == 3
    # a draw / unknown winner contributes nothing under winners_only
    draw = _replay(winner=0); draw["rewards"] = [0, 0]
    rec2 = PolicyRecords()
    assert episode_to_policy_records(draw, rec2, winners_only=True) == 0


def test_policy_trainer_smoke():
    from selfplay.train_policy_np import train_policy
    rec = PolicyRecords()
    # several copies so there's something to split train/val
    for _ in range(20):
        episode_to_policy_records(_replay(winner=0), rec, winners_only=True)
    arr = rec.arrays()
    weights, metrics = train_policy(arr["state"], arr["opt"], arr["group"],
                                    arr["chosen"], hidden=[16], epochs=8, verbose=False)
    assert weights["W1"].shape[0] == STATE_DIM + OPT_FEAT_DIM    # input dim
    assert weights["W2"].shape == (16, 1)                        # scalar score head
    assert 0.0 <= metrics["val_acc"] <= 1.0
    assert metrics["decisions"] == len(arr["group"])
