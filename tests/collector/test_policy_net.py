"""Policy-net inference + PUCT selection -- the engine-free pieces the MCTS uses.

mcts.py itself imports the engine so it can't run in CI; here we test the numpy
parts it calls (PolicyNet scoring/priors/loading and puct_select), plus a
train->load round-trip proving train_policy_np output is inference-compatible.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.policy_features import OPT_FEAT_DIM, STATE_DIM
from agent.policy_net import PolicyNet, puct_select
from collector.selftest import _state            # valid state dict for features.extract

IN = STATE_DIM + OPT_FEAT_DIM
OPTS = [{"type": 13, "attackId": 5}, {"type": 14}, {"type": 7, "area": 2, "index": 0}]


def _toy_net(hidden=8, seed=0):
    rng = np.random.default_rng(seed)
    return PolicyNet([
        ((rng.standard_normal((IN, hidden))).astype(np.float32), np.zeros(hidden, np.float32)),
        ((rng.standard_normal((hidden, 1))).astype(np.float32), np.zeros(1, np.float32)),
    ])


def test_scores_shape_and_priors_normalized():
    net = _toy_net()
    assert net.scores(np.zeros((4, IN), np.float32)).shape == (4,)
    p = net.priors(_state(1, 0), 0, OPTS)
    assert p.shape == (3,)
    assert abs(float(p.sum()) - 1.0) < 1e-5 and (p >= 0).all()


def test_priors_none_on_empty_options():
    assert _toy_net().priors(_state(1, 0), 0, []) is None


def test_maybe_load_roundtrip_and_dim_guard(tmp_path):
    net = _toy_net()
    good = tmp_path / "policy.npz"
    np.savez(good, W1=net.layers[0][0], b1=net.layers[0][1],
             W2=net.layers[1][0], b2=net.layers[1][1])
    assert PolicyNet.maybe_load(str(good)) is not None
    # wrong input dim -> rejected (won't silently mis-score)
    bad = tmp_path / "bad.npz"
    np.savez(bad, W1=np.zeros((5, 4), np.float32), b1=np.zeros(4, np.float32),
             W2=np.zeros((4, 1), np.float32), b2=np.zeros(1, np.float32))
    assert PolicyNet.maybe_load(str(bad)) is None
    assert PolicyNet.maybe_load("/no/such/file.npz") is None


def test_puct_prior_and_value_tradeoff():
    # all unvisited & equal value -> highest prior wins
    assert puct_select([0, 0, 0], [0, 0, 0], [0.1, 0.8, 0.1], c=2.0) == 1
    # c=0 ignores the prior entirely (plain Q; all 0 -> first index)
    assert puct_select([0, 0, 0], [0, 0, 0], [0.1, 0.8, 0.1], c=0.0) == 0
    # a strong exploited value beats a high prior
    assert puct_select([1, 1], [0.9, 0.0], [0.1, 0.9], c=0.5) == 0
    # None prior -> uniform, still returns a valid index
    assert puct_select([0, 0], [0, 0], None, c=1.0) in (0, 1)


def test_train_output_is_inference_compatible(tmp_path):
    from collector.policy_extract import PolicyRecords, episode_to_policy_records
    from selfplay.train_policy_np import train_policy

    def _decision(me, chosen):
        sel = {"context": 0, "type": 0, "minCount": 1, "maxCount": 1, "option": OPTS}
        act = {"status": "ACTIVE", "action": [chosen],
               "observation": {"current": _state(1, me), "select": sel, "logs": []}}
        ina = {"status": "INACTIVE", "action": [],
               "observation": {"current": None, "select": None, "logs": []}}
        return [act, ina] if me == 0 else [ina, act]

    replay = {"rewards": [1, 0], "steps": [_decision(0, 1), _decision(0, 2)]}
    rec = PolicyRecords()
    for _ in range(20):
        episode_to_policy_records(replay, rec, winners_only=True)
    a = rec.arrays()
    w, _ = train_policy(a["state"], a["opt"], a["group"], a["chosen"],
                        hidden=[16], epochs=5, verbose=False)
    f = tmp_path / "policy.npz"
    np.savez(f, **w)
    net = PolicyNet.maybe_load(str(f))
    assert net is not None                      # train layout loads as inference net
    p = net.priors(_state(1, 0), 0, OPTS)
    assert p.shape == (3,) and abs(float(p.sum()) - 1.0) < 1e-5
