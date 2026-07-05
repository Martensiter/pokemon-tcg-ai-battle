"""Unit tests for the L1 card-value table and the L3 root-prior math.

Pure-math parts (softmax_floor) run engine-free; the card-value tests use the
local engine card DB like tests/search_test.py does.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.mcts import softmax_floor  # noqa: E402


def test_softmax_floor_sums_to_one_and_keeps_order():
    p = softmax_floor([1.0, 5.0, 3.0], temp=2.0, floor=0.1)
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[1] > p[2] > p[0]


def test_softmax_floor_uniform_floor_bounds_every_arm():
    # A huge score gap: without the floor arm 0 would get ~0 probability.
    p = softmax_floor([0.0, 100.0], temp=1.0, floor=0.2)
    assert p[0] >= 0.2 / 2 - 1e-12
    assert abs(sum(p) - 1.0) < 1e-9


def test_softmax_floor_flat_scores_are_uniform():
    p = softmax_floor([4.0, 4.0, 4.0, 4.0], temp=6.0, floor=0.15)
    for v in p:
        assert abs(v - 0.25) < 1e-9


def test_softmax_floor_empty_is_empty():
    assert softmax_floor([], temp=6.0, floor=0.15) == []


def test_baseline_has_prior_off_by_default():
    from agent import config as C
    assert float(getattr(C, "HEUR_PRIOR_C", 0.0)) == 0.0


def test_pokemon_value_orders_deck_sensibly():
    from agent.cards import get_db
    db = get_db()
    # Dragapult deck staples: the ex attacker must outrank the basic seed,
    # which must outrank the near-useless baby.
    dragapult, dreepy, budew = 121, 119, 235
    assert db.pokemon_value(dragapult) > db.pokemon_value(dreepy) > db.pokemon_value(budew)
    # Non-Pokemon (a trainer card) is worth 0.
    assert db.pokemon_value(1182) == 0.0  # Boss's Orders


def test_prizes_given_tiers():
    from agent.cards import get_db
    db = get_db()
    assert db.prizes_given(119) == 1    # Dreepy (plain basic)
    assert db.prizes_given(121) == 2    # Dragapult ex
    # Any megaEx in the pool gives 3 prizes.
    mega = next(cid for cid, c in db.all_cards().items() if c.megaEx)
    assert db.prizes_given(mega) == 3


def _fake_state(my_energies=0, op_energies=0):
    def pkm(cid, hp, n_energy):
        return {"id": cid, "hp": hp, "maxHp": hp, "energies": [4] * n_energy}
    return {
        "yourIndex": 0,
        "result": -1,
        "players": [
            {"active": [pkm(121, 320, my_energies)], "bench": [pkm(119, 70, 0)],
             "prize": [0] * 6, "handCount": 5},
            {"active": [pkm(121, 320, op_energies)], "bench": [],
             "prize": [0] * 6, "handCount": 5},
        ],
    }


def test_l2_off_by_default_and_changes_eval_when_on():
    from agent import config as C
    from agent.evaluate import evaluate
    st = _fake_state(my_energies=2, op_energies=0)  # we can attack, they cannot
    assert float(getattr(C, "L2_W", 0.0)) == 0.0    # baseline gate closed
    base = evaluate(st, 0)                          # default: L2 off
    assert evaluate(st, 0, l2_w=0.0) == base
    # Attack-ready with a KO-capable Dragapult vs an empty one: L2 must help.
    assert evaluate(st, 0, l2_w=1.0) > base


def test_l2_attack_readiness_math():
    from agent.evaluate import _attack_readiness
    # Dragapult ex: Jet Headbutt cost 1, Phantom Dive cost 2.
    need, dmg_now = _attack_readiness({"id": 121, "hp": 320, "energies": []})
    assert need == 1 and dmg_now == 0
    need, dmg_now = _attack_readiness({"id": 121, "hp": 320, "energies": [4, 4]})
    assert need == 0 and dmg_now == 200
    assert _attack_readiness(None) == (99, 0)


def test_extract_v2_extends_v1():
    import numpy as np
    from agent.features import extract, extract_v2, FEATURE_DIM, FEATURE_DIM_V2
    st = _fake_state(2, 0)
    v1 = extract(st, 0)
    v2 = extract_v2(st, 0)
    assert v1.shape == (FEATURE_DIM,)
    assert v2.shape == (FEATURE_DIM_V2,)
    assert np.allclose(v2[:FEATURE_DIM], v1)      # strict superset of v1
    assert np.isfinite(v2).all()
    # None-state fallback keeps the v2 shape too.
    assert extract_v2(None, 0).shape == (FEATURE_DIM_V2,)


def test_value_net_picks_extractor_by_weight_dim(tmp_path):
    import numpy as np
    from agent.features import extract, extract_v2, FEATURE_DIM, FEATURE_DIM_V2
    from agent.value_net import ValueNet

    def fake_weights(path, in_dim):
        rng = np.random.default_rng(0)
        np.savez(path, W1=rng.standard_normal((in_dim, 8)).astype(np.float32),
                 b1=np.zeros(8, np.float32),
                 W2=rng.standard_normal((8, 1)).astype(np.float32),
                 b2=np.zeros(1, np.float32))

    p1, p2, p3 = tmp_path / "v1.npz", tmp_path / "v2.npz", tmp_path / "bad.npz"
    fake_weights(p1, FEATURE_DIM)
    fake_weights(p2, FEATURE_DIM_V2)
    fake_weights(p3, 99)
    assert ValueNet.maybe_load(str(p1))._extract is extract
    assert ValueNet.maybe_load(str(p2))._extract is extract_v2
    assert ValueNet.maybe_load(str(p3)) is None   # unknown dim rejected
    # v2 net produces a sane value end-to-end on a real-ish state.
    v = ValueNet.maybe_load(str(p2)).value(_fake_state(1, 1), 0)
    assert -1.0 <= v <= 1.0
