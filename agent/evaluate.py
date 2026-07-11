"""Heuristic state evaluation, V(state) from the to-move player's perspective.

Returns a scalar in roughly [-1, 1] where +1 ~ winning. Prize differential
dominates (you win at 0 prizes remaining); board strength, bench safety,
energy development, and special conditions are secondary. This is the v0 leaf
evaluation for MCTS and the basis of the greedy baseline; the learned value net
(Stage 5) augments/replaces it.
"""
from __future__ import annotations

from .observation import (
    my_state, opp_state, active_of, in_play, prize_remaining,
    total_energy, n_conditions,
)
from .cards import get_db
from . import eval_params as EP


def _board_hp(player: dict) -> int:
    return sum(p.get("hp", 0) for p in in_play(player))


# --- L2: dynamic board terms (OFF unless eval_params.L2_W > 0) -----------------

def _attack_readiness(pkm: dict | None) -> tuple[int, int]:
    """(min energies still needed to use any attack, best damage usable NOW).

    Color-blind pip count (v0): attached energy count vs attack pip count.
    99 needed = no attacks known / no Pokemon.
    """
    if pkm is None:
        return 99, 0
    db = get_db()
    attached = len(pkm.get("energies") or [])
    need_min, best_now = 99, 0
    for ai in db.attacks_of(pkm.get("id")):
        need = max(0, ai.cost - attached)
        if need < need_min:
            need_min = need
        if need == 0 and ai.damage > best_now:
            best_now = ai.damage
    return need_min, best_now


def _board_quality(player: dict) -> float:
    """L1 card values over the board: the active counts full, bench partial."""
    db = get_db()
    q = 0.0
    act = active_of(player)
    if act is not None:
        q += EP.L1_BATTLE_W * db.pokemon_value(act.get("id"))
    for p in (player.get("bench") or []):
        if p is not None:
            q += EP.L1_BENCH_W * db.pokemon_value(p.get("id"))
    return q


def _l2_terms(mp: dict, op: dict) -> float:
    """Extra evaluation in roughly [-1, 1]: attack distance, KO threat, quality."""
    my_act, op_act = active_of(mp), active_of(op)
    my_need, my_dmg = _attack_readiness(my_act)
    op_need, op_dmg = _attack_readiness(op_act)

    # Attack-distance race: being fewer energies away from attacking is worth
    # more; 3+ energies away is as bad as it gets.
    ready = (min(op_need, 3) - min(my_need, 3)) / 3.0

    # KO-now threat, with the opponent's threat weighed heavier (loss aversion).
    my_ko = 1.0 if (op_act is not None and my_dmg >= (op_act.get("hp") or 10 ** 9)) else 0.0
    op_ko = 1.0 if (my_act is not None and op_dmg >= (my_act.get("hp") or 10 ** 9)) else 0.0
    threat = my_ko - EP.L2_ASYM * op_ko

    # L1 board quality differential (battle spot weighted over bench).
    my_q, op_q = _board_quality(mp), _board_quality(op)
    quality = (my_q - op_q) / (my_q + op_q + 300.0)

    return (EP.L2_READY_W * ready
            + EP.L2_THREAT_W * threat
            + EP.L2_QUALITY_W * quality)


def _hand_pokemon_value(player: dict) -> float:
    """② Sum of L1 card values of Pokemon in this player's VISIBLE hand.

    Returns 0 when the hand is hidden (opponent seat: only handCount is known), so
    the term is effectively one-sided toward whichever seat's hand is revealed.
    """
    hand = player.get("hand")
    if not isinstance(hand, list):
        return 0.0
    db = get_db()
    return sum(db.pokemon_value(c.get("id")) for c in hand if isinstance(c, dict))


def _hand_pokemon_count(player: dict) -> int:
    """② control: how many Pokemon sit in this player's VISIBLE hand."""
    hand = player.get("hand")
    if not isinstance(hand, list):
        return 0
    db = get_db()
    return sum(1 for c in hand if isinstance(c, dict) and db.is_pokemon(c.get("id")))


def _hand_value_split(player: dict) -> tuple[float, float]:
    """②' (basic_value, evolution_value) of the VISIBLE hand.

    Basics are deployable now (asset); evolution cards may be the stuck-in-hand
    loss symptom the ③ outcome regression flagged. Split so each can be weighed
    (even with opposite signs) independently.
    """
    hand = player.get("hand")
    if not isinstance(hand, list):
        return 0.0, 0.0
    db = get_db()
    basic = evo = 0.0
    for c in hand:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        v = db.pokemon_value(cid)
        if v <= 0:
            continue
        if db.stage_of(cid) == 0:
            basic += v
        else:
            evo += v
    return basic, evo


# --- ⑦ role/"利き" terms (OFF unless config.ROLE_W > 0) --------------------------

def _threat_on(att: dict | None, dfn: dict | None) -> float:
    """How hard `att` threatens `dfn` RIGHT AT THE SPOT: best type-aware damage
    over the defender's remaining hp, capped (overkill is not extra value)."""
    if att is None or dfn is None:
        return 0.0
    db = get_db()
    best = 0
    for ai in db.attacks_of(att.get("id")):
        dmg = attack_damage_estimate(att.get("id"), dfn.get("id"), ai.damage)
        if dmg > best:
            best = dmg
    return min(best / max(dfn.get("hp") or 1, 1), 1.5)


def _bench_eki(player: dict) -> float:
    """Bench "利き": utility the bench projects (abilities + charged backups)."""
    db = get_db()
    s = 0.0
    for p in (player.get("bench") or []):
        if not isinstance(p, dict):
            continue
        c = db.card(p.get("id"))
        if c is None:
            continue
        if c.skills:
            s += 1.0
        s += min(total_energy(p), 3) / 3.0 * 0.5
    return s


def _role_terms(mp: dict, op: dict) -> float:
    my_act, op_act = active_of(mp), active_of(op)
    threat_diff = _threat_on(my_act, op_act) - _threat_on(op_act, my_act)
    bench_diff = (_bench_eki(mp) - _bench_eki(op)) / 4.0
    return 0.6 * threat_diff + 0.4 * bench_diff


# --- ⑨ bad shapes (OFF unless config.BAD_SHAPE_W > 0) ----------------------------
# Inner mix 0.8 : 1.0 comes from the regression coefficients (-0.105 vs -0.132).

def _bad_shapes(pl: dict) -> float:
    db = get_db()
    chip_fragile = 0
    for p in in_play(pl):
        mx = p.get("maxHp") or 0
        dmg = mx - (p.get("hp") or mx)
        if dmg > 0 and mx <= 90:
            chip_fragile += 1
    bound = 0.0
    act = active_of(pl)
    if act is not None:
        c = db.card(act.get("id"))
        if c is not None and len(act.get("energies") or []) == 0 and (c.retreatCost or 0) >= 2:
            bound = 1.0
    return 0.8 * chip_fragile + 1.0 * bound


# --- ⑧ anti-stall axis (OFF unless config.ANTI_STALL_W > 0) ----------------------

_STALL_NAMES = frozenset({"Crustle", "Dwebble", "Walrein", "Spheal"})
_BREAKER_NAMES = frozenset({"Dusknoir", "Dusclops", "Duskull", "Munkidori"})
_ids_cache: dict[frozenset, frozenset] = {}


def _ids_by_name(names: frozenset) -> frozenset:
    got = _ids_cache.get(names)
    if got is None:
        db = get_db()
        got = frozenset(cid for cid, c in db.all_cards().items() if c.name in names)
        _ids_cache[names] = got
    return got


def _anti_stall(mp: dict, op: dict) -> float:
    """Bonus for holding stall-breakers once the opponent reveals a stall line."""
    stall = _ids_by_name(_STALL_NAMES)
    revealed = [p.get("id") for p in in_play(op) if isinstance(p, dict)]
    revealed += [c.get("id") for c in (op.get("discard") or []) if isinstance(c, dict)]
    if not any(cid in stall for cid in revealed):
        return 0.0
    breakers = _ids_by_name(_BREAKER_NAMES)
    mine = sum(1 for p in in_play(mp) if isinstance(p, dict) and p.get("id") in breakers)
    return min(mine, 3) / 3.0


def _energy_on_attackers(player: dict) -> int:
    return sum(total_energy(p) for p in in_play(player))


def evaluate(state: dict, me: int | None = None, l2_w: float | None = None,
             hand_value_w: float | None = None, hand_count_w: float | None = None,
             hand_basic_w: float | None = None, hand_evo_w: float | None = None,
             role_w: float | None = None, anti_stall_w: float | None = None,
             bad_shape_w: float | None = None) -> float:
    """Value of `state` for player `me` (defaults to state.yourIndex).

    `l2_w` gates the L2 dynamic terms; None falls back to eval_params.L2_W.
    `hand_value_w`/`hand_count_w` gate the ② hand terms, `role_w` the ⑦
    role/"利き" terms, `anti_stall_w` the ⑧ axis (None/0 = OFF, byte-identical).
    All passed explicitly by make_leaf_evaluator so config-sweeps can vary them.
    """
    if state is None:
        return 0.0
    if me is None:
        me = state["yourIndex"]
    opp = 1 - me
    mp = state["players"][me]
    op = state["players"][opp]

    # Terminal: decisive.
    res = state.get("result", -1)
    if res != -1:
        if res == 2:
            return 0.0
        return 1.0 if res == me else -1.0

    # --- Prize differential (primary). Each player starts with 6. ---
    my_pr = prize_remaining(mp)
    op_pr = prize_remaining(op)
    # taken = 6 - remaining; ahead if I've taken more than opponent.
    prize_diff = (op_pr - my_pr) / 6.0  # in [-1, 1]
    # Closing bonus: being near 0 prizes is worth extra (closer to the win).
    close = (6 - my_pr) ** 2 / 72.0 - (6 - op_pr) ** 2 / 72.0  # in [-0.5, 0.5]

    # --- Board strength (HP in play). ---
    my_hp = _board_hp(mp)
    op_hp = _board_hp(op)
    denom = my_hp + op_hp + 1
    board = (my_hp - op_hp) / denom  # in (-1, 1)

    # --- Bench safety: an empty bench means a KO on the active loses the game. ---
    def bench_safety(pl):
        n_bench = len(pl.get("bench") or [])
        act = active_of(pl)
        if act is None:
            return -0.5  # no active is very bad (unless setup)
        if n_bench == 0:
            return -0.3
        if n_bench == 1:
            return -0.05
        return 0.0
    safety = bench_safety(mp) - bench_safety(op)

    # --- Energy development on attackers. ---
    my_e = _energy_on_attackers(mp)
    op_e = _energy_on_attackers(op)
    energy = (my_e - op_e) / (my_e + op_e + 4.0)

    # --- Card advantage (hand + deck, modest weight). ---
    hand_adv = (mp.get("handCount", 0) - op.get("handCount", 0)) / 12.0

    # --- Special conditions: bad on me, good on opponent. ---
    cond = (n_conditions(op) - n_conditions(mp)) * 0.04

    score = (
        2.0 * prize_diff
        + 1.0 * close
        + 0.6 * board
        + 0.5 * safety
        + 0.25 * energy
        + 0.15 * hand_adv
        + cond
    )
    w2 = 0.0 if l2_w is None else l2_w  # gate owned by config.L2_W (via caller)
    if w2 > 0:  # L2 dynamic terms (attack distance / KO threat / L1 quality)
        score += w2 * _l2_terms(mp, op)
    hw = 0.0 if hand_value_w is None else hand_value_w  # ② gate (config.HAND_VALUE_W)
    if hw != 0:  # credit deployable Pokemon held in hand (visible seat only)
        score += hw * (_hand_pokemon_value(mp) - _hand_pokemon_value(op)) / 800.0
    hcw = 0.0 if hand_count_w is None else hand_count_w  # ② count control
    if hcw != 0:
        score += hcw * (_hand_pokemon_count(mp) - _hand_pokemon_count(op)) / 6.0
    hbw = 0.0 if hand_basic_w is None else hand_basic_w  # ②' deployable basics
    hew = 0.0 if hand_evo_w is None else hand_evo_w      # ②' stuck evolutions
    if hbw != 0 or hew != 0:
        mb, mev = _hand_value_split(mp)
        ob, oev = _hand_value_split(op)
        score += hbw * (mb - ob) / 800.0 + hew * (mev - oev) / 800.0
    rw = 0.0 if role_w is None else role_w  # ⑦ gate (config.ROLE_W)
    if rw != 0:
        score += rw * _role_terms(mp, op)
    asw = 0.0 if anti_stall_w is None else anti_stall_w  # ⑧ gate (config.ANTI_STALL_W)
    if asw != 0:
        score += asw * (_anti_stall(mp, op) - _anti_stall(op, mp))
    bsw = 0.0 if bad_shape_w is None else bad_shape_w  # ⑨ gate (config.BAD_SHAPE_W)
    if bsw != 0:  # penalize OUR bad shapes, credit the opponent's
        score -= bsw * (_bad_shapes(mp) - _bad_shapes(op))
    # squash into [-1, 1]
    return max(-1.0, min(1.0, score / 3.0))


def attack_damage_estimate(attacker_id: int, defender_id: int, base_damage: int) -> int:
    """Estimate effective damage applying weakness (x2) / resistance (-30).

    Variable-damage attacks ('x' attacks) are passed in via base_damage as their
    nominal value; this is a heuristic only.
    """
    db = get_db()
    atk = db.card(attacker_id)
    dfn = db.card(defender_id)
    if atk is None or dfn is None or base_damage <= 0:
        return max(0, base_damage)
    dmg = base_damage
    if dfn.weakness is not None and dfn.weakness == atk.energyType:
        dmg *= 2
    if dfn.resistance is not None and dfn.resistance == atk.energyType:
        dmg = max(0, dmg - 30)
    return dmg
