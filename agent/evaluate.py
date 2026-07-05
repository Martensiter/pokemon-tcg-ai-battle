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


def _energy_on_attackers(player: dict) -> int:
    return sum(total_energy(p) for p in in_play(player))


def evaluate(state: dict, me: int | None = None, l2_w: float | None = None) -> float:
    """Value of `state` for player `me` (defaults to state.yourIndex).

    `l2_w` gates the L2 dynamic terms; None falls back to eval_params.L2_W.
    Passed explicitly by make_leaf_evaluator so config-sweeps can vary it.
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
