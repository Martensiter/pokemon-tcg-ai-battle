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


def _board_hp(player: dict) -> int:
    return sum(p.get("hp", 0) for p in in_play(player))


def _energy_on_attackers(player: dict) -> int:
    return sum(total_energy(p) for p in in_play(player))


def evaluate(state: dict, me: int | None = None) -> float:
    """Value of `state` for player `me` (defaults to state.yourIndex)."""
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
