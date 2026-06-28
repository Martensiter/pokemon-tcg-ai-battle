"""State -> fixed-length numpy feature vector for the value network.

Always computed from a given player's perspective (`me`): a per-player block for
`me` then for the opponent, followed by global features. Counts are scaled to
roughly [0, 1] so a plain MLP trains well without learned normalization. The same
function is used for data generation (label = did `me` win) and inference.

Identity-aware extensions (added without breaking the original block layout):
each player block now also includes the active Pokemon's hashed card id, its
evolution stage, an "appeared this turn" count, and a hashed bench card-id sum.
Globals gain first-player flag, turn-action count, and a hashed stadium id.
Together these break the "all decks look identical" ceiling without needing the
card DB at inference time (everything is read straight from the state dict).
"""
from __future__ import annotations

import numpy as np

from .observation import active_of, in_play, prize_remaining, total_energy, n_conditions

# Hash widths chosen modestly (data is ~8k samples; small dim avoids overfit).
_ACTIVE_ID_DIM = 8     # active Pokemon id, hashed one-hot per player (visible both seats)
_BENCH_ID_DIM = 8      # bench Pokemon ids, hashed accumulator (visible both seats)
# Hand is OCCLUDED for the non-acting seat -- including a per-player hand_id hash
# silently flips train/inference distributions, because at training the replay
# observation is always from the actor's view (own hand visible / opp None) but at
# inference MCTS rollouts evaluate opp-turn leaves ~55% of the time, swapping the
# orientation. The value net then sees an OOD pattern at most leaves. Removed for
# correctness; a perspective-invariant rewrite (per acting/non-acting, not per
# seat) is the right longer-term move.
_HAND_ID_DIM = 0
_DISCARD_ID_DIM = 8    # discard pile card ids (public, visible both seats)
_ENERGY_TYPE_DIM = 6   # rough energy-type histogram per player (board energies are public)
_STADIUM_DIM = 4       # stadium card id, hashed one-hot (global)

# 13 (originals) + (active_id, bench_id, discard_id, energy_type) + 5 scalars
_PLAYER_FEATS = (13 + _ACTIVE_ID_DIM + _BENCH_ID_DIM
                 + _DISCARD_ID_DIM + _ENERGY_TYPE_DIM + 5)
_GLOBAL_FEATS = 6 + _STADIUM_DIM + 2                       # +stadium +(firstPlayer, turnActionCount)
FEATURE_DIM = 2 * _PLAYER_FEATS + _GLOBAL_FEATS


def _hash_add(value, dim: int, out: list) -> None:
    """Push a dim-wide one-hot for ``value`` (feature hashing, DB-free)."""
    bucket = [0.0] * dim
    if value is not None:
        try:
            bucket[abs(int(value)) % dim] = 1.0
        except (TypeError, ValueError):
            pass
    out.extend(bucket)


def _hash_accumulate(values, dim: int, out: list) -> None:
    """Push a dim-wide hashed accumulator for ``values`` (sum then normalize)."""
    bucket = [0.0] * dim
    for v in values:
        if v is None:
            continue
        try:
            bucket[abs(int(v)) % dim] += 1.0
        except (TypeError, ValueError):
            continue
    s = sum(bucket)
    if s > 0:
        bucket = [x / s for x in bucket]    # L1-normalize so a 5-bench player isn't 5x louder
    out.extend(bucket)


def _evolution_stage(pokemon: dict) -> int:
    """0 = basic, 1 = stage1, 2 = stage2 (use preEvolution chain length)."""
    pe = pokemon.get("preEvolution") if isinstance(pokemon, dict) else None
    return len(pe) if isinstance(pe, list) else 0


def _player_block(pl: dict, out: list):
    act = active_of(pl)
    bench = pl.get("bench") or []
    play = in_play(pl)
    board_hp = sum(p.get("hp", 0) for p in play)
    energy_play = sum(total_energy(p) for p in play)
    # --- original 13 scalars ---------------------------------------------------
    out.append(prize_remaining(pl) / 6.0)
    out.append(len(play) / 6.0)
    out.append(len(bench) / 5.0)
    out.append(board_hp / 1000.0)
    out.append((act.get("hp", 0) if act else 0) / 350.0)
    out.append((act.get("maxHp", 0) if act else 0) / 350.0)
    out.append((total_energy(act) if act else 0) / 5.0)
    out.append(energy_play / 12.0)
    out.append(pl.get("handCount", 0) / 12.0)
    out.append(pl.get("deckCount", 0) / 60.0)
    out.append(len(pl.get("discard") or []) / 60.0)
    out.append(n_conditions(pl) / 5.0)
    out.append(1.0 if act is not None else 0.0)
    # --- identity-aware additions ---------------------------------------------
    # which Pokemon is in front: lets the value net learn matchup priors.
    _hash_add(act.get("id") if act else None, _ACTIVE_ID_DIM, out)
    # bench composition: hashed accumulator, normalized so size doesn't dominate.
    _hash_accumulate([p.get("id") for p in bench if isinstance(p, dict)],
                     _BENCH_ID_DIM, out)
    # (hand id intentionally dropped -- see _HAND_ID_DIM above for why.)
    # discard composition (visible to both seats -- a strong proxy for which
    # supporters / pokemons have already been spent).
    _hash_accumulate([h.get("id") for h in (pl.get("discard") or []) if isinstance(h, dict)],
                     _DISCARD_ID_DIM, out)
    # rough energy-type histogram on the whole board (which colors are in play).
    energy_types = []
    for p in play:
        for e in (p.get("energies") if isinstance(p, dict) else []) or []:
            energy_types.append(e)
    _hash_accumulate(energy_types, _ENERGY_TYPE_DIM, out)
    # evolution stage of the active mon (Basic / Stage1 / Stage2)
    out.append((_evolution_stage(act) if act else 0) / 2.0)
    # tool cards attached to the active mon (cap at 2)
    tools = act.get("tools") if act else None
    out.append((len(tools) if isinstance(tools, list) else 0) / 2.0)
    # how many of the in-play mons just appeared this turn (tempo signal)
    appeared = sum(1 for p in play if isinstance(p, dict) and p.get("appearThisTurn"))
    out.append(appeared / 3.0)
    # bench-pokemon damage spread: which bencher is closest to / furthest from KO.
    bench_hp_ratio = [
        p.get("hp", 0) / max(p.get("maxHp", 1) or 1, 1)
        for p in bench if isinstance(p, dict)
    ]
    out.append(min(bench_hp_ratio) if bench_hp_ratio else 1.0)
    out.append(sum(bench_hp_ratio) / len(bench_hp_ratio) if bench_hp_ratio else 1.0)


def extract(state: dict, me: int) -> np.ndarray:
    if state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    opp = 1 - me
    mp = state["players"][me]
    op = state["players"][opp]
    out: list = []
    _player_block(mp, out)
    _player_block(op, out)
    # --- original 6 globals ---------------------------------------------------
    out.append(state.get("turn", 0) / 30.0)
    out.append(1.0 if state.get("yourIndex") == me else 0.0)
    out.append((prize_remaining(op) - prize_remaining(mp)) / 6.0)
    out.append(1.0 if state.get("supporterPlayed") else 0.0)
    out.append(1.0 if state.get("energyAttached") else 0.0)
    out.append(1.0 if state.get("stadiumPlayed") else 0.0)
    # --- identity-aware globals -----------------------------------------------
    # stadium card id (different stadiums matter for matchup math)
    stadium = state.get("stadium")
    sid = stadium.get("id") if isinstance(stadium, dict) else None
    _hash_add(sid, _STADIUM_DIM, out)
    # first-player advantage flag
    out.append(1.0 if state.get("firstPlayer") == me else 0.0)
    # actions taken so far this turn (caps the supporter/energy/play sequencing)
    out.append((state.get("turnActionCount", 0) or 0) / 5.0)
    return np.asarray(out, dtype=np.float32)
