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
_DISCARD_ID_DIM = 8    # discard pile card ids (public, visible both seats)
_ENERGY_TYPE_DIM = 6   # rough energy-type histogram per player (board energies are public)
_STADIUM_DIM = 4       # stadium card id, hashed one-hot (global)
# Acting-seat hand: stored ONCE at the global block (not per-player), so it stays
# perspective-invariant. The engine returns observations from the to-move seat's
# view, where ``state["players"][yourIndex]["hand"]`` is visible (and the other
# seat's is None). Keying the hash on the acting seat -- not on the original me --
# means MCTS opp-turn leaves see the *opp's* hand here, exactly as training data
# from opp's MAIN decisions did. So the model sees the same distribution at
# training and at any leaf, regardless of whose turn it is.
_ACTING_HAND_DIM = 8

# 13 (originals) + (active_id, bench_id, discard_id, energy_type) + 5 scalars
_PLAYER_FEATS = (13 + _ACTIVE_ID_DIM + _BENCH_ID_DIM
                 + _DISCARD_ID_DIM + _ENERGY_TYPE_DIM + 5)
_GLOBAL_FEATS = (6 + _STADIUM_DIM + 2                       # +stadium +(firstPlayer, turnActionCount)
                 + _ACTING_HAND_DIM + 1)                    # +acting-hand +acting==me flag
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
    # ACTING-seat hand: keyed on yourIndex (the visible hand), NOT on me. This
    # makes the feature perspective-invariant -- the acting seat's hand is the
    # one the replay observation revealed at training time and is also the only
    # hand visible at MCTS leaves, regardless of which seat is to-move.
    acting = state.get("yourIndex", me)
    acting_hand = (state["players"][acting].get("hand")
                   if 0 <= acting < len(state.get("players") or []) else None) or []
    _hash_accumulate([h.get("id") for h in acting_hand if isinstance(h, dict)],
                     _ACTING_HAND_DIM, out)
    # "is the acting seat me?" -- lets the net learn that acting-hand is mine
    # when this flag is 1 and theirs when 0 (so it can use the same block both ways).
    out.append(1.0 if acting == me else 0.0)
    # first-player advantage flag
    out.append(1.0 if state.get("firstPlayer") == me else 0.0)
    # actions taken so far this turn (caps the supporter/energy/play sequencing)
    out.append((state.get("turnActionCount", 0) or 0) / 5.0)
    return np.asarray(out, dtype=np.float32)


# --- v2: tactical features from the L1/L2 evaluation layers --------------------
# Appended AFTER the v1 vector, so v1 weight files keep working: the value net
# picks the extractor by the loaded W1 input dim (FEATURE_DIM = v1, FEATURE_DIM_V2
# = v2). Unlike the hash features above these need the engine card DB, which the
# agent already loads at runtime for its greedy policy.

_V2_PLAYER_EXTRAS = 6
_V2_GLOBAL_EXTRAS = 3
FEATURE_DIM_V2 = FEATURE_DIM + 2 * _V2_PLAYER_EXTRAS + _V2_GLOBAL_EXTRAS


def _v2_player_extras(pl: dict, other: dict, out: list) -> None:
    from .evaluate import _attack_readiness
    from .cards import get_db
    db = get_db()
    act, op_act = active_of(pl), active_of(other)
    need, dmg_now = _attack_readiness(act)
    out.append(min(need, 3) / 3.0)                    # attack distance (energies to go)
    out.append(min(dmg_now, 400) / 400.0)             # damage available right now
    ko = 1.0 if (op_act is not None and dmg_now >= (op_act.get("hp") or 10 ** 9)) else 0.0
    out.append(ko)                                    # KO available right now
    out.append((db.pokemon_value(act.get("id")) if act else 0.0) / 400.0)   # L1: battle spot
    bench_v = sum(db.pokemon_value(p.get("id")) for p in (pl.get("bench") or [])
                  if isinstance(p, dict))
    out.append(min(bench_v, 1200.0) / 1200.0)         # L1: bench quality
    if act is not None:
        c = db.card(act.get("id"))
        retreat = c.retreatCost if c else 2
        out.append((total_energy(act) - retreat) / 4.0)   # mobility of the active
    else:
        out.append(0.0)


def _kos_to_win(pl: dict, other: dict) -> float:
    """Rough race metric: KOs still needed to clear our remaining prizes,
    assuming each KO on the opponent's CURRENT active pays its prize count."""
    from .cards import get_db
    left = prize_remaining(pl)
    if left <= 0:
        return 0.0
    target = active_of(other)
    per = get_db().prizes_given(target.get("id")) if target else 1
    return -(-left // max(per, 1))  # ceil division


def extract_v2(state: dict, me: int) -> np.ndarray:
    """v1 features + L1/L2 tactical extras (attack distance, KO race, quality)."""
    if state is None:
        return np.zeros(FEATURE_DIM_V2, dtype=np.float32)
    from .evaluate import _attack_readiness
    base = extract(state, me)
    opp = 1 - me
    mp = state["players"][me]
    op = state["players"][opp]
    out: list = []
    _v2_player_extras(mp, op, out)
    _v2_player_extras(op, mp, out)
    # prize-race differential: fewer KOs-to-win than the opponent = closing faster.
    my_kos, op_kos = _kos_to_win(mp, op), _kos_to_win(op, mp)
    out.append((op_kos - my_kos) / 6.0)
    # "can finish right now" flags for both seats (KO now AND it clears the prizes).
    from .cards import get_db
    db = get_db()
    my_act, op_act = active_of(mp), active_of(op)
    _, my_dmg = _attack_readiness(my_act)
    _, op_dmg = _attack_readiness(op_act)
    my_finish = (op_act is not None and my_dmg >= (op_act.get("hp") or 10 ** 9)
                 and prize_remaining(mp) <= db.prizes_given(op_act.get("id")))
    op_finish = (my_act is not None and op_dmg >= (my_act.get("hp") or 10 ** 9)
                 and prize_remaining(op) <= db.prizes_given(my_act.get("id")))
    out.append(1.0 if my_finish else 0.0)
    out.append(1.0 if op_finish else 0.0)
    return np.concatenate([base, np.asarray(out, dtype=np.float32)])
