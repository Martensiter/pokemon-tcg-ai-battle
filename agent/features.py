"""State -> fixed-length numpy feature vector for the value network.

Always computed from a given player's perspective (`me`): a per-player block for
`me` then for the opponent, followed by global features. Counts are scaled to
roughly [0, 1] so a plain MLP trains well without learned normalization. The same
function is used for data generation (label = did `me` win) and inference.
"""
from __future__ import annotations

import numpy as np

from .observation import active_of, in_play, prize_remaining, total_energy, n_conditions

# per-player feature count + globals -> keep in sync with _player_block / extract
_PLAYER_FEATS = 13
_GLOBAL_FEATS = 6
FEATURE_DIM = 2 * _PLAYER_FEATS + _GLOBAL_FEATS


def _player_block(pl: dict, out: list):
    act = active_of(pl)
    bench = pl.get("bench") or []
    play = in_play(pl)
    board_hp = sum(p.get("hp", 0) for p in play)
    energy_play = sum(total_energy(p) for p in play)
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


def extract(state: dict, me: int) -> np.ndarray:
    if state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    opp = 1 - me
    mp = state["players"][me]
    op = state["players"][opp]
    out: list = []
    _player_block(mp, out)
    _player_block(op, out)
    # globals
    out.append(state.get("turn", 0) / 30.0)
    out.append(1.0 if state.get("yourIndex") == me else 0.0)
    out.append((prize_remaining(op) - prize_remaining(mp)) / 6.0)
    out.append(1.0 if state.get("supporterPlayed") else 0.0)
    out.append(1.0 if state.get("energyAttached") else 0.0)
    out.append(1.0 if state.get("stadiumPlayed") else 0.0)
    return np.asarray(out, dtype=np.float32)
