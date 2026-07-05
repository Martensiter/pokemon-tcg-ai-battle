"""Tunable coefficients for the hand-crafted card/board evaluation (L1/L2).

One flat module so every knob is overridable via PTCG_* env vars. Search-behavior
knobs (the L3 root prior: HEUR_PRIOR_C etc.) live in agent/config.py instead so
tools/sweep_config.py can sweep them alongside the other search parameters.

Layer map (session design note):
  L1  static per-card value        -> cards.py::CardDB.pokemon_value()
  L2  dynamic board evaluation     -> evaluate.py (v2 terms)
"""
from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# --- L1: static per-card value ------------------------------------------------
# value = hp/prizes_given + ATK_EFF_W * best_damage/(weighted_cost+1)
#         - EVO_COEF * stage + ABILITY_COEF * has_ability
# Weighted energy cost: colorless pips count COLORLESS_W, pips matching the
# card's own type count 1.0, off-color pips count OFFCOLOR_W (splash cost).
ATK_EFF_W = _f("PTCG_ATK_EFF_W", 1.0)
EVO_COEF = _f("PTCG_EVO_COEF", 15.0)
ABILITY_COEF = _f("PTCG_ABILITY_COEF", 25.0)
COLORLESS_W = _f("PTCG_COLORLESS_W", 0.5)
OFFCOLOR_W = _f("PTCG_OFFCOLOR_W", 1.5)
