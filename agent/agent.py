"""The competition agent: determinized MCTS for strategic single-select decisions,
with the greedy heuristic as a fast, robust fallback for everything else.

`agent(obs_dict) -> list[int]` is the Kaggle entry point (returns the deck when
`select is None`). `MctsAgent` is the harness-friendly object form.
"""
from __future__ import annotations

import os
import random
import sys

from .base import BaseAgent
from .policy import choose as greedy_choose
from .mcts import MCTS
from . import config as C

# Per-move search diagnostics to stderr (visible in Kaggle's Agent Logs). On by
# default so a submission reveals how many simulations actually fit in the time
# budget on the grader's CPU -- the one thing the duration logs can't show. Set
# PTCG_LOG_SIMS=0 to silence. One short line per MCTS decision; negligible cost.
_LOG_SIMS = os.environ.get("PTCG_LOG_SIMS", "1") != "0"


def _candidates(sel: dict):
    """Enumerate full selections for a single-select decision, or None if the
    decision is multi-select (handled by the greedy policy)."""
    n = len(sel["option"])
    lo, hi = sel["minCount"], sel["maxCount"]
    if n == 0:
        return [[]]
    if hi == 1 and lo <= 1:
        cands = [[i] for i in range(n)]
        if lo == 0:
            cands.append([])
        return cands
    return None


class MctsAgent(BaseAgent):
    name = "mcts"

    def __init__(self, deck=None, seed=None, eval_fn=None, cfg=C):
        super().__init__(deck, seed)
        self.cfg = cfg
        if eval_fn is None:
            from .value_net import make_leaf_evaluator
            eval_fn = make_leaf_evaluator(cfg)
        self.mcts = MCTS(self.deck, self.rng, eval_fn=eval_fn, cfg=cfg)

    def decide(self, obs: dict) -> list[int]:
        sel = obs["select"]
        n = len(sel["option"])
        if n == 0:
            return []
        if n == 1:
            return [0] if sel["minCount"] >= 1 else greedy_choose(obs, rng=self.rng)

        cands = _candidates(sel)
        if cands is None:
            return greedy_choose(obs, rng=self.rng)  # multi-select fallback
        try:
            pick = self.mcts.search(obs, cands)
        except Exception:
            pick = None
        if _LOG_SIMS:
            # sims = simulations completed in the budget; fails = determinizations
            # that errored; fallback = sims < MIN so greedy was used instead.
            print(f"sims={self.mcts.last_sims} fails={self.mcts.last_fails} "
                  f"opts={len(cands)} {'fallback' if pick is None else 'mcts'}",
                  file=sys.stderr, flush=True)
        if pick is None:
            return greedy_choose(obs, rng=self.rng)
        return pick


# --- Kaggle entry point ---------------------------------------------------
_AGENT: MctsAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    global _AGENT
    if _AGENT is None:
        from .base import read_deck
        _AGENT = MctsAgent(deck=read_deck(), seed=random.randrange(1 << 30))
    if obs_dict.get("select") is None:
        return list(_AGENT.deck)  # initial deck selection
    return _AGENT.decide(obs_dict)
