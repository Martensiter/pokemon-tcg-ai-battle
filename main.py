"""Kaggle submission entry point.

The competition harness imports this module and calls `agent(obs_dict)` once per
decision. On the first call `obs.select is None` and we return our 60-card deck;
otherwise we return option indices chosen by the determinized-MCTS + value-net
agent (see the `agent/` package).
"""
import os
import sys

# Kaggle's harness runs this file via exec(), so __file__ may be undefined.
# Build a candidate-path list defensively and add whichever exist to sys.path.
_PATHS = ["/kaggle_simulations/agent"]
try:
    _PATHS.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
_PATHS.append(os.getcwd())

for _p in _PATHS:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from agent.agent import agent  # noqa: E402,F401  (re-exported as main.agent)


if __name__ == "__main__":
    # tiny self-check: run one decision against a fresh battle
    from cg.game import battle_start, battle_select, battle_finish
    from agent.base import read_deck
    deck = read_deck()
    obs, _ = battle_start(deck, list(deck))
    out = agent(obs)
    print("first decision ->", out)
    battle_finish()
