"""Shared test helpers: make ``collector`` importable and synthesise replays.

No network, no engine binary, no torch -- only numpy + stdlib. Synthetic replays
mimic the two wrapper formats the parser supports.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def make_player(hp: int = 100, n_bench: int = 1, n_prize: int = 3,
                hand: int = 5, deck: int = 40, discard: int = 2):
    """A minimal-but-plausible PlayerState dict (extra fields default to 0)."""
    return {
        "active": [{"id": 100, "hp": hp, "maxHp": 120, "energies": [0, 1]}],
        "bench": [{"id": 101, "hp": 70, "maxHp": 70, "energies": [0]} for _ in range(n_bench)],
        "benchMax": 5,
        "deckCount": deck,
        "discard": [{"id": 1} for _ in range(discard)],
        "prize": [{"id": 0} for _ in range(n_prize)],
        "handCount": hand,
        "hand": None,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }


def make_state(turn: int, your_index: int, result: int = -1):
    return {
        "turn": turn,
        "turnActionCount": 1,
        "yourIndex": your_index,
        "firstPlayer": 0,
        "supporterPlayed": False,
        "stadiumPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": result,
        "stadium": [],
        "looking": None,
        "players": [make_player(), make_player(hp=80)],
    }


def make_frame(turn: int, your_index: int, context: int = 0, result: int = -1, logs=None):
    return {
        "current": make_state(turn, your_index, result=result),
        "logs": logs or [],
        "select": {"context": context, "type": 0, "minCount": 1, "maxCount": 1,
                   "option": [{"type": 14}]},
    }


def make_frames(n_main: int = 4, winner: int = 0):
    """A sequence of MAIN frames alternating seats, ending with a RESULT log."""
    frames = []
    for i in range(n_main):
        frames.append(make_frame(turn=i + 1, your_index=i % 2, context=0))
    # terminal frame: result state + RESULT log (type 23)
    term = make_frame(turn=n_main + 1, your_index=0, context=0, result=winner,
                      logs=[{"type": 23, "result": winner, "reason": 1}])
    frames.append(term)
    return frames


def make_episode_steps(winner: int = 0, deck=None, agents=("alice", "bob")):
    """Kaggle environment wrapper: steps[0][0].observation.visualize + rewards."""
    deck = deck or list(range(1, 61))
    frames = make_frames(n_main=4, winner=winner)
    rewards = [1, 0] if winner == 0 else ([0, 1] if winner == 1 else [0, 0])
    return {
        "info": {"Agents": [{"Name": a} for a in agents]},
        "rewards": rewards,
        "steps": [
            [{"observation": {"visualize": frames}, "action": deck},
             {"observation": {}, "action": deck}],
            [{"action": deck}, {"action": deck}],
        ],
    }


def make_episode_visualize(winner: int = 0):
    """Lower-level wrapper: top-level visualize array."""
    return {"visualize": make_frames(n_main=4, winner=winner)}
