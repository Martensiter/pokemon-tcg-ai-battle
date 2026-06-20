"""Base agent + small helpers, kept dependency-free so the submission package
(`agent/` + `cg/` + main.py + deck.csv) is self-contained (no selfplay/ needed).
"""
from __future__ import annotations

import os
import random

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_AGENT_DIR)


def deck_path() -> str:
    for p in (os.path.join(_ROOT, "deck.csv"), "deck.csv",
              "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(p):
            return p
    return os.path.join(_ROOT, "deck.csv")


def read_deck(path: str | None = None) -> list[int]:
    path = path or deck_path()
    with open(path) as f:
        ids = [int(x) for x in f.read().split("\n") if x.strip()]
    assert len(ids) == 60, f"deck must be 60 cards, got {len(ids)}"
    return ids


def random_legal(sel: dict, rng: random.Random) -> list[int]:
    """A contract-valid random selection for a SelectData dict."""
    n = len(sel["option"])
    lo, hi = sel["minCount"], min(sel["maxCount"], n)
    if hi <= 0:
        return []
    k = rng.randint(lo, hi) if hi >= lo else lo
    return rng.sample(range(n), max(0, min(k, n)))


class BaseAgent:
    name = "base"

    def __init__(self, deck: list[int] | None = None, seed: int | None = None):
        self.deck = deck if deck is not None else read_deck()
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> list[int]:
        if obs.get("select") is None:
            return list(self.deck)
        return self.decide(obs)

    def decide(self, obs: dict) -> list[int]:
        raise NotImplementedError
