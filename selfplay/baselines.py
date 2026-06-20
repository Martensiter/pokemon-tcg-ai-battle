"""Reference agents for benchmarking and as MCTS rollout opponents.

Every agent is a callable `agent(obs_dict) -> list[int]` that also carries a
`.deck` (60 card ids) and a `.name`, so it can drive either seat in the harness
and double as a submission-style agent (returns the deck when select is None).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.base import BaseAgent, read_deck, random_legal  # noqa: E402,F401


class RandomAgent(BaseAgent):
    name = "random"

    def decide(self, obs: dict) -> list[int]:
        return random_legal(obs["select"], self.rng)


class GreedyAgent(BaseAgent):
    name = "greedy"

    def __init__(self, deck=None, seed=None, epsilon: float = 0.0):
        super().__init__(deck, seed)
        self.epsilon = epsilon

    def decide(self, obs: dict) -> list[int]:
        from agent.policy import choose
        return choose(obs, rng=self.rng, epsilon=self.epsilon)
