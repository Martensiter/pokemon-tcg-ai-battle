"""Determinized Monte-Carlo search (Information-Set MCTS, flat-UCB variant).

For a single-select decision we treat the options as arms of a UCB1 bandit. Each
simulation: sample a determinization of the hidden information, `search_begin` a
private forward model, play the chosen option, run a truncated greedy rollout
(both seats) through `search_step`, then evaluate the leaf. Because the engine
resolves shuffles/coins stochastically inside `search_step`, re-sampling a world
each simulation makes this a sound imperfect-information search rather than a
brittle deterministic tree. The leaf evaluator is injected so the heuristic
(Stage 4) and the learned value net (Stage 5) are interchangeable.
"""
from __future__ import annotations

import json
import math
import time
import ctypes
import random

from cg import api
from cg.sim import lib
from cg.api import to_observation_class

from . import config as C
from .determinize import determinize
from .policy import choose as policy_choose
from .evaluate import evaluate as heuristic_evaluate
from .policy_net import PolicyNet, puct_select  # numpy-only; opt-in policy prior


def _step_dict(search_id: int, select: list[int]) -> dict:
    """Advance a search state, returning the raw dict (skips dataclass parsing)."""
    arr = (ctypes.c_int * len(select))(*select)
    js = lib.SearchStep(api.agent_ptr, search_id, arr, len(select))
    d = json.loads(js)
    if d["error"] != 0:
        raise RuntimeError(f"search_step error {d['error']}")
    return d["state"]  # {"observation": {...}, "searchId": int}


def terminal_value(state: dict, me: int) -> float:
    res = state.get("result", -1)
    if res == 2:
        return 0.0
    return 1.0 if res == me else -1.0


class MCTS:
    def __init__(self, deck: list[int], rng: random.Random, eval_fn=None, cfg=C):
        self.deck = deck
        self.rng = rng
        self.cfg = cfg
        self.eval_fn = eval_fn or heuristic_evaluate
        self.last_sims = 0
        self.last_fails = 0
        # Opt-in behavioral-cloning prior. Loaded only when POLICY_PUCT_C > 0, so
        # the default agent is byte-for-byte unchanged (plain UCB1, no policy.npz).
        self.puct_c = float(getattr(cfg, "POLICY_PUCT_C", 0.0) or 0.0)
        self.policy = PolicyNet.maybe_load(getattr(cfg, "POLICY_PATH", "")) if self.puct_c > 0 else None

    # ---- rollout ----
    def _rollout(self, state: dict, me: int) -> float:
        sid = state["searchId"]
        obs = state["observation"]
        for _ in range(self.cfg.ROLLOUT_DEPTH):
            cur = obs.get("current")
            if cur and cur.get("result", -1) != -1:
                return terminal_value(cur, me)
            sel = obs.get("select")
            if sel is None:
                break
            choice = policy_choose(obs, rng=self.rng, epsilon=self.cfg.ROLLOUT_EPSILON)
            state = _step_dict(sid, choice)
            sid = state["searchId"]
            obs = state["observation"]
        cur = obs.get("current")
        if cur and cur.get("result", -1) != -1:
            return terminal_value(cur, me)
        return self.eval_fn(cur, me)

    # ---- root bandit ----
    def _ucb_select(self, visits, values, total) -> int:
        for i, v in enumerate(visits):
            if v == 0:
                return i
        logt = math.log(total + 1)
        best, bi = -1e9, 0
        for i in range(len(visits)):
            mean = values[i] / visits[i]
            u = mean + self.cfg.UCB_C * math.sqrt(logt / visits[i])
            if u > best:
                best, bi = u, i
        return bi

    def _policy_priors(self, obs: dict, candidates: list[list[int]]):
        """Prior over `candidates` from the policy net (sums to 1), or None.

        Single-select candidates `[i]` map to option `i`'s probability; a pass
        `[]` (or any out-of-range candidate) gets the mean prior. None on any
        problem so the caller stays on plain UCB1.
        """
        try:
            options = (obs.get("select") or {}).get("option") or []
            if not options:
                return None
            me = obs["current"]["yourIndex"]
            probs = self.policy.priors(obs["current"], me, options)
            if probs is None:
                return None
            out = [float(probs[c[0]]) if (len(c) == 1 and 0 <= c[0] < len(probs)) else None
                   for c in candidates]
            known = [p for p in out if p is not None]
            fill = (sum(known) / len(known)) if known else 1.0
            out = [p if p is not None else fill for p in out]
            s = sum(out)
            return [p / s for p in out] if s > 0 else None
        except Exception:
            return None

    def search(self, obs: dict, candidates: list[list[int]]) -> list[int]:
        """Return the best selection among `candidates` for a single-select decision."""
        me = obs["current"]["yourIndex"]
        agent_obs = to_observation_class(obs)
        n = len(candidates)
        visits = [0] * n
        values = [0.0] * n

        # Policy prior over the candidates (None unless the opt-in policy is loaded
        # and produces a usable prior -> falls back to plain UCB1).
        priors = self._policy_priors(obs, candidates) if self.policy is not None else None

        t_end = time.perf_counter() + self.cfg.MOVE_TIME_BUDGET
        sims = 0
        fails = 0
        try:
            while sims < self.cfg.MAX_SIMULATIONS and time.perf_counter() < t_end:
                ci = (puct_select(visits, values, priors, self.puct_c)
                      if priors is not None else self._ucb_select(visits, values, sims))
                try:
                    det = determinize(obs, self.deck, self.rng)
                    root = api.search_begin(
                        agent_obs, det.your_deck, det.your_prize,
                        det.opponent_deck, det.opponent_prize, det.opponent_hand,
                        det.opponent_active, manual_coin=False,
                    )
                    state = _step_dict(root.searchId, candidates[ci])
                    val = self._rollout(state, me)
                except Exception:
                    fails += 1
                    if fails > 32 and sims == 0:
                        break  # search is unusable for this state; bail to fallback
                    continue
                visits[ci] += 1
                values[ci] += val
                sims += 1
        finally:
            try:
                api.search_end()
            except Exception:
                pass

        self.last_sims = sims
        self.last_fails = fails
        if sims < self.cfg.MIN_SIMULATIONS:
            return None  # signal caller to use the greedy fallback

        # robust choice: best mean among visited arms
        best_i, best_mean, best_v = 0, -1e9, -1
        for i in range(n):
            if visits[i] == 0:
                continue
            mean = values[i] / visits[i]
            if (mean, visits[i]) > (best_mean, best_v):
                best_mean, best_v, best_i = mean, visits[i], i
        return candidates[best_i]
