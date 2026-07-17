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

# Optional fast JSON: raw state parsing is ~40% of search wall-time. orjson is
# vendored into the submission bundle (dual-ABI .so, see make_submission
# --extra-dir); stdlib json is the always-works fallback, so a missing/broken
# wheel costs speed, never correctness.
try:
    import orjson as _fastjson
    _loads = _fastjson.loads
except ImportError:  # pragma: no cover - environment-dependent
    _fastjson = None
    _loads = json.loads

from cg import api
from cg.sim import lib
from cg.api import SelectContext

from . import config as C
from .determinize import determinize
from .policy import choose as policy_choose, option_scores
from .evaluate import evaluate as heuristic_evaluate
from .policy_net import PolicyNet, puct_select  # numpy-only; opt-in policy prior


def _step_dict(search_id: int, select: list[int]) -> dict:
    """Advance a search state, returning the raw dict (skips dataclass parsing)."""
    arr = (ctypes.c_int * len(select))(*select)
    js = lib.SearchStep(api.agent_ptr, search_id, arr, len(select))
    d = _loads(js)
    if d["error"] != 0:
        raise RuntimeError(f"search_step error {d['error']}")
    return d["state"]  # {"observation": {...}, "searchId": int}


def _begin_lean(obs: dict, det) -> int:
    """SearchBegin without the dataclass round-trip.

    api.search_begin() converts the full root state into nested dataclasses
    (~14s/game profiled) but the search loop only ever uses ``searchId``.
    Same engine call, same error semantics (raises -> caught per-sim), raw
    dict parsing only. The engine agent pointer is created lazily exactly as
    the api module does.
    """
    if not hasattr(api, "agent_ptr"):
        api.agent_ptr = lib.AgentStart()
    sbi = obs.get("search_begin_input")
    if not sbi:
        raise RuntimeError("no search_begin_input on observation")
    state = obs["current"]
    yi = state["yourIndex"]
    your_deck = [] if (obs.get("select") or {}).get("deck") is not None else det.your_deck
    opp_active_list = (state["players"][1 - yi].get("active") or [])
    opp_active = det.opponent_active if (opp_active_list and opp_active_list[0] is None) else []
    bs = lib.SearchBegin(
        api.agent_ptr, sbi.encode("ascii"), len(sbi),
        (ctypes.c_int * len(your_deck))(*your_deck),
        (ctypes.c_int * len(det.your_prize))(*det.your_prize),
        (ctypes.c_int * len(det.opponent_deck))(*det.opponent_deck),
        (ctypes.c_int * len(det.opponent_prize))(*det.opponent_prize),
        (ctypes.c_int * len(det.opponent_hand))(*det.opponent_hand),
        (ctypes.c_int * len(opp_active))(*opp_active),
        0)
    d = _loads(bs)
    if d["error"] != 0:
        raise RuntimeError(f"search_begin error {d['error']}")
    return d["state"]["searchId"]


def terminal_value(state: dict, me: int) -> float:
    res = state.get("result", -1)
    if res == 2:
        return 0.0
    return 1.0 if res == me else -1.0


def softmax_floor(raw: list[float], temp: float, floor: float) -> list[float]:
    """Softmax(raw/temp) blended with a uniform floor; sums to 1.

    The floor keeps every arm explorable even when the heuristic is confidently
    wrong (the policy-net lesson: a bad sharp prior actively hurts search).
    Pure function so it is unit-testable without the engine.
    """
    n = len(raw)
    if n == 0:
        return []
    t = max(temp, 1e-6)
    m = max(raw)
    exps = [math.exp((r - m) / t) for r in raw]
    s = sum(exps)
    fl = min(max(floor, 0.0), 1.0)
    if s <= 0:
        return [1.0 / n] * n
    return [(1.0 - fl) * e / s + fl / n for e in exps]


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
        # Opt-in heuristic prior (L3): softmax over policy.option_scores(). Also
        # OFF by default; when both priors are enabled the policy net wins and
        # the heuristic serves as its fallback.
        self.heur_c = float(getattr(cfg, "HEUR_PRIOR_C", 0.0) or 0.0)
        # Opt-in rollout policy: the BC net as the PLAYOUT policy (distinct from
        # the root prior). OFF by default -> greedy playout, unchanged baseline.
        self.rollout_c = float(getattr(cfg, "ROLLOUT_POLICY_C", 0.0) or 0.0)
        self.rollout_policy = (PolicyNet.maybe_load(getattr(cfg, "POLICY_PATH", ""))
                               if self.rollout_c > 0 else None)
        # Opt-in archetype-aware opponent prior for determinization. 0 = OFF ->
        # stock mirror prior, byte-identical (agent/determinize.py).
        self.arch_prior = float(getattr(cfg, "ARCH_PRIOR", 0.0) or 0.0)

    # ---- rollout ----
    def _rollout_choose(self, obs: dict) -> list[int]:
        """Playout move: policy net for single-select MAIN, greedy otherwise.

        The net only covers single-select MAIN decisions (its training domain);
        everything else (multi-select, sub-decisions) falls back to the greedy
        playout policy. Epsilon still injects exploration so rollouts don't
        collapse to one deterministic line.
        """
        try:
            sel = obs.get("select") or {}
            options = sel.get("option") or []
            if (sel.get("context") == SelectContext.MAIN.value
                    and sel.get("maxCount") == 1 and len(options) >= 2
                    and self.rng.random() >= self.cfg.ROLLOUT_EPSILON):
                cur = obs.get("current") or {}
                me = cur.get("yourIndex")
                probs = self.rollout_policy.priors(cur, me, options)
                if probs is not None and len(probs) == len(options):
                    return [max(range(len(probs)), key=lambda i: probs[i])]
        except Exception:
            pass
        return policy_choose(obs, rng=self.rng, epsilon=self.cfg.ROLLOUT_EPSILON)

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
            choice = (self._rollout_choose(obs) if self.rollout_policy is not None
                      else policy_choose(obs, rng=self.rng, epsilon=self.cfg.ROLLOUT_EPSILON))
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

    def _heuristic_priors(self, obs: dict, candidates: list[list[int]]):
        """Prior over `candidates` from the fast option scorer (L3), or None.

        Single-select candidates `[i]` take option i's score; a pass `[]` scores
        like an END option. Scores go through a tempered softmax with a uniform
        floor (see softmax_floor). None on any problem -> plain UCB1.
        """
        try:
            scores = option_scores(obs)
            if not scores:
                return None
            raw = [float(scores[c[0]]) if (len(c) == 1 and 0 <= c[0] < len(scores))
                   else 3.0  # pass: same baseline as an explicit END option
                   for c in candidates]
            return softmax_floor(raw,
                                 float(getattr(self.cfg, "HEUR_PRIOR_TEMP", 6.0)),
                                 float(getattr(self.cfg, "HEUR_PRIOR_FLOOR", 0.15)))
        except Exception:
            return None

    def search(self, obs: dict, candidates: list[list[int]]) -> list[int]:
        """Return the best selection among `candidates` for a single-select decision."""
        me = obs["current"]["yourIndex"]
        n = len(candidates)
        visits = [0] * n
        values = [0.0] * n

        # Root prior over the candidates: policy net first (when loaded), then the
        # heuristic option scorer (when enabled); both unusable -> plain UCB1.
        priors, prior_c = None, 0.0
        if self.policy is not None:
            priors, prior_c = self._policy_priors(obs, candidates), self.puct_c
        if priors is None and self.heur_c > 0:
            priors, prior_c = self._heuristic_priors(obs, candidates), self.heur_c

        t_end = time.perf_counter() + self.cfg.MOVE_TIME_BUDGET
        sims = 0
        fails = 0
        try:
            while sims < self.cfg.MAX_SIMULATIONS and time.perf_counter() < t_end:
                ci = (puct_select(visits, values, priors, prior_c)
                      if priors is not None else self._ucb_select(visits, values, sims))
                try:
                    det = determinize(obs, self.deck, self.rng, arch_prior=self.arch_prior)
                    sid = _begin_lean(obs, det)
                    state = _step_dict(sid, candidates[ci])
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
