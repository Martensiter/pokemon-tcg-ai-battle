"""Self-play harness: run matches between two agents and report win-rate.

Drives the live engine via cg.game, routing each decision to the agent whose
seat is to move (`obs.current.yourIndex`). Tracks per-decision latency so we can
keep MCTS within budget. Seats are alternated across games for fairness.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


@dataclass
class MatchResult:
    games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    steps_total: int = 0
    move_times: list = field(default_factory=list)  # per-decision seconds

    def winrate_a(self) -> float:
        decided = self.wins_a + self.wins_b
        return self.wins_a / decided if decided else 0.5

    def summary(self, name_a: str, name_b: str) -> str:
        mt = self.move_times
        mean_ms = 1000 * sum(mt) / len(mt) if mt else 0.0
        max_ms = 1000 * max(mt) if mt else 0.0
        return (
            f"{name_a} vs {name_b}: {self.wins_a}-{self.wins_b}-{self.draws} "
            f"(winrate_a={self.winrate_a():.1%}, n={self.games}) | "
            f"avg_steps={self.steps_total / max(1, self.games):.0f} | "
            f"move_ms mean={mean_ms:.1f} max={max_ms:.1f}"
        )


def play_one(agent0, agent1, time_moves_of: int | None = None, res: MatchResult | None = None) -> int:
    """Play a single game. agent0 is seat 0, agent1 is seat 1. Returns winner idx (0/1/2)."""
    agents = (agent0, agent1)
    obs, start = battle_start(agent0.deck, agent1.deck)
    if obs is None:
        raise RuntimeError(f"battle_start failed: {start.errorPlayer}/{start.errorType}")
    steps = 0
    winner = 2
    try:
        while True:
            state = obs.get("current")
            if state and state.get("result", -1) != -1:
                winner = state["result"]
                break
            sel = obs.get("select")
            if sel is None:
                break
            who = state["yourIndex"]
            t0 = time.perf_counter()
            choice = agents[who](obs)
            dt = time.perf_counter() - t0
            if res is not None and (time_moves_of is None or who == time_moves_of):
                res.move_times.append(dt)
            obs = battle_select(choice)
            steps += 1
            if steps > 30000:
                raise RuntimeError("game did not terminate")
    finally:
        battle_finish()
    if res is not None:
        res.steps_total += steps
    return winner


def play_match(agent0, agent1, n_games: int = 100, alternate: bool = True,
               time_agent: int = 0, verbose: bool = False) -> MatchResult:
    """Play n_games. `agent0`/`agent1` are logical contestants A and B.

    With alternate=True, A and B swap seats every other game. `time_agent` selects
    which contestant's move latency to record (0=A, 1=B, None=both)."""
    res = MatchResult()
    for g in range(n_games):
        a_is_seat0 = (g % 2 == 0) or not alternate
        seat0, seat1 = (agent0, agent1) if a_is_seat0 else (agent1, agent0)
        # record latency for contestant A's seat
        a_seat = 0 if a_is_seat0 else 1
        winner = play_one(seat0, seat1, time_moves_of=a_seat if time_agent == 0 else (1 - a_seat),
                          res=res)
        res.games += 1
        if winner == 2:
            res.draws += 1
        else:
            a_won = (winner == 0 and a_is_seat0) or (winner == 1 and not a_is_seat0)
            if a_won:
                res.wins_a += 1
            else:
                res.wins_b += 1
        if verbose and (g + 1) % max(1, n_games // 10) == 0:
            print(f"  [{g + 1}/{n_games}] {res.wins_a}-{res.wins_b}-{res.draws}")
    return res


def make_agent(spec: str, seed: int):
    """spec in {random, greedy, mcts}."""
    from selfplay.baselines import RandomAgent, GreedyAgent
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec == "greedy":
        return GreedyAgent(seed=seed)
    if spec == "mcts":
        from agent.agent import MctsAgent
        return MctsAgent(seed=seed)
    raise ValueError(f"unknown agent spec: {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="greedy")
    ap.add_argument("--b", default="random")
    ap.add_argument("-n", "--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    agent_a = make_agent(args.a, seed=args.seed)
    agent_b = make_agent(args.b, seed=args.seed + 1000)
    t0 = time.perf_counter()
    res = play_match(agent_a, agent_b, n_games=args.games, verbose=args.verbose)
    dt = time.perf_counter() - t0
    print(res.summary(args.a, args.b))
    print(f"wall time: {dt:.1f}s ({dt / max(1, args.games):.2f}s/game)")


if __name__ == "__main__":
    main()
