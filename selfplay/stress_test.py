"""Stress-test a deck against the meta gauntlet.

Runs the chosen agent piloting `--deck` against each deck_meta_*.csv opponent
(seats alternated) and prints win rates. Greedy is fast for a broad read; mcts
gives a truer picture on the matchups that matter.

  python selfplay/stress_test.py --deck deck_crustle.csv --agent greedy -n 100
  python selfplay/stress_test.py --deck deck_crustle.csv --agent mcts   -n 20
"""
import os
import sys
import glob
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from selfplay.baselines import read_deck, GreedyAgent  # noqa: E402
from selfplay.harness import play_match  # noqa: E402


def make(agent, deck, seed):
    if agent == "greedy":
        return GreedyAgent(deck=deck, seed=seed)
    if agent == "mcts":
        from agent.agent import MctsAgent
        return MctsAgent(deck=deck, seed=seed)
    raise ValueError(agent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="deck_crustle.csv")
    ap.add_argument("--agent", default="greedy", choices=["greedy", "mcts"])
    ap.add_argument("--opp-agent", default=None, choices=[None, "greedy", "mcts"])
    ap.add_argument("-n", "--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    opp_agent = args.opp_agent or args.agent

    hero_deck = read_deck(os.path.join(ROOT, args.deck))
    metas = sorted(glob.glob(os.path.join(ROOT, "deck_meta_*.csv")))
    print(f"hero={args.deck} agent={args.agent} vs opp_agent={opp_agent}, n={args.games}/matchup\n")

    overall = []
    for mpath in metas:
        name = os.path.basename(mpath).replace("deck_meta_", "").replace(".csv", "")
        opp_deck = read_deck(mpath)
        hero = make(args.agent, hero_deck, seed=1)
        opp = make(opp_agent, opp_deck, seed=2)
        t0 = time.perf_counter()
        res = play_match(hero, opp, n_games=args.games, alternate=True)
        dt = time.perf_counter() - t0
        wr = res.winrate_a()
        overall.append(wr)
        print(f"  vs {name:<8} : {res.wins_a:>3}-{res.wins_b:<3}-{res.draws:<2} "
              f"winrate={wr:.0%}  ({dt:.0f}s)")
    if overall:
        print(f"\n  gauntlet mean winrate = {sum(overall)/len(overall):.0%}")


if __name__ == "__main__":
    main()
