"""Round-robin deck tournament: rank decks by aggregate win rate.

Plays every pair of decks (seats alternated) with the chosen agent and tallies
each deck's wins across all its games. Greedy is fast for a broad ranking; rerun
the top few with --agent mcts to confirm under real search.

  python selfplay/round_robin.py --agent greedy -n 30
  python selfplay/round_robin.py --decks deck_crustle_v2.csv,deck_cand_psy_latias.csv --agent mcts -n 20
"""
import os
import sys
import time
import argparse
import itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from selfplay.baselines import read_deck, GreedyAgent  # noqa: E402
from selfplay.harness import play_match  # noqa: E402

DEFAULT = [
    "deck.csv", "deck_crustle_v2.csv",
    "deck_cand_dark_yveltal.csv", "deck_cand_psy_latias.csv", "deck_cand_fight_koraidon.csv",
    "deck_meta_dragapult.csv", "deck_meta_lucario.csv", "deck_meta_abomasnow.csv",
    "deck_meta_nonex.csv", "deck_meta_mixed.csv", "deck_meta_fire_ex.csv",
]


def make(kind, deck, seed):
    if kind == "mcts":
        from agent.agent import MctsAgent
        return MctsAgent(deck=deck, seed=seed)
    return GreedyAgent(deck=deck, seed=seed)


def short(name):
    return name.replace("deck_", "").replace("cand_", "").replace("meta_", "").replace(".csv", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default=None, help="comma-separated; default = built-in pool")
    ap.add_argument("--agent", default="greedy", choices=["greedy", "mcts"])
    ap.add_argument("-n", "--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [s.strip() for s in args.decks.split(",")] if args.decks else DEFAULT
    files = [f for f in files if os.path.exists(os.path.join(ROOT, f))]
    decks = {f: read_deck(os.path.join(ROOT, f)) for f in files}

    wins = {f: 0 for f in files}
    games = {f: 0 for f in files}
    matrix = {f: {} for f in files}
    t0 = time.perf_counter()
    for a, b in itertools.combinations(files, 2):
        A = make(args.agent, decks[a], seed=1)
        B = make(args.agent, decks[b], seed=2)
        r = play_match(A, B, n_games=args.games, alternate=True)
        wins[a] += r.wins_a; games[a] += r.wins_a + r.wins_b
        wins[b] += r.wins_b; games[b] += r.wins_a + r.wins_b
        matrix[a][b] = r.winrate_a(); matrix[b][a] = 1 - r.winrate_a()
        print(f"  {short(a):<16} vs {short(b):<16} {r.wins_a:>3}-{r.wins_b:<3} ({r.winrate_a():.0%})")
    dt = time.perf_counter() - t0

    print(f"\n=== RANKING ({args.agent}, n={args.games}/pair, {dt:.0f}s) ===")
    rank = sorted(files, key=lambda f: wins[f] / max(1, games[f]), reverse=True)
    for i, f in enumerate(rank, 1):
        wr = wins[f] / max(1, games[f])
        print(f"  {i:>2}. {short(f):<18} {wr:.1%}  ({wins[f]}/{games[f]})")


if __name__ == "__main__":
    main()
