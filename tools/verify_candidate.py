"""Automated A/B verification: is a candidate value-net stronger than current?

Runs head-to-head self-play (candidate-weights agent vs current-weights agent)
across the deck pool and gates on win rate. This is the human-replaceable step
between "daily pipeline produced candidate weights" and "submit a new agent" --
so the whole improve loop can be automated on a machine that HAS the engine
binary (the Hub can't run this; it has no engine).

  python tools/verify_candidate.py --new candidate.npz --games 60 --threshold 0.53
  python tools/verify_candidate.py --new candidate.npz --promote   # copy to agent/weights.npz if it passes

Exit code 0 = candidate passed (>= threshold), 1 = failed/insufficient. With
--promote, a passing candidate is copied over agent/weights.npz.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def passes(new_wins: int, old_wins: int, threshold: float) -> bool:
    """Gate: candidate passes if its decided win rate is >= threshold.

    Pure (no engine) so it is unit-testable. A tie / zero decided games fails.
    """
    decided = new_wins + old_wins
    if decided == 0:
        return False
    return (new_wins / decided) >= threshold


def _eval_for(weights_path: str):
    """Build a leaf evaluator that uses a specific weights file."""
    from agent import config as C
    from agent.value_net import make_leaf_evaluator
    shim = types.SimpleNamespace(WEIGHTS_PATH=weights_path,
                                 VALUE_NET_WEIGHT=C.VALUE_NET_WEIGHT)
    return make_leaf_evaluator(shim)


def run_ab(new_path: str, old_path: str, decks, games_per_deck: int, seed: int = 0):
    """Play candidate vs current across decks. Returns (new_wins, old_wins, draws)."""
    from agent.agent import MctsAgent
    from selfplay.harness import play_match

    new_eval = _eval_for(new_path)
    old_eval = _eval_for(old_path)
    new_wins = old_wins = draws = 0
    for i, deck in enumerate(decks):
        # Mirror match (same deck both sides) isolates the value-net difference.
        a_new = MctsAgent(deck=list(deck), seed=seed + 2 * i, eval_fn=new_eval)
        a_old = MctsAgent(deck=list(deck), seed=seed + 2 * i + 1, eval_fn=old_eval)
        res = play_match(a_new, a_old, n_games=games_per_deck, alternate=True)
        new_wins += res.wins_a
        old_wins += res.wins_b
        draws += res.draws
        print(f"  deck {i+1}/{len(decks)}: new {res.wins_a} - old {res.wins_b} "
              f"(draws {res.draws})", flush=True)
    return new_wins, old_wins, draws


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, help="candidate weights .npz")
    ap.add_argument("--old", default=os.path.join(ROOT, "agent", "weights.npz"),
                    help="current weights .npz (default: agent/weights.npz)")
    ap.add_argument("--decks", default="pool", help="'pool', 'all', or comma-separated filenames")
    ap.add_argument("--games", type=int, default=8, help="games per deck (head-to-head)")
    ap.add_argument("--threshold", type=float, default=0.53,
                    help="min candidate win rate to pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--promote", action="store_true",
                    help="copy --new over --old if it passes")
    args = ap.parse_args(argv)

    if not os.path.exists(args.new):
        print(f"candidate not found: {args.new}")
        return 1

    from selfplay.gen_data import resolve_decks
    decks = resolve_decks(args.decks)

    print(f"verifying {args.new} vs {args.old} "
          f"({len(decks)} decks x {args.games} games, threshold {args.threshold:.0%})")
    new_wins, old_wins, draws = run_ab(args.new, args.old, decks, args.games, args.seed)
    decided = new_wins + old_wins
    wr = (new_wins / decided) if decided else 0.0
    ok = passes(new_wins, old_wins, args.threshold)
    print(f"RESULT: new {new_wins} - old {old_wins} - draws {draws} | "
          f"candidate win rate {wr:.1%} | {'PASS' if ok else 'FAIL'}")

    if ok and args.promote:
        shutil.copyfile(args.new, args.old)
        print(f"promoted: {args.new} -> {args.old}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
