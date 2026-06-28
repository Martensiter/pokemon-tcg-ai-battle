"""Sweep a search/eval config knob and A/B each value against the baseline.

The value net is fixed to ~0.6s search depth (see agent/config.py), so raising
the time budget loses points. The cheaper lever is the *search shape* at the same
budget -- determinizations, rollout depth, exploration, value-net blend. This
runs head-to-head self-play (variant config vs the current baseline config, same
weights) across the deck pool and reports the variant's win rate, so you can pick
a stronger config WITHOUT collecting new data. Engine machine only (needs cg).

  # does a lighter determinization count play better at the same 0.6s budget?
  python tools/sweep_config.py --param DETERMINIZATIONS_PER_MOVE --values 8,12,16,24 --games 12
  # value-net blend sweep
  python tools/sweep_config.py --param VALUE_NET_WEIGHT --values 0.5,0.6,0.7,0.8 --games 12

A value with win rate >= --threshold beats the baseline. Nothing is promoted
automatically; apply a winning value by editing agent/config.py (or its env var).
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _config_attrs(base) -> dict:
    """The UPPERCASE tunables the agent/mcts/value-net read off the config."""
    return {k: getattr(base, k) for k in dir(base)
            if k.isupper() and not callable(getattr(base, k))}


def _clone_cfg(base, overrides: dict):
    """Copy base config's tunables into a namespace, applying typed overrides.

    Each override value is cast to the type of the base attribute (so CLI strings
    become the right int/float), which keeps the search params well-typed.
    """
    attrs = _config_attrs(base)
    for key, val in overrides.items():
        if key not in attrs:
            raise KeyError(f"unknown config param: {key} (have {sorted(attrs)})")
        cur = attrs[key]
        if isinstance(cur, bool):
            cast = val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            cast = int(val)
        elif isinstance(cur, float):
            cast = float(val)
        else:
            cast = type(cur)(val)
        attrs[key] = cast
    return SimpleNamespace(**attrs)


def run_ab(variant_cfg, base_cfg, decks, games_per_deck: int, seed: int = 0):
    """Play variant-config vs baseline-config (same weights) across decks."""
    from agent.agent import MctsAgent
    from selfplay.harness import play_match

    v_wins = b_wins = draws = 0
    for i, deck in enumerate(decks):
        a = MctsAgent(deck=list(deck), seed=seed + 2 * i, cfg=variant_cfg)
        b = MctsAgent(deck=list(deck), seed=seed + 2 * i + 1, cfg=base_cfg)
        res = play_match(a, b, n_games=games_per_deck, alternate=True)
        v_wins += res.wins_a
        b_wins += res.wins_b
        draws += res.draws
        print(f"  deck {i+1}/{len(decks)}: variant {res.wins_a} - base {res.wins_b} "
              f"(draws {res.draws})", flush=True)
    return v_wins, b_wins, draws


def win_rate(v_wins: int, b_wins: int) -> float:
    decided = v_wins + b_wins
    return (v_wins / decided) if decided else 0.0


def main(argv: list[str] | None = None) -> int:
    from agent import config as C
    from selfplay.gen_data import resolve_decks

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", required=True, help="config knob to sweep, e.g. VALUE_NET_WEIGHT")
    ap.add_argument("--values", required=True, help="comma-separated values to test vs baseline")
    ap.add_argument("--decks", default="pool", help="'pool', 'all', or comma-separated filenames")
    ap.add_argument("--games", type=int, default=12, help="games per deck per value")
    ap.add_argument("--threshold", type=float, default=0.53, help="win rate to beat baseline")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    decks = resolve_decks(args.decks)
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    base_val = _config_attrs(C).get(args.param, "?")
    print(f"sweeping {args.param} (baseline={base_val}) over {values} | "
          f"{len(decks)} decks x {args.games} games, threshold {args.threshold:.0%}")

    results = []
    for val in values:
        variant = _clone_cfg(C, {args.param: val})
        applied = getattr(variant, args.param)
        print(f"\n== {args.param}={applied} vs baseline {base_val} ==")
        v_wins, b_wins, draws = run_ab(variant, C, decks, args.games, args.seed)
        wr = win_rate(v_wins, b_wins)
        verdict = "BETTER" if wr >= args.threshold else ("~tie" if wr >= 0.47 else "worse")
        print(f"  -> {args.param}={applied}: variant {v_wins} - base {b_wins} "
              f"- draws {draws} | win rate {wr:.1%} | {verdict}")
        results.append((applied, wr, v_wins, b_wins, draws))

    print("\n=== summary (vs baseline) ===")
    for applied, wr, v, b, d in sorted(results, key=lambda r: -r[1]):
        print(f"  {args.param}={applied}: {wr:.1%}  ({v}-{b}-{d})")
    best = max(results, key=lambda r: r[1]) if results else None
    if best and best[1] >= args.threshold:
        print(f"\nbest: {args.param}={best[0]} ({best[1]:.1%}). Apply by editing "
              f"agent/config.py, then re-verify with tools/verify_candidate.py.")
    else:
        print("\nno value beat the baseline by the threshold (baseline stays).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
