"""Stage 1 smoke test.

Goals:
  * confirm the engine DLL loads,
  * confirm a full random-vs-random game runs start -> finish,
  * exercise + document the agent option-return contract,
  * print a sample of real observations so we can validate our mental model
    of the SelectContext / Option structures.

Run from the project root:  python tests/smoke_test.py
"""
import os
import sys
import random
import collections

# Make `import cg` work regardless of cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.api import (  # noqa: E402
    all_card_data, all_attack, SelectContext, SelectType, OptionType, LogType,
)


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        ids = [int(line) for line in f.read().split("\n") if line.strip()]
    assert len(ids) == 60, f"deck must be 60 cards, got {len(ids)}"
    return ids


def random_legal_choice(sel: dict) -> list[int]:
    """Return a contract-valid selection for a SelectData dict.

    Contract (from main.py): length in [minCount, maxCount], unique,
    every element in [0, len(option)).
    """
    n = len(sel["option"])
    lo = sel["minCount"]
    hi = min(sel["maxCount"], n)
    if hi <= 0:
        return []
    k = random.randint(lo, hi) if hi >= lo else lo
    k = max(0, min(k, n))
    return random.sample(range(n), k)


def play_random_game(deck0: list[int], deck1: list[int], verbose_first: int = 12):
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} errorType={start.errorType}"
        )

    ctx_counter = collections.Counter()
    seltype_counter = collections.Counter()
    steps = 0
    printed = 0
    result = -1

    while True:
        state = obs.get("current")
        if state is not None and state.get("result", -1) != -1:
            result = state["result"]
            break

        sel = obs.get("select")
        if sel is None:
            # No decision to make and game not flagged finished: nothing to do.
            break

        ctx = sel["context"]
        ctx_counter[ctx] += 1
        seltype_counter[sel["type"]] += 1

        if printed < verbose_first:
            who = state["yourIndex"] if state else "?"
            opt_types = collections.Counter(o["type"] for o in sel["option"])
            opt_types_named = {OptionType(t).name: c for t, c in opt_types.items()}
            print(
                f"[step {steps:>3}] player={who} turn={state['turn'] if state else '?'} "
                f"ctx={SelectContext(ctx).name} selType={SelectType(sel['type']).name} "
                f"min={sel['minCount']} max={sel['maxCount']} nOpt={len(sel['option'])} "
                f"opts={opt_types_named}"
            )
            printed += 1

        choice = random_legal_choice(sel)
        obs = battle_select(choice)
        steps += 1
        if steps > 20000:
            raise RuntimeError("game did not terminate within 20000 steps")

    battle_finish()
    return result, steps, ctx_counter, seltype_counter


def main():
    cards = all_card_data()
    attacks = all_attack()
    print(f"engine loaded: {len(cards)} cards, {len(attacks)} attacks")

    deck_path = os.path.join(ROOT, "deck_sample.csv")
    deck = read_deck(deck_path)
    print(f"sample deck loaded: {len(deck)} cards, unique ids={len(set(deck))}")

    wins = collections.Counter()
    total_steps = 0
    n_games = 5
    agg_ctx = collections.Counter()
    for g in range(n_games):
        result, steps, ctx_counter, _ = play_random_game(deck, deck, verbose_first=12 if g == 0 else 0)
        wins[result] += 1
        total_steps += steps
        agg_ctx.update(ctx_counter)
        print(f"game {g}: result={result} (winner index, 2=draw) steps={steps}")

    print(f"\n{n_games} games OK. wins(by player idx)={dict(wins)} avg_steps={total_steps/n_games:.0f}")
    print("decision-context frequency (random play):")
    for ctx, c in agg_ctx.most_common():
        print(f"   {SelectContext(ctx).name:<28} {c}")


if __name__ == "__main__":
    main()
