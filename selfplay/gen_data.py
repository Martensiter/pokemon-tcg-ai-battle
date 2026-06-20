"""Generate (features, outcome) training data via self-play across many decks.

For each visited MAIN decision we record the feature vector from the to-move
player's perspective and, at game end, label = 1.0 if that player won, 0.0 lost,
0.5 draw. Crucially, each game samples a *random matchup* from a diverse deck pool
(ex aggro, non-ex toolbox, the Crustle wall, evolution decks, the official
samples) so the value net learns to judge any board state instead of overfitting
one deck. Greedy self-play (~0.02-0.05 s/game) yields tens of thousands of labels
quickly; the resulting net upgrades the MCTS leaf evaluation.
"""
from __future__ import annotations

import os
import sys
import time
import random
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from agent.features import extract, FEATURE_DIM  # noqa: E402
from agent.policy import choose as greedy_choose  # noqa: E402
from selfplay.baselines import read_deck, random_legal  # noqa: E402

# Diverse default pool: covers ex / non-ex / wall / evolution / official samples.
DEFAULT_POOL = [
    "deck.csv", "deck_crustle_v2.csv",
    "deck_meta_dragapult.csv", "deck_meta_lucario.csv", "deck_meta_abomasnow.csv",
    "deck_meta_nonex.csv", "deck_meta_mixed.csv", "deck_meta_fire_ex.csv",
    "deck_meta_alakazam.csv", "deck_meta_rocket_spidops.csv",
]


def gen(n_games: int, decks: list[list[int]], seed: int = 0,
        sample_prob: float = 0.5, epsilon: float = 0.12, agent: str = "greedy"):
    rng = random.Random(seed)
    mcts_eval = None
    if agent == "mcts":
        from agent.value_net import make_leaf_evaluator  # share one net load across all games
        mcts_eval = make_leaf_evaluator()
    X: list = []
    rows: list[tuple[int, int]] = []   # (feature_index, owning_player)
    labels: dict[int, float] = {}
    for g in range(n_games):
        if g and g % 25 == 0:
            print(f"  ...{g}/{n_games} games, {len(X)} states", flush=True)
        try:
            d0 = rng.choice(decks)
            d1 = rng.choice(decks)
            obs, start = battle_start(list(d0), list(d1))
            if obs is None:
                continue
            seat = None
            if agent == "mcts":
                from agent.agent import MctsAgent
                seat = [MctsAgent(deck=d0, seed=rng.randrange(1 << 30), eval_fn=mcts_eval),
                        MctsAgent(deck=d1, seed=rng.randrange(1 << 30), eval_fn=mcts_eval)]
            rows_this: list[tuple[int, int]] = []
            steps = 0
            winner = 2
            while True:
                st = obs.get("current")
                if st and st.get("result", -1) != -1:
                    winner = st["result"]
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                me = st["yourIndex"]
                if sel["context"] == 0 and rng.random() < sample_prob:  # MAIN decisions
                    X.append(extract(st, me))
                    rows_this.append((len(X) - 1, me))
                if seat is not None:
                    choice = seat[me].decide(obs)
                else:
                    choice = random_legal(sel, rng) if rng.random() < epsilon else greedy_choose(obs, rng=rng)
                obs = battle_select(choice)
                steps += 1
                if steps > 30000:
                    break
            battle_finish()
            for idx, pl in rows_this:
                labels[idx] = 0.5 if winner == 2 else (1.0 if winner == pl else 0.0)
        except Exception as e:
            print(f"  game {g} skipped: {type(e).__name__}: {e}", flush=True)
            try:
                battle_finish()
            except Exception:
                pass
    if not X:
        return np.zeros((0, FEATURE_DIM), np.float32), np.zeros((0,), np.float32)
    Xa = np.stack(X).astype(np.float32)
    y = np.array([labels[i] for i in range(len(X))], dtype=np.float32)
    return Xa, y


def resolve_decks(spec: str) -> list[list[int]]:
    import glob
    if spec == "all":
        files = [os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "deck*.csv"))]
        files = [f for f in files if f != "deck_sample.csv"]
    elif spec == "pool":
        files = DEFAULT_POOL
    else:
        files = [s.strip() for s in spec.split(",") if s.strip()]
    decks = []
    for f in files:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            decks.append(read_deck(p))
    if not decks:
        raise SystemExit(f"no decks resolved from '{spec}'")
    print(f"deck pool ({len(decks)}): {', '.join(files)}")
    return decks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=6000)
    ap.add_argument("--decks", default="pool", help="'pool', 'all', or comma-separated filenames")
    ap.add_argument("--agent", default="greedy", choices=["greedy", "mcts"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "selfplay", "data.npz"))
    args = ap.parse_args()

    decks = resolve_decks(args.decks)
    t0 = time.perf_counter()
    X, y = gen(args.games, decks, seed=args.seed, agent=args.agent)
    dt = time.perf_counter() - t0
    np.savez_compressed(args.out, X=X, y=y)
    print(f"generated {len(y)} states from {args.games} games in {dt:.1f}s "
          f"(mean label={y.mean():.3f}); saved {args.out}")


if __name__ == "__main__":
    main()
