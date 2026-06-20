"""Merge self-play datasets (greedy data.npz + MCTS chunks) into one for training.

  python selfplay/merge_data.py            # data.npz + data_mcts_*.npz -> data_all.npz
"""
import os
import sys
import glob
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, "selfplay")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data.npz,data_mcts_*.npz")
    ap.add_argument("--out", default=os.path.join(SP, "data_all.npz"))
    args = ap.parse_args()

    files = []
    for pat in args.glob.split(","):
        files += sorted(glob.glob(os.path.join(SP, pat.strip())))
    files = [f for f in dict.fromkeys(files) if os.path.basename(f) != os.path.basename(args.out)]
    if not files:
        raise SystemExit("no datasets found")

    Xs, ys = [], []
    for f in files:
        d = np.load(f)
        if len(d["y"]):
            Xs.append(d["X"]); ys.append(d["y"])
        print(f"  {os.path.basename(f)}: {len(d['y'])} states")
    X = np.concatenate(Xs); y = np.concatenate(ys)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"merged {len(y)} states from {len(files)} files -> {args.out} (mean label={y.mean():.3f})")


if __name__ == "__main__":
    main()
