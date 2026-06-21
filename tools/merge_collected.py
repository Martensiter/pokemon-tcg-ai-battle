"""Bridge: merge collector chunks into one dataset the value-net trainer reads.

The collector writes value-net chunks to ``collector_data/value/data_collected_*.npz``
(arrays ``X (N, FEATURE_DIM)`` + ``y (N,)`` -- the same layout ``selfplay/gen_data.py``
emits). ``selfplay/merge_data.py`` only globs inside ``selfplay/``, so this helper
merges the collector's output (optionally together with existing self-play
datasets) into a single ``.npz`` that ``selfplay/train_value.py --data`` consumes
unchanged.

  # on the offline training box, after pulling the Kaggle Dataset / local chunks:
  python tools/merge_collected.py --src collector_data/value --out selfplay/data_collected_all.npz
  python tools/train_value.py --data selfplay/data_collected_all.npz   # (selfplay/train_value.py)

Pure numpy -- no torch, no engine binary -- so it runs anywhere the collector does.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Iterable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    # Authoritative feature width for value-net data (numpy-only import).
    from agent.features import FEATURE_DIM as _DEFAULT_DIM
except Exception:  # noqa: BLE001  (fall back to first chunk's width)
    _DEFAULT_DIM = None


def find_chunks(srcs: Iterable[str], pattern: str) -> list[str]:
    """Resolve chunk files from a list of dirs/globs/files (dedup, sorted).

    Each ``src`` may be a directory (``<src>/<pattern>`` is globbed), a glob
    pattern, or a single ``.npz`` file.
    """
    files: list[str] = []
    for src in srcs:
        if os.path.isdir(src):
            files += glob.glob(os.path.join(src, pattern))
        elif any(ch in src for ch in "*?[") or not src.endswith(".npz"):
            files += glob.glob(src)
        elif os.path.exists(src):
            files.append(src)
    # stable dedup
    return sorted(dict.fromkeys(os.path.abspath(f) for f in files))


def merge(files: list[str], verbose: bool = True,
          expected_dim: int | None = _DEFAULT_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ``X``/``y`` across chunks, validating a consistent feature dim.

    The canonical width is ``expected_dim`` (the value net's ``FEATURE_DIM`` by
    default); chunks of any other width are skipped rather than poisoning the
    merge. If ``expected_dim`` is None it is taken from the first valid chunk.
    Empty/malformed chunks are skipped with a warning so one bad file never sinks
    a long collection run.
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dim: int | None = expected_dim
    for f in files:
        try:
            d = np.load(f)
            X, y = d["X"], d["y"]
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [skip] {os.path.basename(f)}: {type(e).__name__}: {e}")
            continue
        if len(y) == 0:
            continue
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            if verbose:
                print(f"  [skip] {os.path.basename(f)}: bad shapes X={X.shape} y={y.shape}")
            continue
        if dim is None:
            dim = X.shape[1]
        elif X.shape[1] != dim:
            if verbose:
                print(f"  [skip] {os.path.basename(f)}: feature dim {X.shape[1]} != {dim}")
            continue
        Xs.append(X.astype(np.float32))
        ys.append(y.astype(np.float32))
        if verbose:
            print(f"  {os.path.basename(f)}: {len(y)} states")
    if not Xs:
        return np.zeros((0, dim or 0), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(Xs), np.concatenate(ys)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", nargs="+",
                    default=[os.path.join(ROOT, "collector_data", "value")],
                    help="dirs / globs / files of collector chunks (and optional self-play npz)")
    ap.add_argument("--glob", default="data_collected_*.npz",
                    help="pattern used when a --src entry is a directory")
    ap.add_argument("--out", default=os.path.join(ROOT, "selfplay", "data_collected_all.npz"))
    args = ap.parse_args(argv)

    files = find_chunks(args.src, args.glob)
    if not files:
        print(f"no chunks found under: {', '.join(args.src)} (pattern {args.glob})")
        return 1
    print(f"merging {len(files)} chunk(s) ...")
    X, y = merge(files)
    if len(y) == 0:
        print("no usable states found")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"merged {len(y)} states (dim={X.shape[1]}, mean label={y.mean():.3f}) -> {args.out}")
    print(f"train with:  python selfplay/train_value.py --data {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
