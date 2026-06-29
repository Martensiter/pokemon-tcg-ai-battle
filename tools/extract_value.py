"""Build value-net training data (X, y) at the current FEATURE_DIM from raw replays.

The collector writes value chunks (``data_collected_*.npz``) in real time, but
those are frozen at whichever ``agent.features.FEATURE_DIM`` was live when they
were saved. Every time the feature vector grows (new identity hashes, new
counters), the historical chunks become byte-incompatible -- the trainer would
crash on a dim mismatch. This tool re-extracts the X/y pairs from the raw
replay JSON archive so the trainer can run at the *current* dim.

  python tools/extract_value.py --src ds/raw --out value_data.npz
  python selfplay/train_value_np.py --data value_data.npz --out agent/weights.npz

Pure-numpy + collector parser, no engine / card DB / network -- mirrors the
existing ``tools/extract_policy.py`` shape on the value side.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from collector.parse import parse_episode  # noqa: E402
from collector.convert import ValueRecords, episode_to_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(ROOT, "ds", "raw"),
                    help="directory of raw replay json (default: ds/raw)")
    ap.add_argument("--out", default=os.path.join(ROOT, "value_data.npz"))
    ap.add_argument("--min-size", type=int, default=5000,
                    help="skip raw json files smaller than this bytes "
                         "(filters incomplete games that never started)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.src):
        print(f"no such dir: {args.src}")
        return 1

    rec = ValueRecords()
    groups: list[int] = []   # one episode id per row, for episode-aware train/val split
    counts: Counter = Counter()
    for ep_idx, fname in enumerate(sorted(os.listdir(args.src))):
        path = os.path.join(args.src, fname)
        if not fname.endswith(".json"):
            continue
        if os.path.getsize(path) < args.min_size:
            counts["skip_too_small"] += 1
            continue
        try:
            blob = json.load(open(path, encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            counts["skip_unreadable"] += 1
            continue
        ep = parse_episode(blob, episode_id=fname)
        if not ep.ok:
            counts["skip_parse_failed"] += 1
            continue
        before = len(rec)
        added = episode_to_records(ep, rec)
        if added > 0:
            counts["ok"] += 1
            groups.extend([ep_idx] * (len(rec) - before))    # mark rows with this episode
        else:
            counts["skip_no_value_rows"] += 1               # eg. draw / no MAIN frames

    X, y = rec.arrays()
    if len(y) == 0:
        print(f"no rows extracted from {args.src}")
        for k in ("ok", "skip_too_small", "skip_parse_failed",
                  "skip_no_value_rows", "skip_unreadable"):
            if counts.get(k):
                print(f"  {k:24s} {counts[k]}")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    group_arr = np.asarray(groups, dtype=np.int32)
    np.savez_compressed(args.out, X=X, y=y, group=group_arr)
    print(f"replays: ok={counts['ok']}, skip_small={counts['skip_too_small']}, "
          f"skip_parse={counts['skip_parse_failed']}, "
          f"skip_no_rows={counts['skip_no_value_rows']}, "
          f"skip_unreadable={counts['skip_unreadable']}")
    print(f"rows: {X.shape[0]} | dim: {X.shape[1]} | y_mean: {y.mean():.3f}")
    print(f"saved: {args.out}")
    print(f"train with:  python selfplay/train_value_np.py --data {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
