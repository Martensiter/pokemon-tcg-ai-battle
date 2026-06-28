"""Build a policy-cloning dataset from saved raw replays (top-agent distillation).

The collector saves raw replays (``keep_raw``) of leaderboard-top games under
``<data>/raw/<episode>.json``. This reads those (a dir / glob / files), extracts
``(state, options, chosen)`` from each MAIN single-select decision via
``collector.policy_extract``, and writes one npz the policy trainer consumes:

  python tools/extract_policy.py --src collector_data/raw --out policy_data.npz
  python selfplay/train_policy_np.py --data policy_data.npz --out agent/policy.npz

Numpy-only, no engine / card CSV / torch -- runs anywhere the collector does.
By default keeps only the winner's decisions (``--all-seats`` to keep both).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from collector.policy_extract import PolicyRecords, episode_to_policy_records  # noqa: E402


def find_files(srcs, pattern: str) -> list[str]:
    out: list[str] = []
    for s in srcs:
        if os.path.isdir(s):
            out += glob.glob(os.path.join(s, pattern))
        elif any(c in s for c in "*?["):
            out += glob.glob(s)
        elif os.path.exists(s):
            out.append(s)
    return sorted(dict.fromkeys(os.path.abspath(f) for f in out))


def _payload(blob):
    """Unwrap a ``{"replay": ...}`` shell if a file happens to carry one."""
    if isinstance(blob, dict) and "steps" not in blob and isinstance(blob.get("replay"), (dict, list)):
        return blob["replay"]
    return blob


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", nargs="+",
                    default=[os.path.join(ROOT, "collector_data", "raw")],
                    help="dirs / globs / files of raw replay json")
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "policy_data.npz"))
    ap.add_argument("--winners-only", action="store_true",
                    help="keep only the winner's decisions (default: BOTH seats -- "
                         "top-vs-top losers also play well, so keep ~3.5x more data)")
    args = ap.parse_args(argv)

    files = find_files(args.src, args.glob)
    if not files:
        print(f"no replay files under: {', '.join(args.src)} (pattern {args.glob})")
        return 1

    rec = PolicyRecords()
    counts: Counter = Counter()
    used = 0
    for f in files:
        try:
            blob = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {os.path.basename(f)}: {type(e).__name__}: {e}")
            counts["skip_unreadable_file"] += 1
            continue
        added = episode_to_policy_records(_payload(blob), rec,
                                          winners_only=args.winners_only, counts=counts)
        used += 1 if added else 0

    # What got dropped, and why (debugging the data funnel).
    print(f"replays: {used}/{len(files)} contributed decisions "
          f"(winners_only={args.winners_only})")
    for k in ("kept", "skip_loser_seat", "skip_non_main", "skip_single_option",
              "skip_multi_select", "skip_bad_seat", "skip_bad_state",
              "skip_draw_or_unknown_episode", "skip_unreadable_file"):
        if counts.get(k):
            print(f"  {k:30s} {counts[k]}")

    arr = rec.arrays()
    if len(arr["group"]) == 0:
        print(f"no decisions extracted from {len(files)} file(s)")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **arr)
    print(f"{len(arr['group'])} decisions / {len(arr['opt'])} options -> {args.out}")
    print(f"train with:  python selfplay/train_policy_np.py --data {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
