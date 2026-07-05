"""Fetch Kaggle's official "Daily Top Episodes" datasets for distillation data.

The competition hosts publish a curated dataset of each day's top-rated
episodes (~5-8k replay JSONs, ~21 GB unpacked but only ~750 MB zipped per
day), indexed at ``kaggle/pokemon-tcg-ai-battle-episodes-index`` (competition
discussion #709160). Episodes are selected by highest average participant
rating, i.e. top-competitor games -- roughly 1000x the collector's
leaderboard trickle, and exactly the behavioural-cloning fodder the policy
pipeline (HANDOFF SS5b step 4) has been waiting for.

One day per invocation: download -> unzip -> extract -> optional cleanup.

  python tools/fetch_top_episodes.py --latest --cleanup
  python tools/fetch_top_episodes.py --date 2026-06-30 \
      --value-out value_data_20260630.npz

Downloads go through the official kaggle CLI (legacy datasets auth is fine:
``pip install "kaggle<1.7"``). Extraction shells out to
``tools/extract_policy.py`` (and ``tools/extract_value.py`` when
``--value-out`` is given; note value rows from these episodes may overlap
with Hub-collected chunks for the same episode ids -- keep the sources in
separate npz files and merge deliberately).
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_SLUG = "kaggle/pokemon-tcg-ai-battle-episodes-index"


def read_index(manifest_text: str) -> list[dict]:
    """Parse the index manifest.csv into a list of day rows (oldest first)."""
    rows = []
    for r in csv.DictReader(io.StringIO(manifest_text)):
        rows.append({
            "date": r["date"],
            "slug": r["daily_dataset_slug"],
            "ref": dataset_ref(r.get("daily_dataset_url", ""),
                               r["daily_dataset_slug"]),
            "episode_count": int(r["episode_count"]),
            "total_bytes": int(r["total_bytes"]),
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def dataset_ref(url: str, slug: str) -> str:
    """Derive the ``owner/slug`` dataset ref from the manifest's URL.

    The daily datasets are published by ``kaggle`` today, but the manifest
    carries a full URL (``.../datasets/<owner>/<slug>``) -- trust that over a
    hard-coded owner, falling back to ``kaggle/<slug>``.
    """
    parts = [p for p in url.split("/") if p]
    if "datasets" in parts:
        i = parts.index("datasets")
        if len(parts) >= i + 3:
            return f"{parts[i + 1]}/{parts[i + 2]}"
    return f"kaggle/{slug}"


def pick_day(rows: list[dict], date: str | None = None,
             latest: bool = False) -> dict:
    """Select one day's row by --date or --latest."""
    if not rows:
        raise SystemExit("index manifest is empty")
    if latest:
        return rows[-1]
    if date:
        for r in rows:
            if r["date"] == date:
                return r
        known = ", ".join(r["date"] for r in rows)
        raise SystemExit(f"date {date} not in index (known: {known})")
    raise SystemExit("pass --date YYYY-MM-DD or --latest")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch_index(kaggle_bin: str) -> list[dict]:
    """Download the index dataset and parse its manifest.csv."""
    with tempfile.TemporaryDirectory() as td:
        _run([kaggle_bin, "datasets", "download", "-d", INDEX_SLUG,
              "-p", td, "--unzip"])
        path = os.path.join(td, "manifest.csv")
        return read_index(open(path, encoding="utf-8").read())


def download_day(day: dict, day_dir: str, kaggle_bin: str,
                 force: bool = False) -> int:
    """Download + unzip one day's episode JSONs into ``day_dir``.

    Idempotent: skipped when the directory already holds at least the
    expected number of JSONs (use ``--force`` to re-download).
    """
    have = len(glob.glob(os.path.join(day_dir, "*.json")))
    if 0 < day["episode_count"] <= have and not force:
        print(f"{day['date']}: {have} JSONs already staged, skipping download")
        return have
    os.makedirs(day_dir, exist_ok=True)
    _run([kaggle_bin, "datasets", "download", "-d", day["ref"], "-p", day_dir])
    for z in glob.glob(os.path.join(day_dir, "*.zip")):
        print(f"unzipping {os.path.basename(z)} ...", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(day_dir)
        os.remove(z)
    n = len(glob.glob(os.path.join(day_dir, "*.json")))
    print(f"{day['date']}: staged {n}/{day['episode_count']} episode JSONs")
    if n == 0:
        raise SystemExit(f"{day['date']}: no episode JSONs staged -- "
                         "download or unzip failed")
    if n < day["episode_count"]:
        print(f"WARNING: {day['episode_count'] - n} episodes missing vs the "
              "index manifest; extraction proceeds on the staged subset")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    day_sel = ap.add_mutually_exclusive_group()
    day_sel.add_argument("--date", help="index day to fetch (YYYY-MM-DD)")
    day_sel.add_argument("--latest", action="store_true",
                         help="fetch the most recent day in the index")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "episodes"),
                    help="staging root; JSONs land in <out-root>/<date>/")
    ap.add_argument("--policy-out", default=None,
                    help="policy npz path (default: selfplay/policy_data_<date>.npz)")
    ap.add_argument("--no-policy", action="store_true",
                    help="skip the policy extraction step")
    ap.add_argument("--value-out", default=None,
                    help="also extract value-net (X, y) rows to this npz")
    ap.add_argument("--winners-only", action="store_true",
                    help="forwarded to extract_policy (default keeps both seats)")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the staged JSON dir after successful extraction")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the day is already staged")
    ap.add_argument("--kaggle-bin", default=shutil.which("kaggle") or "kaggle",
                    help="kaggle CLI to use (needs legacy datasets auth)")
    args = ap.parse_args(argv)

    rows = fetch_index(args.kaggle_bin)
    day = pick_day(rows, date=args.date, latest=args.latest)
    print(f"selected {day['date']}: {day['episode_count']} episodes, "
          f"{day['total_bytes'] / 1e9:.1f} GB unpacked")

    day_dir = os.path.join(args.out_root, day["date"])
    download_day(day, day_dir, args.kaggle_bin, force=args.force)

    py = sys.executable
    compact = day["date"].replace("-", "")
    if not args.no_policy:
        policy_out = args.policy_out or os.path.join(
            ROOT, "selfplay", f"policy_data_{compact}.npz")
        cmd = [py, os.path.join(ROOT, "tools", "extract_policy.py"),
               "--src", day_dir, "--out", policy_out]
        if args.winners_only:
            cmd.append("--winners-only")
        _run(cmd)
    if args.value_out:
        _run([py, os.path.join(ROOT, "tools", "extract_value.py"),
              "--src", day_dir, "--out", args.value_out])

    if args.cleanup:
        print(f"cleanup: removing {day_dir}")
        shutil.rmtree(day_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
