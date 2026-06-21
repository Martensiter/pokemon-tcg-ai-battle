"""Collector entry point.

  uv run python -m collector --once          # single pass (CI / smoke)
  uv run python -m collector                 # long-running loop
  nohup uv run python -m collector >> collector.out 2>&1 &   # ARM daemon

Configuration is via environment variables / ``.env`` (see ``.env.example`` and
:class:`collector.config.CollectorConfig`). CLI flags override a few common knobs
for convenience.
"""
from __future__ import annotations

import argparse
import sys

from .collector import Collector
from .config import CollectorConfig
from .logutil import get_logger, log_kv


def build_config(args: argparse.Namespace) -> CollectorConfig:
    cfg = CollectorConfig.from_env()
    if args.rps is not None:
        cfg.rps = args.rps
    if args.top_n is not None:
        cfg.top_n_leaders = args.top_n
    if args.sink is not None:
        cfg.sink = args.sink
    if args.keep_raw:
        cfg.keep_raw = True
    if args.targets:
        cfg.target_teams = [t.strip() for t in args.targets.split(",") if t.strip()]
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pokemon TCG AI Battle replay collector")
    ap.add_argument("--once", action="store_true", help="run a single pass and exit")
    ap.add_argument("--rps", type=float, default=None, help="requests/sec cap for replay fetches")
    ap.add_argument("--top-n", type=int, default=None, help="leaderboard rows to target")
    ap.add_argument("--targets", default=None, help="comma-separated team/agent name substrings")
    ap.add_argument("--sink", choices=["local", "kaggle"], default=None)
    ap.add_argument("--keep-raw", action="store_true", help="also persist raw replay JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="print resolved config and exit (no network)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the full pipeline offline on synthetic data (UAT; no creds/network)")
    args = ap.parse_args(argv)

    if args.self_test:
        from .selftest import run_selftest
        log = get_logger("collector")
        try:
            summary = run_selftest()
        except AssertionError as e:
            log_kv(log, "self_test_failed", level=40, err=str(e))
            return 1
        log_kv(log, "self_test_ok", **summary)
        return 0

    cfg = build_config(args)
    log = get_logger("collector", cfg.state_dir / "collector.log")

    if args.dry_run:
        log_kv(log, "dry_run", **cfg.redacted())
        return 0

    if not cfg.has_credentials():
        log_kv(log, "no_credentials",
               hint="set KAGGLE_USERNAME / KAGGLE_KEY in env or .env; running anyway "
                    "(will fail on real calls)")

    collector = Collector(cfg, logger=log)
    if args.once:
        collector.run_once()
    else:
        collector.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
