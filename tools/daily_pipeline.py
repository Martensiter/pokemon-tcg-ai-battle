"""Daily self-improvement pipeline: merge -> retrain -> publish (numpy-only).

Closes the loop so collaborators get a *fresh candidate model* every day, not just
fresh data. Runs on the aarch64 collector device (no torch, no engine binary):

  1. merge the collector's chunks (collector_data/value/data_collected_*.npz)
  2. retrain the value net in numpy (selfplay/train_value_np)
  3. write the candidate weights + a training report into the upload dir and
     publish a new private Kaggle Dataset version (data + weights together)

This is *scheduled batch* retraining, NOT online learning -- consistent with the
project's collect -> offline retrain -> resubmit design. Strength verification and
competition resubmission stay OFF this device (they need the engine binary) and
remain a human-gated step on an engine machine.

Single-publisher model: run the collector with COLLECTOR_SINK=local and let THIS
job be the only thing that versions the dataset (avoids two writers racing).

  # one-shot
  uv run python tools/daily_pipeline.py --publish
  # cron-independent daily loop (nohup-friendly)
  nohup uv run python tools/daily_pipeline.py --publish --loop --interval 86400 \
      >> pipeline.out 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.merge_collected import find_chunks, merge          # noqa: E402
from selfplay.train_value_np import train_mlp                 # noqa: E402
from collector.config import CollectorConfig                  # noqa: E402
from collector.logutil import get_logger, log_kv              # noqa: E402
from collector.sink import KaggleDatasetSink, LocalSink       # noqa: E402


def run_pipeline(data_dir: str | os.PathLike[str], *, hidden: list[int],
                 epochs: int, min_rows: int, publish: bool, dataset_slug: str,
                 logger=None) -> dict:
    """One pass: merge -> train -> write candidate weights -> (optional) publish.

    Returns a summary dict. Never raises on "not enough data yet" -- it logs and
    returns ``{"trained": False, ...}`` so the daily loop keeps going.
    """
    log = logger or get_logger("pipeline")
    data_dir = Path(data_dir)
    value_dir = data_dir / "value"

    chunks = find_chunks([str(value_dir)], "data_collected_*.npz")
    X, y = merge(chunks, verbose=False)
    if len(y) < min_rows:
        log_kv(log, "pipeline_skip", reason="insufficient_rows", rows=int(len(y)),
               min_rows=min_rows)
        return {"trained": False, "rows": int(len(y))}

    weights, metrics = train_mlp(X, y, hidden, epochs=epochs, verbose=False)

    wdir = data_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    wpath = wdir / "weights_candidate.npz"
    np.savez(wpath, **weights)
    report = {
        "ts": int(time.time()),
        "rows": int(len(y)),
        "chunks": len(chunks),
        "hidden": list(hidden),
        "val_loss": round(metrics["val_loss"], 4),
        "val_acc": round(metrics["val_acc"], 4),
        "mean_label": round(float(y.mean()), 4),
    }
    (wdir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_kv(log, "pipeline_trained", **report, weights=str(wpath))

    published = False
    if publish and dataset_slug:
        sink = KaggleDatasetSink(LocalSink(data_dir), dataset_slug, logger=log)
        msg = (f"daily {time.strftime('%Y-%m-%d %H:%M')}: {len(y)} rows, "
               f"val_acc={report['val_acc']}")
        published = sink.publish(msg)
    elif publish and not dataset_slug:
        log_kv(log, "pipeline_publish_skip", reason="no_dataset_slug")

    return {"trained": True, "published": published, **report}


def main(argv: list[str] | None = None) -> int:
    cfg = CollectorConfig.from_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min-rows", type=int, default=2000,
                    help="skip retraining until at least this many states are collected")
    ap.add_argument("--publish", action="store_true",
                    help="version the Kaggle Dataset with data + candidate weights")
    ap.add_argument("--dataset-slug", default=cfg.dataset_slug)
    ap.add_argument("--loop", action="store_true", help="run forever on --interval")
    ap.add_argument("--interval", type=float, default=86400.0, help="loop period (s)")
    args = ap.parse_args(argv)

    log = get_logger("pipeline", Path(cfg.state_dir) / "pipeline.log")
    log_kv(log, "pipeline_start", data_dir=args.data_dir, publish=args.publish,
           slug=(args.dataset_slug or ""), loop=args.loop, interval=args.interval,
           min_rows=args.min_rows)

    def once():
        try:
            return run_pipeline(args.data_dir, hidden=args.hidden, epochs=args.epochs,
                                min_rows=args.min_rows, publish=args.publish,
                                dataset_slug=args.dataset_slug, logger=log)
        except Exception as e:  # noqa: BLE001  (never let the loop die)
            log_kv(log, "pipeline_error", level=40, err=f"{type(e).__name__}: {e}")
            return {"trained": False, "error": str(e)}

    if not args.loop:
        once()
        return 0

    while True:
        once()
        time.sleep(max(60.0, args.interval))


if __name__ == "__main__":
    sys.exit(main())
