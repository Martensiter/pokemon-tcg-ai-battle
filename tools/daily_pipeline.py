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
import shutil
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


def _stage_state(data_dir: Path, state_dir: str | os.PathLike[str] | None,
                 log) -> bool:
    """Copy the collector's manifest/status into the upload dir for durability.

    The manifest (``state/manifest.jsonl``) is the *only* record of which
    episodes we've already processed; idempotency + crash-resume both rest on
    it. It normally lives in ``state_dir``, OUTSIDE the published ``data_dir``,
    so a totalled device would lose it -- and a fresh collector, re-fetching the
    same episodes, would emit duplicate rows into the dataset. Copying it into
    ``data_dir/state/`` makes it ride along in every Kaggle Dataset version, so
    recovery is: download the dataset, restore ``state/manifest.jsonl`` to the
    collector's ``state_dir``, and the collector skips everything already seen.
    See ``docs/HANDOFF.md`` §4a (disaster recovery). Returns True if staged.
    """
    if not state_dir:
        return False
    src_dir = Path(state_dir)
    dst_dir = data_dir / "state"
    staged = False
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.jsonl", "status.json"):
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)
                staged = staged or name == "manifest.jsonl"
    except Exception as e:  # noqa: BLE001  (durability is best-effort; never block publish)
        log_kv(log, "stage_state_failed", level=30, err=f"{type(e).__name__}: {e}")
        return False
    if staged:
        log_kv(log, "stage_state", manifest=str(dst_dir / "manifest.jsonl"))
    return staged


def _value_xy(data_dir: Path, log):
    """Value (X, y) at the CURRENT FEATURE_DIM, plus (source, n_sources).

    Prefer re-extracting from the durable raw replay archive. The incremental
    ``data_collected_*.npz`` chunks are frozen at whatever ``FEATURE_DIM`` was
    live when written, so after a feature change (e.g. 32 -> 124) merge()'s dim
    guard silently DROPS them and the trainer would shrink to only new data.
    Re-extracting from ``raw/`` keeps ALL history at the live dim (raw is small
    and kept via COLLECTOR_KEEP_RAW). Fall back to the chunks if no raw is kept.
    """
    raw_dir = data_dir / "raw"
    raws = sorted(raw_dir.glob("*.json")) if raw_dir.is_dir() else []
    if raws:
        from collector.parse import parse_episode               # noqa: E402
        from collector.convert import ValueRecords, episode_to_records  # noqa: E402
        rec = ValueRecords()
        used = 0
        for f in raws:
            try:
                blob = json.load(open(f, encoding="utf-8"))
            except Exception:  # noqa: BLE001  (skip an unreadable replay)
                continue
            ep = parse_episode(blob, episode_id=f.stem)
            if ep.ok and episode_to_records(ep, rec):
                used += 1
        X, y = rec.arrays()
        if len(y):
            log_kv(log, "value_source", src="raw", replays=len(raws), used=used,
                   rows=int(len(y)), dim=int(X.shape[1]))
            return X, y, "raw", used
    chunks = find_chunks([str(data_dir / "value")], "data_collected_*.npz")
    X, y = merge(chunks, verbose=False)
    log_kv(log, "value_source", src="chunks", chunks=len(chunks), rows=int(len(y)))
    return X, y, "chunks", len(chunks)


def run_pipeline(data_dir: str | os.PathLike[str], *, hidden: list[int],
                 epochs: int, min_rows: int, publish: bool, dataset_slug: str,
                 state_dir: str | os.PathLike[str] | None = None,
                 logger=None) -> dict:
    """One pass: merge -> train -> write candidate weights -> (optional) publish.

    Returns a summary dict. Never raises on "not enough data yet" -- it logs and
    returns ``{"trained": False, ...}`` so the daily loop keeps going.

    ``state_dir`` (the collector's manifest dir) is staged into the upload dir
    before publishing so the manifest is captured in the Kaggle Dataset version
    -- this is what makes a wiped device recover without duplicate-collecting.
    """
    log = logger or get_logger("pipeline")
    data_dir = Path(data_dir)

    X, y, value_src, n_sources = _value_xy(data_dir, log)
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
        "value_src": value_src,
        "sources": n_sources,
        "hidden": list(hidden),
        "val_loss": round(metrics["val_loss"], 4),
        "val_acc": round(metrics["val_acc"], 4),
        "mean_label": round(float(y.mean()), 4),
    }
    (wdir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_kv(log, "pipeline_trained", **report, weights=str(wpath))

    published = False
    if publish and dataset_slug:
        # Capture the manifest in the dataset version (disaster recovery).
        _stage_state(data_dir, state_dir, log)
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
                                dataset_slug=args.dataset_slug,
                                state_dir=cfg.state_dir, logger=log)
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
