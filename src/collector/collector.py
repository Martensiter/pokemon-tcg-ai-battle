"""Collector orchestration: discover -> fetch -> convert -> persist, on a loop.

Strategy (high signal first): find leaderboard-top submissions, list each
submission's episodes, and pull replays we haven't seen. Every replay is streamed
through the converter into compact value-net records; raw JSON is discarded by
default (opt-in via ``keep_raw``). The manifest makes the whole thing idempotent
and crash-resumable -- already-seen episode ids are never re-fetched.

The long loop is ``nohup``-friendly: it logs to stdout + file, survives transient
network errors via backoff, and persists progress continuously so a restart
continues where it left off.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import CollectorConfig
from .convert import ValueRecords, episode_metadata, episode_to_records
from .kaggle_client import KaggleClient
from .logutil import get_logger, log_kv
from .manifest import Manifest
from .parse import parse_episode
from .ratelimit import FatalError, RetryableError
from .sink import Sink, build_sink


def _scan_id(row: dict[str, Any], wanted: tuple[str, ...]) -> Optional[str]:
    """Pull the first id-like value from a CSV row (defensive about columns)."""
    for k, v in row.items():
        kl = str(k).lower().replace(" ", "")
        if any(w in kl for w in wanted) and str(v).strip():
            return str(v).strip()
    return None


@dataclass
class PassStats:
    """Counters for one discovery pass (logged at INFO)."""

    submissions: int = 0
    episodes_listed: int = 0
    fetched: int = 0
    converted_rows: int = 0
    empty: int = 0
    failed: int = 0
    skipped: int = 0
    chunks: int = 0

    def as_kv(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Collector:
    """Wires the pieces together and runs collection passes."""

    def __init__(self, config: CollectorConfig,
                 client: KaggleClient | None = None,
                 sink: Sink | None = None,
                 manifest: Manifest | None = None,
                 logger: logging.Logger | None = None):
        self.cfg = config
        self.log = logger or get_logger("collector", config.state_dir / "collector.log")
        self.client = client or KaggleClient(config, logger=self.log)
        self.sink = sink or build_sink(config, logger=self.log)
        self.manifest = manifest or Manifest(config.state_dir / "manifest.jsonl")
        self._stop = False
        self._buf = ValueRecords()
        self._meta_buf: list[dict[str, Any]] = []
        self._buf_episodes = 0
        self._chunk_seq = 0

    # --- discovery --------------------------------------------------------
    def discover_submission_ids(self) -> list[str]:
        """Leaderboard-top (and optionally targeted) submission ids to mine."""
        rows: list[dict[str, Any]] = []
        try:
            rows = self.client.leaderboard()
        except (RetryableError, FatalError) as e:
            log_kv(self.log, "leaderboard_failed", level=logging.WARNING, err=str(e)[:200])
        targets = [t.lower() for t in self.cfg.target_teams]
        ids: list[str] = []
        seen: set[str] = set()
        for i, row in enumerate(rows):
            if self.cfg.top_n_leaders and i >= self.cfg.top_n_leaders and not targets:
                break
            if targets:
                blob = " ".join(str(v) for v in row.values()).lower()
                if not any(t in blob for t in targets):
                    continue
            sid = _scan_id(row, ("submissionid", "submission", "id"))
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
        # Fall back to our own submissions if leaderboard yielded nothing usable.
        if not ids:
            try:
                for row in self.client.submissions():
                    sid = _scan_id(row, ("submissionid", "id", "ref"))
                    if sid and sid not in seen:
                        seen.add(sid)
                        ids.append(sid)
            except (RetryableError, FatalError) as e:
                log_kv(self.log, "submissions_failed", level=logging.WARNING, err=str(e)[:200])
        log_kv(self.log, "discovered", submissions=len(ids))
        return ids

    def list_episode_ids(self, submission_id: str) -> list[str]:
        try:
            rows = self.client.episodes(submission_id)
        except (RetryableError, FatalError) as e:
            log_kv(self.log, "episodes_failed", level=logging.WARNING,
                   submission=submission_id, err=str(e)[:200])
            return []
        ids: list[str] = []
        seen: set[str] = set()
        for row in rows:
            eid = _scan_id(row, ("episodeid", "episode", "id"))
            eid = eid.strip() if eid else ""
            # Kaggle episode ids are integers; `kaggle competitions replay`
            # rejects anything else. Filter defensively so a stray header/usage
            # line from the CSV never gets passed as an id.
            if eid.isdigit() and eid not in seen:
                seen.add(eid)
                ids.append(eid)
        if self.cfg.episodes_per_submission:
            ids = ids[: self.cfg.episodes_per_submission]
        return ids

    # --- per-episode ------------------------------------------------------
    def process_episode(self, episode_id: str, stats: PassStats) -> None:
        """Fetch, convert, and buffer one episode (idempotent via manifest)."""
        if self.manifest.has(episode_id):
            stats.skipped += 1
            return
        try:
            payload = self.client.replay(episode_id)
        except RetryableError as e:
            # Retries already exhausted inside the client; treat as failed-for-now
            # but do NOT mark seen, so a later pass can retry.
            log_kv(self.log, "replay_retryable", level=logging.WARNING,
                   episode=episode_id, err=str(e)[:160])
            stats.failed += 1
            return
        except FatalError as e:
            log_kv(self.log, "replay_failed", level=logging.WARNING,
                   episode=episode_id, err=str(e)[:160])
            self.manifest.record(episode_id, "failed", extra={"err": str(e)[:160]})
            stats.failed += 1
            return

        stats.fetched += 1
        ep = parse_episode(payload.get("replay"), episode_id=episode_id)
        if not ep.ok:
            self.manifest.record(episode_id, "empty", extra={"err": ep.error})
            stats.empty += 1
            return

        rows = 0
        if self.cfg.save_value_records:
            rows = episode_to_records(ep, self._buf)
        if self.cfg.save_episode_meta:
            self._meta_buf.append(episode_metadata(ep))
        if self.cfg.keep_raw:
            self.sink.write_raw(episode_id, payload.get("replay"))

        self.manifest.record(episode_id, "converted" if rows else "empty", rows=rows)
        stats.converted_rows += rows
        if rows == 0:
            stats.empty += 1
        self._buf_episodes += 1
        if self._buf_episodes >= self.cfg.chunk_size:
            self._flush(stats)

    # --- flushing ---------------------------------------------------------
    def _flush(self, stats: PassStats) -> None:
        """Write the buffered chunk + metadata to the sink and reset buffers."""
        if self._buf_episodes == 0:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"data_collected_{stamp}_{self._chunk_seq:04d}"
        self._chunk_seq += 1
        if self.cfg.save_value_records and len(self._buf):
            X, y = self._buf.arrays()
            path = self.sink.write_value_chunk(name, X, y)
            log_kv(self.log, "chunk_written", path=path, rows=int(len(y)),
                   mean_label=round(float(y.mean()), 4) if len(y) else 0.0)
            stats.chunks += 1
        if self.cfg.save_episode_meta and self._meta_buf:
            self.sink.write_metadata(f"episodes_{stamp}", self._meta_buf)
        self._buf = ValueRecords()
        self._meta_buf = []
        self._buf_episodes = 0

    # --- passes / loop ----------------------------------------------------
    def run_once(self) -> PassStats:
        """One full discovery + collection pass."""
        stats = PassStats()
        sub_ids = self.discover_submission_ids()
        stats.submissions = len(sub_ids)
        for sid in sub_ids:
            if self._stop:
                break
            ep_ids = self.list_episode_ids(sid)
            stats.episodes_listed += len(ep_ids)
            for eid in ep_ids:
                if self._stop:
                    break
                self.process_episode(eid, stats)
        self._flush(stats)
        self._write_status(stats)
        log_kv(self.log, "pass_complete", **stats.as_kv(), seen_total=self.manifest.seen_count)
        return stats

    def run_forever(self) -> None:
        """Long-running loop with graceful shutdown on SIGINT/SIGTERM."""
        self._install_signal_handlers()
        log_kv(self.log, "collector_start", **{k: v for k, v in self.cfg.redacted().items()
                                               if k in ("competition", "sink", "rps", "top_n_leaders",
                                                        "chunk_size", "loop_interval")})
        while not self._stop:
            try:
                stats = self.run_once()
            except Exception as e:  # noqa: BLE001  (never let the loop die)
                log_kv(self.log, "pass_error", level=logging.ERROR, err=f"{type(e).__name__}: {e}")
                self.log.exception("pass crashed")
                stats = None
            if self._stop:
                break
            # Publish to the canonical store ONLY when this pass wrote a new chunk
            # (no-op for LocalSink). Avoids spamming Kaggle Dataset versions every
            # loop when there are no new episodes.
            self._maybe_publish(stats)
            self._interruptible_sleep(self.cfg.loop_interval)
        log_kv(self.log, "collector_stop", seen_total=self.manifest.seen_count)

    def _maybe_publish(self, stats: "PassStats | None") -> bool:
        """Publish to the sink iff the pass produced a new chunk. Never raises."""
        if stats is None or stats.chunks <= 0:
            return False
        try:
            return bool(self.sink.publish(
                f"collector pass {time.strftime('%Y-%m-%d %H:%M:%S')} (+{stats.converted_rows} rows)"))
        except Exception as e:  # noqa: BLE001
            log_kv(self.log, "publish_error", level=logging.WARNING, err=str(e)[:160])
            return False

    # --- helpers ----------------------------------------------------------
    def _write_status(self, stats: PassStats) -> None:
        self.manifest.write_summary(self.cfg.state_dir / "status.json",
                                    extra={"last_pass": stats.as_kv()})

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log_kv(self.log, "signal", signum=signum)
            self._stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not in main thread (e.g. under tests)

    def _interruptible_sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))

    def request_stop(self) -> None:
        self._stop = True
