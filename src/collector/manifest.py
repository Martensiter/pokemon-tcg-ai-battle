"""Episode manifest: which episode ids we've already processed.

Idempotency + resume after crash come from this file. It is append-friendly
(one JSON line per processed episode) so a kill -9 mid-write loses at most the
last line, and reloading rebuilds the seen-set. A small JSON ``summary`` sidecar
holds running counters for at-a-glance status.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional


class Manifest:
    """Tracks processed episode ids and per-status counts.

    Args:
        path: JSON-lines file; each line ``{"episode_id":..., "status":...,
            "rows":..., "ts":...}``.
    """

    STATUSES = ("converted", "empty", "failed", "skipped")

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._seen: set[str] = set()
        self._counts: Counter = Counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line
                eid = str(rec.get("episode_id", ""))
                if eid:
                    self._seen.add(eid)
                    self._counts[rec.get("status", "unknown")] += 1

    def has(self, episode_id: str) -> bool:
        return str(episode_id) in self._seen

    def record(self, episode_id: str, status: str, rows: int = 0,
               extra: Optional[dict[str, Any]] = None) -> None:
        """Append a processing record and mark the episode seen.

        Re-recording an already-seen id is a no-op (keeps idempotency under
        retries / overlapping passes).
        """
        eid = str(episode_id)
        if eid in self._seen:
            return
        import time
        rec: dict[str, Any] = {"episode_id": eid, "status": status, "rows": rows,
                               "ts": int(time.time())}
        if extra:
            rec.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seen.add(eid)
        self._counts[status] += 1

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def write_summary(self, path: str | os.PathLike[str], extra: Optional[dict[str, Any]] = None) -> None:
        """Atomically write a small JSON status summary."""
        summary: dict[str, Any] = {"seen": self.seen_count, "counts": self.counts()}
        if extra:
            summary.update(extra)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
