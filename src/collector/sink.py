"""Storage sinks: where converted records land.

Abstraction so the canonical store (a private Kaggle Dataset) can later be swapped
for GCS without touching the collector. Three implementations:

  * :class:`Sink` -- abstract interface.
  * :class:`LocalSink` -- writes the npz chunks / metadata / (optional) raw to a
    local directory tree. This is what the *existing* offline training reads
    (point ``selfplay/merge_data.py --glob`` at it).
  * :class:`KaggleDatasetSink` -- wraps a LocalSink, then publishes versions via
    ``kaggle datasets version``. Many small files are zipped first (Kaggle caps a
    single file near 2 GB; lots of tiny files are slow to sync).

Chunk files are named ``data_collected_*.npz`` so the existing merge step picks
them up with ``--glob "data_collected_*.npz"``.
"""
from __future__ import annotations

import json
import os
import subprocess
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


class Sink(ABC):
    """Destination for converted training data and metadata."""

    @abstractmethod
    def write_value_chunk(self, name: str, X: np.ndarray, y: np.ndarray) -> str:
        """Persist a value-net training chunk; return its identifier/path."""

    @abstractmethod
    def write_metadata(self, name: str, records: list[dict[str, Any]]) -> str:
        """Persist per-episode metadata (JSON lines); return its path."""

    def write_raw(self, name: str, blob: Any) -> Optional[str]:
        """Optionally persist a raw replay (opt-in). Default: no-op."""
        return None

    def publish(self, message: str) -> bool:
        """Push accumulated data to the canonical store. Default: no-op."""
        return False


class LocalSink(Sink):
    """Write chunks/metadata/raw under a local directory.

    Layout::

        <root>/value/data_collected_*.npz
        <root>/meta/episodes_*.jsonl
        <root>/raw/<episode>.json          (only if keep_raw)
    """

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.value_dir = self.root / "value"
        self.meta_dir = self.root / "meta"
        self.raw_dir = self.root / "raw"
        for d in (self.value_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)

    def write_value_chunk(self, name: str, X: np.ndarray, y: np.ndarray) -> str:
        path = self.value_dir / f"{name}.npz"
        # np.savez_compressed matches gen_data.py / merge_data.py output exactly.
        np.savez_compressed(path, X=X, y=y)
        return str(path)

    def write_metadata(self, name: str, records: list[dict[str, Any]]) -> str:
        path = self.meta_dir / f"{name}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(path)

    def write_raw(self, name: str, blob: Any) -> Optional[str]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{name}.json"
        path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def zip_value_chunks(self, out_name: str = "value_chunks.zip") -> Optional[str]:
        """Bundle all value chunks into one zip (for upload). None if empty."""
        chunks = sorted(self.value_dir.glob("data_collected_*.npz"))
        if not chunks:
            return None
        out = self.root / out_name
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
            for c in chunks:
                zf.write(c, arcname=c.name)  # already-compressed npz; store-only
        return str(out)


class KaggleDatasetSink(Sink):
    """Persist locally, then publish new versions to a private Kaggle Dataset.

    The dataset is the canonical store: private, shared with collaborators, and
    reachable from Kaggle notebooks with zero egress. ``publish`` zips the small
    chunks and runs ``kaggle datasets version``.

    Args:
        local: backing :class:`LocalSink`.
        dataset_slug: ``owner/dataset-name``.
        runner: subprocess seam (tests). ``runner(args, timeout)`` ->
            ``(code, stdout, stderr)``.
        timeout: per-CLI-call timeout.
    """

    def __init__(self, local: LocalSink, dataset_slug: str,
                 runner: Callable[[list[str], float], tuple[int, str, str]] | None = None,
                 timeout: float = 600.0,
                 logger: Any | None = None):
        self.local = local
        self.dataset_slug = dataset_slug
        self._runner = runner or self._default_runner
        self.timeout = timeout
        self._log = logger
        self._ensure_metadata()

    def _default_runner(self, args: list[str], timeout: float) -> tuple[int, str, str]:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    def _ensure_metadata(self) -> None:
        """Ensure a ``dataset-metadata.json`` exists in the upload dir."""
        meta_path = self.local.root / "dataset-metadata.json"
        if meta_path.exists() or not self.dataset_slug:
            return
        title = self.dataset_slug.split("/")[-1].replace("-", " ")
        meta_path.write_text(json.dumps({
            "title": title,
            "id": self.dataset_slug,
            "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8")

    def write_value_chunk(self, name: str, X: np.ndarray, y: np.ndarray) -> str:
        return self.local.write_value_chunk(name, X, y)

    def write_metadata(self, name: str, records: list[dict[str, Any]]) -> str:
        return self.local.write_metadata(name, records)

    def write_raw(self, name: str, blob: Any) -> Optional[str]:
        return self.local.write_raw(name, blob)

    def publish(self, message: str) -> bool:
        """Run ``kaggle datasets version`` on the local upload dir."""
        if not self.dataset_slug:
            return False
        self._ensure_metadata()
        args = ["kaggle", "datasets", "version", "-p", str(self.local.root),
                "-m", message, "--dir-mode", "zip"]
        try:
            code, out, err = self._runner(args, self.timeout)
        except FileNotFoundError:
            if self._log is not None:
                from .logutil import log_kv
                log_kv(self._log, "publish_failed", reason="kaggle CLI not found")
            return False
        ok = code == 0
        if self._log is not None:
            from .logutil import log_kv
            log_kv(self._log, "publish", ok=ok, slug=self.dataset_slug,
                   detail=(err.strip()[:200] if not ok else out.strip()[:120]))
        return ok


def build_sink(config, logger: Any | None = None) -> Sink:
    """Factory: construct the configured sink from a :class:`CollectorConfig`."""
    local = LocalSink(config.data_dir)
    if config.sink == "kaggle":
        return KaggleDatasetSink(local, config.dataset_slug,
                                 timeout=config.request_timeout * 5, logger=logger)
    return local
