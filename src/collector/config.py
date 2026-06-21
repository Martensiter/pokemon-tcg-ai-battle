"""Collector configuration: a dataclass populated from environment variables.

All secrets (Kaggle credentials) come from the environment only -- never from
files committed to the repo. A ``.env.example`` documents every knob. We avoid a
hard dependency on ``pydantic``/``python-dotenv`` so the collector stays
pure-Python and trivially installable on a locked-down ARM device; a tiny
``.env`` loader is included for convenience.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    """Repo root: this file is ``<root>/src/collector/config.py``."""
    return Path(__file__).resolve().parents[2]


def load_dotenv(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Minimal ``.env`` reader.

    Loads ``KEY=VALUE`` lines into ``os.environ`` (without overwriting existing
    values) and returns the parsed mapping. Lines starting with ``#`` and blank
    lines are ignored. Surrounding quotes are stripped. This is intentionally
    tiny -- we do not need the full python-dotenv feature set, and keeping the
    dependency surface small matters on the ARM target.
    """
    p = Path(path) if path else _project_root() / ".env"
    parsed: dict[str, str] = {}
    if not p.exists():
        return parsed
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        parsed[key] = val
        os.environ.setdefault(key, val)
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    v = os.environ.get(name, "")
    return [s.strip() for s in v.split(",") if s.strip()]


@dataclass
class CollectorConfig:
    """All collector knobs. Construct via :meth:`from_env`.

    Attributes:
        competition: Kaggle competition slug.
        kaggle_username / kaggle_key: credentials (env only; may be blank in
            mock/offline mode).
        rps: target requests-per-second cap for replay/log fetches. The
            collector sleeps to keep at or below this rate, independent of how
            fast submissions allow us to *submit* (a separate, slower limit).
        max_retries: retry budget for throttled (429) / server (5xx) responses.
        backoff_base / backoff_cap: exponential backoff seconds.
        request_timeout: per-CLI-call timeout (seconds).
        top_n_leaders: how many leaderboard rows to target.
        target_teams: explicit team/agent name substrings to prioritise (case
            insensitive). Empty -> derive from the leaderboard top-N.
        episodes_per_submission: cap episodes pulled per submission per pass
            (0 = unlimited).
        keep_raw: if True, also persist the raw replay JSON (opt-in; default is
            converted records only to save space).
        save_value_records: emit value-net training npz chunks.
        save_episode_meta: emit small per-episode metadata records.
        chunk_size: flush a training chunk after this many converted episodes.
        loop_interval: seconds to sleep between discovery passes in the long
            loop.
        sink: 'local' or 'kaggle'.
        dataset_slug: Kaggle Dataset slug (``owner/name``) for KaggleDatasetSink.
        data_dir / state_dir: local paths for outputs and manifest.
    """

    competition: str = "pokemon-tcg-ai-battle"
    kaggle_username: str = ""
    kaggle_key: str = ""

    rps: float = 0.2
    max_retries: int = 6
    backoff_base: float = 2.0
    backoff_cap: float = 120.0
    request_timeout: float = 120.0

    top_n_leaders: int = 10
    target_teams: list[str] = field(default_factory=list)
    episodes_per_submission: int = 0

    keep_raw: bool = False
    save_value_records: bool = True
    save_episode_meta: bool = True
    chunk_size: int = 200
    loop_interval: float = 900.0

    sink: str = "local"
    dataset_slug: str = ""

    data_dir: Path = field(default_factory=lambda: _project_root() / "collector_data")
    state_dir: Path = field(default_factory=lambda: _project_root() / "collector_state")

    @classmethod
    def from_env(cls, load_env_file: bool = True) -> "CollectorConfig":
        """Build a config from process environment (optionally loading ``.env``)."""
        if load_env_file:
            load_dotenv()
        root = _project_root()
        data_dir = Path(os.environ.get("COLLECTOR_DATA_DIR", str(root / "collector_data")))
        state_dir = Path(os.environ.get("COLLECTOR_STATE_DIR", str(root / "collector_state")))
        return cls(
            competition=os.environ.get("COLLECTOR_COMPETITION", "pokemon-tcg-ai-battle"),
            kaggle_username=os.environ.get("KAGGLE_USERNAME", ""),
            kaggle_key=os.environ.get("KAGGLE_KEY", ""),
            rps=_env_float("COLLECTOR_RPS", 0.2),
            max_retries=_env_int("COLLECTOR_MAX_RETRIES", 6),
            backoff_base=_env_float("COLLECTOR_BACKOFF_BASE", 2.0),
            backoff_cap=_env_float("COLLECTOR_BACKOFF_CAP", 120.0),
            request_timeout=_env_float("COLLECTOR_REQUEST_TIMEOUT", 120.0),
            top_n_leaders=_env_int("COLLECTOR_TOP_N", 10),
            target_teams=_env_list("COLLECTOR_TARGET_TEAMS"),
            episodes_per_submission=_env_int("COLLECTOR_EPISODES_PER_SUB", 0),
            keep_raw=_env_bool("COLLECTOR_KEEP_RAW", False),
            save_value_records=_env_bool("COLLECTOR_SAVE_VALUE", True),
            save_episode_meta=_env_bool("COLLECTOR_SAVE_META", True),
            chunk_size=_env_int("COLLECTOR_CHUNK_SIZE", 200),
            loop_interval=_env_float("COLLECTOR_LOOP_INTERVAL", 900.0),
            sink=os.environ.get("COLLECTOR_SINK", "local").strip().lower(),
            dataset_slug=os.environ.get("DATASET_SLUG", ""),
            data_dir=data_dir,
            state_dir=state_dir,
        )

    def has_credentials(self) -> bool:
        return bool(self.kaggle_username and self.kaggle_key)

    def min_interval(self) -> float:
        """Minimum seconds between throttled requests, derived from ``rps``."""
        return 1.0 / self.rps if self.rps > 0 else 0.0

    def redacted(self) -> dict[str, object]:
        """A logging-safe view: secrets masked, paths stringified."""
        out: dict[str, object] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in ("kaggle_key",):
                val = "***" if val else ""
            elif f.name == "kaggle_username":
                val = (val[:2] + "***") if val else ""
            elif isinstance(val, Path):
                val = str(val)
            out[f.name] = val
        return out
