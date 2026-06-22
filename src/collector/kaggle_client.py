"""Thin wrapper over the official Kaggle CLI for the collection endpoints.

Only the *official* CLI/API is used -- no HTML scraping (that hits reCAPTCHA).
Every call goes through the single :meth:`KaggleClient._run` seam (subprocess),
which makes the whole client trivially mockable in tests: patch ``_run`` to feed
canned stdout instead of hitting the network.

Endpoints wrapped (verified against Kaggle CLI simulation-competition docs):
  * ``kaggle competitions leaderboard <comp> -s --csv``   -> top submissions (CSV)
  * ``kaggle competitions submissions <comp> --csv``      -> active agents (CSV)
  * ``kaggle competitions episodes <submission_id> --csv``-> episode rows (CSV)
  * ``kaggle competitions replay <episode_id> -p <dir>``  -> downloads
        ``episode-<id>-replay.json`` (NOT stdout)
  * ``kaggle competitions logs <episode_id> <idx> -p <dir>`` -> downloads
        ``episode-<id>-agent-<idx>-logs.json``

``replay``/``logs`` write a *file* (per the official docs) rather than printing to
stdout, so we point ``-p`` at a controlled download dir, read the known filename,
and delete it after parsing (the collector re-persists raw via the sink only when
``keep_raw`` is set, so a 24/7 run never accumulates stray files). Parsing stays
defensive and never assumes a rigid schema.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from .config import CollectorConfig
from .ratelimit import FatalError, RateLimiter, RetryableError, retry_with_backoff

# Substrings that indicate a server-side throttle or transient failure.
_RETRYABLE_MARKERS = (
    "429", "too many requests", "throttl", "rate limit", "rate-limit",
    "500", "502", "503", "504", "timed out", "timeout", "temporarily unavailable",
    "connection reset", "connection aborted", "service unavailable",
)


def _looks_retryable(text: str) -> Optional[int]:
    """Return a best-effort status code if ``text`` looks transient, else None."""
    low = text.lower()
    for m in _RETRYABLE_MARKERS:
        if m in low:
            for code in (429, 503, 502, 504, 500):
                if str(code) in low:
                    return code
            return 429
    return None


class KaggleClient:
    """Rate-limited, retrying client around the Kaggle CLI.

    Args:
        config: collector config (rps, retries, timeouts, credentials).
        runner: optional override for the subprocess runner (for tests). It is
            called as ``runner(args, timeout)`` and must return a
            ``(returncode, stdout, stderr)`` tuple.
        limiter / sleep: injectable rate limiter and sleep (tests).
        logger: optional structured logger; backoff events are reported via it.
    """

    def __init__(self, config: CollectorConfig,
                 runner: Callable[[list[str], float], tuple[int, str, str]] | None = None,
                 limiter: RateLimiter | None = None,
                 sleep: Callable[[float], None] | None = None,
                 logger: Any | None = None):
        self.cfg = config
        self._runner = runner or self._default_runner
        self._limiter = limiter or RateLimiter(config.min_interval())
        import time as _time
        self._sleep = sleep or _time.sleep
        self._log = logger
        # replay/logs download files here; cleaned up after each read.
        self._dl_dir = Path(config.data_dir) / ".downloads"

    # --- subprocess seam --------------------------------------------------
    def _default_runner(self, args: list[str], timeout: float) -> tuple[int, str, str]:
        """Run the kaggle CLI. Credentials flow via env (KAGGLE_USERNAME/KEY)."""
        env = dict(os.environ)
        if self.cfg.kaggle_username:
            env["KAGGLE_USERNAME"] = self.cfg.kaggle_username
        if self.cfg.kaggle_key:
            env["KAGGLE_KEY"] = self.cfg.kaggle_key
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired as e:
            return 124, "", f"timeout after {timeout}s: {e}"
        except FileNotFoundError as e:
            raise FatalError(f"kaggle CLI not found: {e}") from e
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    def _run(self, args: list[str]) -> str:
        """Execute a CLI call with rate-limiting + backoff. Returns stdout.

        Raises :class:`RetryableError` (after exhausting retries) on transient
        failures, or :class:`FatalError` on permanent ones.
        """
        def attempt() -> str:
            self._limiter.wait()
            code, out, err = self._runner(args, self.cfg.request_timeout)
            if code == 0:
                return out
            blob = f"{out}\n{err}"
            status = _looks_retryable(blob)
            if code == 124:  # our timeout marker
                status = status or 504
            if status is not None:
                raise RetryableError(f"transient kaggle CLI failure (status~{status}): {err.strip()[:200]}",
                                     status=status)
            raise FatalError(f"kaggle CLI failed (code {code}): {err.strip()[:300] or out.strip()[:300]}")

        def on_backoff(attempt_i: int, delay: float, e: RetryableError) -> None:
            if self._log is not None:
                from .logutil import log_kv
                log_kv(self._log, "backoff", event="backoff", attempt=attempt_i,
                       delay=round(delay, 2), status=e.status, cmd=args[:3])

        return retry_with_backoff(
            attempt,
            max_retries=self.cfg.max_retries,
            base=self.cfg.backoff_base,
            cap=self.cfg.backoff_cap,
            sleep=self._sleep,
            on_backoff=on_backoff,
        )

    # --- command builders (centralised for easy adaptation) ---------------
    def _cmd_leaderboard(self) -> list[str]:
        return ["kaggle", "competitions", "leaderboard", self.cfg.competition, "-s", "--csv"]

    def _cmd_submissions(self) -> list[str]:
        return ["kaggle", "competitions", "submissions", self.cfg.competition, "--csv"]

    def _cmd_episodes(self, submission_id: str) -> list[str]:
        return ["kaggle", "competitions", "episodes", str(submission_id), "--csv"]

    def _cmd_replay(self, episode_id: str) -> list[str]:
        return ["kaggle", "competitions", "replay", str(episode_id),
                "-p", str(self._dl_dir)]

    def _cmd_logs(self, episode_id: str, agent_index: int) -> list[str]:
        return ["kaggle", "competitions", "logs", str(episode_id), str(agent_index),
                "-p", str(self._dl_dir)]

    # --- public API -------------------------------------------------------
    def leaderboard(self) -> list[dict[str, str]]:
        """Top-of-leaderboard rows (best-effort CSV parse)."""
        return _parse_csv(self._run(self._cmd_leaderboard()))

    def submissions(self) -> list[dict[str, str]]:
        """Submissions for the competition (CSV parse)."""
        return _parse_csv(self._run(self._cmd_submissions()))

    def episodes(self, submission_id: str) -> list[dict[str, str]]:
        """Episode rows for a submission (CSV parse, falls back to JSON)."""
        out = self._run(self._cmd_episodes(submission_id))
        rows = _parse_csv(out)
        if rows:
            return rows
        data = _maybe_json(out)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        return []

    def replay(self, episode_id: str) -> dict[str, Any]:
        """Fetch a replay and return it as a dict.

        The CLI downloads ``episode-<id>-replay.json`` (we also accept JSON on
        stdout for forward/back compat and tests). The downloaded file is deleted
        after parsing. A non-JSON / empty result is a :class:`FatalError` for this
        episode (the caller records it failed and moves on -- not retried forever).
        """
        self._dl_dir.mkdir(parents=True, exist_ok=True)
        out = self._run(self._cmd_replay(episode_id))
        data = _maybe_json(out)
        if data is None:
            path = self._find_downloaded(str(episode_id), suffix="replay")
            if path is not None:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                finally:
                    path.unlink(missing_ok=True)  # don't accumulate raw files
        if not isinstance(data, (dict, list)):
            raise FatalError(f"replay {episode_id}: no JSON payload (expected "
                             f"{self._dl_dir}/episode-{episode_id}-replay.json)")
        return {"episode_id": episode_id, "replay": data}

    def logs(self, episode_id: str, agent_index: int) -> Any:
        """Agent logs for one seat (JSON if available, else raw text/None).

        Downloads ``episode-<id>-agent-<idx>-logs.json``; falls back to stdout.
        """
        self._dl_dir.mkdir(parents=True, exist_ok=True)
        out = self._run(self._cmd_logs(episode_id, agent_index))
        data = _maybe_json(out)
        if data is None:
            path = self._find_downloaded(str(episode_id), suffix=f"agent-{agent_index}-logs")
            if path is not None:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001  (logs are best-effort, not critical)
                    data = path.read_text(encoding="utf-8")
                finally:
                    path.unlink(missing_ok=True)
        return data if data is not None else (out or None)

    def _find_downloaded(self, episode_id: str, suffix: str):
        """Locate a CLI-downloaded file ``episode-<id>-<suffix>.json``.

        Checks the configured download dir first, then the CWD (some CLI versions
        ignore ``-p`` / default to CWD). Tolerates legacy ``<id>.json`` names.
        """
        names = [f"episode-{episode_id}-{suffix}.json", f"{episode_id}.json", f"{episode_id}"]
        for base in (self._dl_dir, Path.cwd()):
            for nm in names:
                cand = base / nm
                if cand.exists():
                    return cand
        return None


def _parse_csv(text: str) -> list[dict[str, str]]:
    """Parse CLI CSV output into a list of dict rows. Empty/garbage -> []."""
    text = (text or "").strip()
    if not text or "," not in text.splitlines()[0]:
        return []
    try:
        reader = csv.DictReader(io.StringIO(text))
        return [dict(r) for r in reader]
    except Exception:  # noqa: BLE001
        return []


def _maybe_json(text: str) -> Any | None:
    """Return parsed JSON if ``text`` is a JSON document, else None."""
    text = (text or "").strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
