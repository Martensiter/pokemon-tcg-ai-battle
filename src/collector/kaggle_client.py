"""Thin wrapper over the official Kaggle CLI for the collection endpoints.

Only the *official* CLI/API is used -- no HTML scraping (that hits reCAPTCHA).
Every call goes through the single :meth:`KaggleClient._run` seam (subprocess),
which makes the whole client trivially mockable in tests: patch ``_run`` to feed
canned stdout instead of hitting the network.

Endpoints wrapped (command strings per the task spec):
  * ``kaggle competitions leaderboard <comp> -s``      -> top submissions
  * ``kaggle competitions submissions <comp>``         -> our/leader active agents
  * ``kaggle competitions episodes <submission_id>``   -> episode ids
  * ``kaggle competitions replay <episode_id>``        -> replay JSON
  * ``kaggle competitions logs <episode_id> <idx>``    -> agent logs

NOTE (unconfirmed): the exact subcommand surface for ``episodes``/``replay`` may
differ across Kaggle CLI versions / simulation-competition APIs. Command
construction is centralised in the small ``_cmd_*`` helpers so it can be adapted
in one place once verified against the live CLI. Parsing is defensive and never
assumes a rigid schema.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
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
        return ["kaggle", "competitions", "replay", str(episode_id)]

    def _cmd_logs(self, episode_id: str, agent_index: int) -> list[str]:
        return ["kaggle", "competitions", "logs", str(episode_id), str(agent_index)]

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

        The CLI may print JSON to stdout or download a file; we handle both. A
        non-JSON / empty body is a :class:`FatalError` for this episode (the
        caller records it as failed and moves on -- it is not retried forever).
        """
        out = self._run(self._cmd_replay(episode_id))
        data = _maybe_json(out)
        if data is None:
            # Some CLI versions write to a file named after the episode.
            path = self._find_downloaded(str(episode_id))
            if path is not None:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    raise FatalError(f"replay {episode_id}: unreadable download: {e}") from e
        if not isinstance(data, (dict, list)):
            raise FatalError(f"replay {episode_id}: no JSON payload")
        return {"episode_id": episode_id, "replay": data}

    def logs(self, episode_id: str, agent_index: int) -> Any:
        """Agent logs for one seat (JSON if parseable, else raw text)."""
        out = self._run(self._cmd_logs(episode_id, agent_index))
        data = _maybe_json(out)
        return data if data is not None else out

    def _find_downloaded(self, episode_id: str):
        from pathlib import Path
        cwd = Path.cwd()
        for cand in (cwd / f"{episode_id}.json", cwd / f"{episode_id}",
                     self.cfg.data_dir / f"{episode_id}.json"):
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
