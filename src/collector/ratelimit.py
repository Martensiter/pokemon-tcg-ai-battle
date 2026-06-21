"""Self rate-limiting + exponential backoff for the replay endpoints.

The replay/log endpoints are throttled server-side, so we *always* sleep between
calls (``RateLimiter``) and back off on 429/5xx (``retry_with_backoff``). The
clock function is injectable so tests run without real sleeps.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """Raised by a callable to request an exponential-backoff retry.

    ``status`` carries the HTTP-ish code when known (429, 503, ...) for logging.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class FatalError(Exception):
    """Non-retryable failure (bad request, auth, parse error)."""


class RateLimiter:
    """Enforce a minimum interval between successive calls.

    Args:
        min_interval: minimum seconds between calls (``0`` disables).
        sleep: injectable sleep function (defaults to ``time.sleep``).
        clock: injectable monotonic clock (defaults to ``time.monotonic``).
    """

    def __init__(self, min_interval: float,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.min_interval = max(0.0, min_interval)
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> float:
        """Block until at least ``min_interval`` has elapsed since the last call.

        Returns the number of seconds actually slept (useful for logging/tests).
        """
        if self.min_interval <= 0:
            self._last = self._clock()
            return 0.0
        now = self._clock()
        slept = 0.0
        if self._last is not None:
            elapsed = now - self._last
            remaining = self.min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
                slept = remaining
        self._last = self._clock()
        return slept


def backoff_delay(attempt: int, base: float, cap: float,
                  jitter: Callable[[], float] | None = None) -> float:
    """Full-jitter exponential backoff delay for a zero-based ``attempt``.

    delay = random(0, min(cap, base * 2**attempt)). ``jitter`` returns a value in
    [0, 1) and is injectable for deterministic tests.
    """
    raw = min(cap, base * (2 ** attempt))
    j = jitter() if jitter is not None else random.random()
    return raw * j


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base: float,
    cap: float,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] | None = None,
    on_backoff: Callable[[int, float, RetryableError], None] | None = None,
) -> T:
    """Call ``fn`` retrying on :class:`RetryableError` with exponential backoff.

    Raises the last :class:`RetryableError` if the retry budget is exhausted, or
    propagates :class:`FatalError` immediately.
    """
    attempt = 0
    last: RetryableError | None = None
    while True:
        try:
            return fn()
        except FatalError:
            raise
        except RetryableError as e:
            last = e
            if attempt >= max_retries:
                raise
            delay = backoff_delay(attempt, base, cap, jitter)
            if on_backoff is not None:
                on_backoff(attempt, delay, e)
            sleep(delay)
            attempt += 1
    # Unreachable, but keeps type checkers happy.
    assert last is not None
    raise last
