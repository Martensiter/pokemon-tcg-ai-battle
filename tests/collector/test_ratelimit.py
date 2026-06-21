"""Rate limiter + exponential backoff tests (no real sleeps)."""
from __future__ import annotations

import pytest

from collector.ratelimit import (
    FatalError, RateLimiter, RetryableError, backoff_delay, retry_with_backoff,
)


def test_rate_limiter_sleeps_to_min_interval():
    slept = []
    t = [0.0]
    rl = RateLimiter(min_interval=1.0, sleep=lambda s: slept.append(s), clock=lambda: t[0])
    rl.wait()                 # first call: no wait
    assert slept == []
    t[0] = 0.3                # only 0.3s elapsed
    rl.wait()                 # should sleep ~0.7s
    assert slept and abs(slept[0] - 0.7) < 1e-9


def test_rate_limiter_disabled():
    rl = RateLimiter(min_interval=0.0, sleep=lambda s: pytest.fail("should not sleep"))
    assert rl.wait() == 0.0
    assert rl.wait() == 0.0


def test_backoff_delay_monotone_cap():
    full = lambda: 1.0  # no jitter (use full window)
    d0 = backoff_delay(0, base=2.0, cap=100.0, jitter=full)
    d3 = backoff_delay(3, base=2.0, cap=100.0, jitter=full)
    dbig = backoff_delay(20, base=2.0, cap=100.0, jitter=full)
    assert d0 == 2.0
    assert d3 == 16.0
    assert dbig == 100.0  # capped


def test_retry_succeeds_after_transient():
    calls = {"n": 0}
    sleeps = []

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("429", status=429)
        return "ok"

    out = retry_with_backoff(fn, max_retries=5, base=1.0, cap=10.0,
                             sleep=sleeps.append, jitter=lambda: 0.5)
    assert out == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # backed off twice


def test_retry_exhausts_budget():
    def fn():
        raise RetryableError("503", status=503)

    with pytest.raises(RetryableError):
        retry_with_backoff(fn, max_retries=2, base=1.0, cap=5.0,
                           sleep=lambda s: None, jitter=lambda: 0.0)


def test_fatal_not_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise FatalError("bad request")

    with pytest.raises(FatalError):
        retry_with_backoff(fn, max_retries=5, base=1.0, cap=5.0,
                           sleep=lambda s: None)
    assert calls["n"] == 1
