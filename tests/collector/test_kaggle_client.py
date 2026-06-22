"""Kaggle client tests: throttle detection -> backoff -> success, CSV/JSON parse."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import conftest as cf

from collector.config import CollectorConfig
from collector.kaggle_client import KaggleClient, _looks_retryable, _parse_csv
from collector.ratelimit import FatalError, RateLimiter


def _client(runner, **over):
    # Use a throwaway data_dir so replay/logs downloads never touch the repo.
    over.setdefault("data_dir", Path(tempfile.mkdtemp(prefix="kclient_")))
    cfg = CollectorConfig(rps=0.0, max_retries=4, backoff_base=0.0, **over)
    # rps=0 -> no real sleep; backoff sleep is also stubbed below.
    return KaggleClient(cfg, runner=runner, limiter=RateLimiter(0.0),
                        sleep=lambda s: None)


def test_looks_retryable():
    assert _looks_retryable("Error 429 Too Many Requests") == 429
    assert _looks_retryable("HTTP 503 service unavailable") == 503
    assert _looks_retryable("invalid card id") is None


def test_parse_csv_basic():
    rows = _parse_csv("a,b\n1,2\n3,4\n")
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    assert _parse_csv("not csv") == []
    assert _parse_csv("") == []


def test_leaderboard_parses_csv():
    def runner(args, timeout):
        return 0, "teamName,submissionId,score\nAce,111,950\nBee,222,940\n", ""

    c = _client(runner)
    rows = c.leaderboard()
    assert len(rows) == 2
    assert rows[0]["submissionId"] == "111"


def test_replay_parses_json_stdout():
    blob = cf.make_episode_steps(winner=0)

    def runner(args, timeout):
        return 0, json.dumps(blob), ""

    c = _client(runner)
    out = c.replay("ep1")
    assert out["episode_id"] == "ep1"
    assert "steps" in out["replay"]


def test_retries_then_succeeds():
    state = {"n": 0}
    blob = cf.make_episode_visualize(winner=1)

    def runner(args, timeout):
        state["n"] += 1
        if state["n"] < 3:
            return 1, "", "429 Too Many Requests"
        return 0, json.dumps(blob), ""

    c = _client(runner)
    out = c.replay("ep9")
    assert state["n"] == 3
    assert "visualize" in out["replay"]


def test_fatal_error_not_retried():
    state = {"n": 0}

    def runner(args, timeout):
        state["n"] += 1
        return 1, "", "404 not found: invalid episode"

    c = _client(runner)
    with pytest.raises(FatalError):
        c.replay("bad")
    assert state["n"] == 1  # no retry on fatal


def test_replay_no_json_is_fatal():
    def runner(args, timeout):
        return 0, "<html>captcha</html>", ""

    c = _client(runner)
    with pytest.raises(FatalError):
        c.replay("ep1")


def test_timeout_marker_is_retryable():
    state = {"n": 0}
    blob = cf.make_episode_visualize(winner=0)

    def runner(args, timeout):
        state["n"] += 1
        if state["n"] == 1:
            return 124, "", "timeout after 120s"
        return 0, json.dumps(blob), ""

    c = _client(runner)
    out = c.replay("ep1")
    assert state["n"] == 2 and "visualize" in out["replay"]


def test_replay_reads_downloaded_file_and_cleans_up():
    """Real CLI writes episode-<id>-replay.json (no stdout); we read + delete it."""
    blob = cf.make_episode_steps(winner=0)
    eid = "98765432"
    c = _client(lambda a, t: (0, "", ""))  # placeholder; replaced below
    dl_file = c._dl_dir / f"episode-{eid}-replay.json"

    def runner(args, timeout):
        c._dl_dir.mkdir(parents=True, exist_ok=True)
        dl_file.write_text(json.dumps(blob), encoding="utf-8")
        return 0, "", ""          # CLI prints nothing; it downloads a file

    c._runner = runner
    out = c.replay(eid)
    assert "steps" in out["replay"]
    assert not dl_file.exists()    # cleaned up after parsing


def test_replay_missing_file_is_fatal():
    c = _client(lambda a, t: (0, "", ""))   # success exit, but no file, no stdout
    with pytest.raises(FatalError):
        c.replay("123")


def test_episodes_csv_then_json_fallback():
    def runner(args, timeout):
        return 0, json.dumps([{"episodeId": "e1"}, {"episodeId": "e2"}]), ""

    c = _client(runner)
    rows = c.episodes("sub1")
    assert [r["episodeId"] for r in rows] == ["e1", "e2"]
