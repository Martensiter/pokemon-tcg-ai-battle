"""Unit-test the score report's pure logic (W-L tallies, trajectory thinning,
team-name resolution). The network layer (fetch_episodes) is a separate
function and is never called here -- everything runs on fixture dicts."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.report_scores import (
    build_report, classify_reward, parse_utc, render_report, render_summary,
    resolve_team, team_names, thin_series,
)

MY_SUB = 54269263
MY_TEAM = 16421669


def _agent(sub, reward, score, team):
    return {"submissionId": sub, "reward": reward, "updatedScore": score,
            "teamId": team}


def _episode(eid, t, agents, etype="EPISODE_TYPE_PUBLIC"):
    return {"id": eid, "createTime": t, "endTime": t, "state": "COMPLETED",
            "type": etype, "agents": agents}


def make_payload():
    """Mimics a real ListEpisodes response: validation first, then 3 public
    games (W, L, missing-reward) plus one public game with unknown teamId."""
    eps = [
        _episode(1, "2026-07-02T15:14:29.371374300Z",
                 [_agent(MY_SUB, 1, 600, MY_TEAM),
                  _agent(MY_SUB, -1, 600, MY_TEAM)],
                 etype="EPISODE_TYPE_VALIDATION"),
        _episode(2, "2026-07-02T15:21:24.979046900Z",
                 [_agent(999, -1, 595.9, 111),
                  _agent(MY_SUB, 1, 708.5, MY_TEAM)]),
        _episode(3, "2026-07-02T15:25:26.424011200Z",
                 [_agent(998, 1, 759.0, 222),
                  _agent(MY_SUB, -1, 624.7, MY_TEAM)]),
        _episode(4, "2026-07-02T15:29:25.117294800Z",
                 [_agent(MY_SUB, None, 630.0, MY_TEAM),
                  _agent(997, None, 590.0, 333)]),
        _episode(5, "2026-07-02T16:17:30.906299800Z",
                 [_agent(MY_SUB, 1, 700.4, MY_TEAM),
                  _agent(996, -1, 674.8, 44444)]),  # 44444 not in teams[]
    ]
    teams = [
        {"id": 111, "teamName": "alpha"},
        {"id": 222, "teamName": "beta"},
        {"id": 333, "teamName": "gamma"},
        {"id": MY_TEAM, "teamName": "ichitaro3"},
    ]
    return {"episodes": eps, "teams": teams, "submissions": []}


# cutoff = NOW - 24 h = 2026-07-02 16:00 -> only episode 5 (16:17) is "recent"
NOW = datetime(2026, 7, 3, 16, 0, tzinfo=timezone.utc)


def test_parse_utc_handles_variable_fractions():
    a = parse_utc("2026-07-02T16:17:30.906299800Z")   # 9 fractional digits
    b = parse_utc("2026-07-02T15:59:52.447421Z")      # 6 digits
    c = parse_utc("2026-07-02T15:00:00Z")             # none
    assert a.tzinfo is timezone.utc and a.microsecond == 906299
    assert b.second == 52 and c.minute == 0


def test_classify_reward():
    assert classify_reward(1) == "W"
    assert classify_reward(-1) == "L"
    assert classify_reward(None) == "D/ERR"   # missing reward = draw / error
    assert classify_reward(0) == "D/ERR"


def test_team_name_resolution():
    names = team_names(make_payload())
    assert resolve_team(111, names) == "alpha"
    assert resolve_team(MY_TEAM, names) == "ichitaro3"
    assert resolve_team(44444, names) == "team 44444"   # unknown -> placeholder


def test_thin_series():
    assert thin_series([1, 2, 3]) == [1, 2, 3]          # short passes through
    long = list(range(100))
    out = thin_series(long, max_points=10)
    assert len(out) == 10
    assert out[0] == 0 and out[-1] == 99                # endpoints preserved
    assert out == sorted(out)                           # order preserved


def test_build_report_aggregates():
    r = build_report(make_payload(), MY_SUB, recent=2, now=NOW)
    assert r["submission_id"] == MY_SUB
    assert r["public_episodes"] == 4                    # validation excluded
    assert (r["wins"], r["losses"], r["other"]) == (2, 1, 1)
    assert r["score"] == 700.4                          # latest public score
    assert r["trajectory"][0] == 708.5 and r["trajectory"][-1] == 700.4
    assert r["first_episode_utc"] == "2026-07-02 15:21"
    assert r["last_episode_utc"] == "2026-07-02 16:17"
    assert r["last_24h_games"] == 1                     # only ep 5 within 24 h


def test_build_report_recent_games():
    r = build_report(make_payload(), MY_SUB, recent=2, now=NOW)
    assert len(r["recent_games"]) == 2                  # capped by recent=
    newest, older = r["recent_games"]                   # newest first
    assert newest["time_utc"] == "2026-07-02 16:17"
    assert newest["result"] == "W" and newest["score"] == 700.4
    assert newest["opponent"] == "team 44444"           # unknown team fallback
    assert older["result"] == "D/ERR"
    assert older["opponent"] == "gamma"


def test_build_report_validation_only_falls_back_to_600():
    payload = make_payload()
    payload["episodes"] = payload["episodes"][:1]       # validation episode only
    r = build_report(payload, MY_SUB, now=NOW)
    assert r["public_episodes"] == 0
    assert r["score"] == 600                            # baseline fallback
    assert r["trajectory"] == [] and r["recent_games"] == []


def test_render_smoke():
    r = build_report(make_payload(), MY_SUB, recent=3, now=NOW)
    text = render_report(r)
    assert "Submission 54269263" in text and "W-L: 2-1" in text
    summary = render_summary([r, {"submission_id": 1, "error": "HTTP 500"}])
    assert "700.4" in summary and "HTTP 500" in summary


def test_ordering_uses_end_time_not_create_time():
    # Episode A was created first but finished LAST (long game): the current
    # score must come from A (endTime-latest), not B (createTime-latest).
    ep_a = _episode(10, "2026-07-02T15:00:00Z",
                    [_agent(MY_SUB, 1, 720.0, MY_TEAM),
                     _agent(999, -1, 600.0, 111)])
    ep_a["endTime"] = "2026-07-02T15:30:00Z"
    ep_b = _episode(11, "2026-07-02T15:05:00Z",
                    [_agent(MY_SUB, -1, 690.0, MY_TEAM),
                     _agent(998, 1, 610.0, 222)])
    ep_b["endTime"] = "2026-07-02T15:10:00Z"
    payload = {"episodes": [ep_a, ep_b], "teams": [], "submissions": []}
    r = build_report(payload, MY_SUB,
                     now=datetime(2026, 7, 2, 16, 0, tzinfo=timezone.utc))
    assert r["score"] == 720.0            # endTime-latest episode wins
    assert r["trajectory"] == [690.0, 720.0]
    assert r["recent_games"][0]["score"] == 720.0  # newest first by endTime


def test_missing_end_time_falls_back_to_create_time():
    ep = _episode(12, "2026-07-02T15:00:00Z",
                  [_agent(MY_SUB, 1, 640.0, MY_TEAM),
                   _agent(999, -1, 600.0, 111)])
    del ep["endTime"]
    r = build_report({"episodes": [ep], "teams": [], "submissions": []},
                     MY_SUB,
                     now=datetime(2026, 7, 2, 16, 0, tzinfo=timezone.utc))
    assert r["score"] == 640.0
    assert r["last_episode_utc"] == "2026-07-02 15:00"


def test_fetch_retries_transient_and_fails_fast_on_client_error(monkeypatch):
    import io
    import json as _json
    import urllib.error
    import urllib.request

    import tools.report_scores as rs

    calls = {"n": 0}

    def flaky_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:  # two 503s, then success
            raise urllib.error.HTTPError(req.full_url, 503, "boom", {}, io.BytesIO())

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Resp(_json.dumps({"episodes": []}).encode())

    slept = []
    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    payload = rs.fetch_episodes(1, sleep=slept.append)
    assert payload == {"episodes": []}
    assert calls["n"] == 3 and len(slept) == 2   # backed off twice
    assert slept[0] < slept[1]                   # exponential

    def always_404(req, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "nope", {}, io.BytesIO())

    calls["n"] = 0
    monkeypatch.setattr(urllib.request, "urlopen", always_404)
    try:
        rs.fetch_episodes(1, sleep=slept.append)
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError:
        pass
    assert calls["n"] == 1                       # no retry on permanent 4xx
