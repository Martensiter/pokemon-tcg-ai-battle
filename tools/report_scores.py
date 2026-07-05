"""One-shot score report for our Kaggle simulation submissions.

For every tracked submission this prints: current score, public-episode count,
W-L record, first/last episode times, games in the last 24 h, a thinned score
trajectory, and the most recent matches (opponent team resolved by name).
A comparison table across all submissions closes the report.

Data source is the same public read-only JSON endpoint the kaggle.com episode
page uses (POST ListEpisodes with a submissionId body) -- no auth, no HTML
scraping. We sleep between requests to stay polite and retry transient
failures (HTTP 429/5xx, network errors) with exponential backoff.

CAVEAT -- unofficial endpoint: this tool depends on the internal JSON RPC used
by the kaggle.com frontend (api/i/competitions.EpisodeService/ListEpisodes).
We rely on it because the per-episode rating (``updatedScore``) is not exposed
by the official Kaggle CLI. If the endpoint shape ever changes, fall back to
the official ``kaggle competitions episodes <id> --csv`` -- that keeps the
episode list working but loses ``updatedScore`` (no score trajectory).

Usage:
    python tools/report_scores.py                       # DEFAULT_SUBMISSIONS
    python tools/report_scores.py --subs 54269263       # explicit list
    python tools/report_scores.py --recent 12 --json    # machine-readable

stdlib only (urllib.request / json / argparse); Python 3.11+.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable

LIST_EPISODES_URL = (
    "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
)

MY_TEAM_ID = 16421669  # ichitaro3

DEFAULT_SUBMISSIONS = [
    54297461,  # value-D: 117-dim on official top-episodes 419k (2026-07-03)
    54269242,  # 117-dim acting-hand, hub data (2026-07-02)
    54269263,  # 124-dim rerun (2026-07-02)
    54144305,  # 108-dim hand_id removed (2026-06-28)
    54143607,  # 124-dim original (2026-06-28)
]

REQUEST_SLEEP_S = 1.5   # pause between API calls (be nice to Kaggle)
TRAJECTORY_POINTS = 10  # target length of the thinned score trajectory

MAX_RETRIES = 3         # extra attempts after the first (4 tries total)
BACKOFF_BASE_S = 2.0    # retry delay = min(cap, base * 2**attempt)
BACKOFF_CAP_S = 30.0


# ---------------------------------------------------------------------------
# network layer (kept separate so the pure-logic unit tests never touch it)
# ---------------------------------------------------------------------------

def _is_retryable_status(status: int) -> bool:
    """429 (rate limit) and 5xx (server hiccup) are transient; other 4xx are
    permanent client errors and must fail immediately."""
    return status == 429 or 500 <= status <= 599


def fetch_episodes(submission_id: int, timeout: float = 30.0, *,
                   max_retries: int = MAX_RETRIES,
                   sleep: Callable[[float], None] = time.sleep) -> dict:
    """POST the public ListEpisodes endpoint and return the parsed payload.

    Same retry semantics as src/collector/ratelimit.retry_with_backoff (which
    is not importable from this standalone stdlib-only script): transient
    failures -- HTTP 429/5xx, network errors, timeouts -- are retried up to
    ``max_retries`` times with exponential backoff; any other HTTP error is
    raised immediately. ``sleep`` is injectable so tests run without delays.
    """
    body = json.dumps({"submissionId": submission_id}).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            LIST_EPISODES_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if not _is_retryable_status(e.code):
                raise  # permanent client error (400/401/404/...): fail fast
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e  # network-level failure: treat as transient
        if attempt < max_retries:
            sleep(min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** attempt)))
    assert last_err is not None  # loop always sets it before falling through
    raise last_err


# ---------------------------------------------------------------------------
# pure aggregation / formatting logic (unit-tested, no network)
# ---------------------------------------------------------------------------

def parse_utc(ts: str) -> datetime:
    """Parse Kaggle's RFC-3339 timestamps (variable-width fraction, 'Z')."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1]
    if "." in ts:
        head, frac = ts.split(".", 1)
        ts = head + "." + (frac + "000000")[:6]  # clamp to microseconds
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def classify_reward(reward) -> str:
    """Map an agent reward to W / L / D-ERR (missing reward = draw or error)."""
    if reward == 1:
        return "W"
    if reward == -1:
        return "L"
    return "D/ERR"


def team_names(payload: dict) -> dict[int, str]:
    """teamId -> teamName map from a ListEpisodes payload."""
    return {t["id"]: t.get("teamName", f"team {t['id']}")
            for t in payload.get("teams", [])}


def resolve_team(team_id, names: dict[int, str]) -> str:
    """Team name for an id, falling back to a readable placeholder."""
    return names.get(team_id, f"team {team_id}")


def thin_series(values: list, max_points: int = TRAJECTORY_POINTS) -> list:
    """Evenly subsample a series, always keeping the first and last points."""
    if len(values) <= max_points:
        return list(values)
    step = (len(values) - 1) / (max_points - 1)
    idx = sorted({round(i * step) for i in range(max_points)})
    return [values[i] for i in idx]


def build_report(payload: dict, submission_id: int, *,
                 my_team_id: int = MY_TEAM_ID, recent: int = 8,
                 now: datetime | None = None) -> dict:
    """Aggregate one ListEpisodes payload into a per-submission report dict.

    Only EPISODE_TYPE_PUBLIC episodes count toward score/W-L/trajectory;
    validation episodes are ignored except as a score fallback (baseline 600
    when no public game has been played yet).
    """
    now = now or datetime.now(timezone.utc)
    names = team_names(payload)

    rows = []  # (time, our_agent, opponents, episode_type)
    for ep in payload.get("episodes", []):
        agents = ep.get("agents", [])
        ours = [a for a in agents if a.get("submissionId") == submission_id]
        if not ours:
            continue
        opponents = [a for a in agents if a.get("submissionId") != submission_id]
        # Ratings update when an episode ENDS: concurrent episodes can finish
        # out of createTime order, so the score series must be endTime-ordered
        # (createTime is only a fallback for never-finished episodes).
        ts = ep.get("endTime") or ep["createTime"]
        rows.append((parse_utc(ts), ours[0], opponents, ep.get("type", "")))
    rows.sort(key=lambda r: r[0])

    public = [r for r in rows if r[3] == "EPISODE_TYPE_PUBLIC"]

    wins = sum(1 for _, a, _, _ in public if classify_reward(a.get("reward")) == "W")
    losses = sum(1 for _, a, _, _ in public if classify_reward(a.get("reward")) == "L")
    other = len(public) - wins - losses

    scores = [a.get("updatedScore") for _, a, _, _ in public
              if a.get("updatedScore") is not None]
    score = scores[-1] if scores else None
    if score is None:  # no public games yet -> validation baseline (600)
        fallback = [a.get("updatedScore") for _, a, _, _ in rows
                    if a.get("updatedScore") is not None]
        score = fallback[-1] if fallback else None

    cutoff = now - timedelta(hours=24)
    last24h = sum(1 for t, _, _, _ in public if t >= cutoff)

    recent_games = []
    for t, agent, opponents, _ in public[-recent:][::-1]:  # newest first
        opp_ids = {o.get("teamId") for o in opponents}
        recent_games.append({
            "time_utc": t.strftime("%Y-%m-%d %H:%M"),
            "opponent": " / ".join(resolve_team(i, names) for i in sorted(
                i for i in opp_ids if i is not None)) or "?",
            "result": classify_reward(agent.get("reward")),
            "score": round(agent["updatedScore"], 1)
            if agent.get("updatedScore") is not None else None,
        })

    return {
        "submission_id": submission_id,
        "score": round(score, 1) if score is not None else None,
        "public_episodes": len(public),
        "wins": wins,
        "losses": losses,
        "other": other,  # draws / missing reward (agent error)
        "first_episode_utc": public[0][0].strftime("%Y-%m-%d %H:%M") if public else None,
        "last_episode_utc": public[-1][0].strftime("%Y-%m-%d %H:%M") if public else None,
        "last_24h_games": last24h,
        "trajectory": [round(v, 1) for v in thin_series(scores)],
        "recent_games": recent_games,
    }


def render_report(r: dict) -> str:
    """Human-readable block for one submission report."""
    lines = [f"=== Submission {r['submission_id']} ==="]
    if "error" in r:
        lines.append(f"  ERROR: {r['error']}")
        return "\n".join(lines)
    lines += [
        f"  score:           {r['score']}",
        f"  public episodes: {r['public_episodes']}   "
        f"(W-L: {r['wins']}-{r['losses']}, D/ERR: {r['other']})",
        f"  first episode:   {r['first_episode_utc']} UTC",
        f"  last episode:    {r['last_episode_utc']} UTC",
        f"  last 24h:        {r['last_24h_games']} games",
        f"  trajectory:      " + " -> ".join(str(v) for v in r["trajectory"]),
        f"  recent games:",
    ]
    for g in r["recent_games"]:
        lines.append(f"    {g['time_utc']}  {g['result']:<5}  "
                     f"{g['score']!s:>7}  vs {g['opponent']}")
    return "\n".join(lines)


def render_summary(reports: list[dict]) -> str:
    """Comparison table across all submissions."""
    lines = [
        "=== Summary ===",
        f"{'submission':<12} {'score':>7} {'episodes':>9} {'W-L':>7} "
        f"{'D/ERR':>6}  last activity (UTC)",
    ]
    for r in reports:
        if "error" in r:
            lines.append(f"{r['submission_id']:<12} ERROR: {r['error']}")
            continue
        lines.append(
            f"{r['submission_id']:<12} {r['score']!s:>7} "
            f"{r['public_episodes']:>9} {r['wins']:>3}-{r['losses']:<3} "
            f"{r['other']:>6}  {r['last_episode_utc'] or '-'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subs", default=None,
                    help="comma-separated submission ids "
                         "(default: DEFAULT_SUBMISSIONS)")
    ap.add_argument("--recent", type=int, default=8,
                    help="how many recent games to show per submission")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of text")
    args = ap.parse_args()

    subs = ([int(s) for s in args.subs.split(",") if s.strip()]
            if args.subs else list(DEFAULT_SUBMISSIONS))

    reports = []
    for i, sub in enumerate(subs):
        if i:
            time.sleep(REQUEST_SLEEP_S)
        try:
            payload = fetch_episodes(sub)
            reports.append(build_report(payload, sub, recent=args.recent))
        except Exception as e:  # keep going; report the failure inline
            reports.append({"submission_id": sub, "error": str(e)})

    if args.json:
        print(json.dumps({
            "generated_at_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S"),
            "team_id": MY_TEAM_ID,
            "reports": reports,
        }, ensure_ascii=False, indent=2))
        return

    for r in reports:
        print(render_report(r))
        print()
    print(render_summary(reports))


if __name__ == "__main__":
    main()
