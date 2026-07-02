"""Pure-logic tests for the daily top-episodes fetcher (no network)."""
from __future__ import annotations

import pytest

from tools.fetch_top_episodes import pick_day, read_index

MANIFEST = """\
date,daily_dataset_slug,daily_dataset_url,episode_count,total_bytes,top_avg_score,median_avg_score
2026-06-17,pokemon-tcg-ai-battle-episodes-2026-06-17,https://example/x,7819,21474554811,1259.8,761.0
2026-06-16,pokemon-tcg-ai-battle-episodes-2026-06-16,https://example/x,1277,2854565943,1024.6,627.8
2026-07-01,pokemon-tcg-ai-battle-episodes-2026-07-01,https://example/x,5266,21472709170,1344.6,1180.3
"""


def test_read_index_parses_and_sorts_by_date():
    rows = read_index(MANIFEST)
    assert [r["date"] for r in rows] == ["2026-06-16", "2026-06-17", "2026-07-01"]
    assert rows[0]["episode_count"] == 1277
    assert rows[0]["total_bytes"] == 2854565943
    assert rows[-1]["slug"] == "pokemon-tcg-ai-battle-episodes-2026-07-01"


def test_pick_day_latest_and_by_date():
    rows = read_index(MANIFEST)
    assert pick_day(rows, latest=True)["date"] == "2026-07-01"
    assert pick_day(rows, date="2026-06-17")["episode_count"] == 7819


def test_pick_day_rejects_unknown_and_missing_selector():
    rows = read_index(MANIFEST)
    with pytest.raises(SystemExit):
        pick_day(rows, date="2026-01-01")
    with pytest.raises(SystemExit):
        pick_day(rows)
    with pytest.raises(SystemExit):
        pick_day([], latest=True)
