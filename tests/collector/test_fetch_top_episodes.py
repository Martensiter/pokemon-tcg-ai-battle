"""Pure-logic tests for the daily top-episodes fetcher (no network)."""
from __future__ import annotations

import pytest

from tools.fetch_top_episodes import dataset_ref, pick_day, read_index

MANIFEST = """\
date,daily_dataset_slug,daily_dataset_url,episode_count,total_bytes,top_avg_score,median_avg_score
2026-06-17,pokemon-tcg-ai-battle-episodes-2026-06-17,https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-2026-06-17,7819,21474554811,1259.8,761.0
2026-06-16,pokemon-tcg-ai-battle-episodes-2026-06-16,https://example/x,1277,2854565943,1024.6,627.8
2026-07-01,pokemon-tcg-ai-battle-episodes-2026-07-01,https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-01,5266,21472709170,1344.6,1180.3
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


def test_dataset_ref_prefers_manifest_url_owner():
    rows = read_index(MANIFEST)
    # proper URL: owner/slug parsed from .../datasets/<owner>/<slug>
    assert rows[1]["ref"] == "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-17"
    # unparseable URL: falls back to the kaggle/<slug> convention
    assert rows[0]["ref"] == "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-16"
    assert dataset_ref("https://www.kaggle.com/datasets/someorg/some-slug",
                       "ignored") == "someorg/some-slug"
    assert dataset_ref("", "fallback-slug") == "kaggle/fallback-slug"
