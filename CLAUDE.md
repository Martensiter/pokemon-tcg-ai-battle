See [AGENTS.md](AGENTS.md) for the full guide.

Quick reminders:
- **Collection runs on an ARM device 24/7; training is offline on another box.**
  Improvement is collect → offline re-train → re-submit, never online learning.
- Collector is **pure-Python + numpy only** (no torch, no engine binary), uv +
  Python 3.11+. Official Kaggle CLI only — never HTML-scrape. Rate-limit + back
  off; idempotent + resumable via the manifest.
- **Never commit** raw replays, large data, `.env`, `kaggle.json`, engine binary,
  or card-data CSVs. Canonical store is a private Kaggle Dataset.
- Collected value-net records are `data_collected_*.npz` (`X`, `y`), byte-
  compatible with `selfplay/gen_data.py` output — they feed `merge_data.py` +
  `train_value.py` unchanged.
- Do not rewrite/break `agent/`, the value net, or existing tests. Keep the MIT
  LICENSE and copyright notices.
- Run `uv run pytest tests/collector` (mock-only, no network) before committing.
- Task queue: register incoming tasks with
  `python scripts/claude/manage_tasks.py add "description"` and mark them
  `complete <id>` when done — the SessionStart hook lists pending tasks.
