# UAT runbook — replay collector

For collaborators acceptance-testing the collector. The first three steps need
**no Kaggle credentials, no engine binary, and no torch** — they validate the
whole pipeline offline. The last step is the live ladder run.

## 0. Prerequisites

- Python **3.11+** and [`uv`](https://docs.astral.sh/uv/) (user-space install,
  works on locked-down ARM boxes):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## 1. Install

```bash
git clone https://github.com/Martensiter/pokemon-tcg-ai-battle.git
cd pokemon-tcg-ai-battle
uv venv
uv pip install -e ".[dev,kaggle]"   # drop "kaggle" if you only do offline UAT
```

## 2. Offline acceptance tests (no credentials)

```bash
uv run pytest tests/collector        # expect: all passed
uv run python -m collector --self-test
```

`--self-test` runs one full offline pass on synthetic replays and asserts a
training chunk is produced with the value-net contract. Expected log line:

```
self_test_ok ... rows=24 feature_dim=32 chunk=.../data_collected_*.npz mean_label=0.5
```

✅ **Acceptance criterion:** tests pass and `self_test_ok` is printed with
`feature_dim=32` and `rows>0`.

## 3. Config dry-run

```bash
cp .env.example .env          # fill later for live mode; not needed for dry-run
uv run python -m collector --dry-run
```

Prints the resolved config with secrets redacted. Confirm `competition`,
`sink`, `rps`, and paths look right.

## 4. Live ladder run (needs Kaggle credentials)

```bash
export KAGGLE_USERNAME=...        # or put these in .env
export KAGGLE_KEY=...
uv run python -m collector --once --rps 0.2 --top-n 5
```

Watch for structured log lines: `discovered submissions=...`,
`chunk_written rows=...`, and `pass_complete ...`. Outputs land under
`collector_data/value/` (npz chunks) and `collector_data/meta/`; progress is
tracked in `collector_state/manifest.jsonl` (re-running skips fetched episodes).

✅ **Acceptance criteria:**
- `pass_complete` shows `fetched>0` and `converted_rows>0`;
- a `data_collected_*.npz` exists and loads:
  ```bash
  uv run python -c "import numpy as np,glob; d=np.load(sorted(glob.glob('collector_data/value/*.npz'))[-1]); print(d['X'].shape, d['y'].shape)"
  ```
- re-running `--once` reports `skipped` for already-seen episodes (idempotent).

## 5. Feed the existing training (offline box with torch)

```bash
python selfplay/merge_data.py --glob "data_collected_*.npz" --out selfplay/data_all.npz
python selfplay/train_value.py --data selfplay/data_all.npz --out agent/weights.npz
```

(Copy `collector_data/value/*.npz` next to `selfplay/`, or point `--glob` at the
collector output directory.)

## Notes

- The collector publishes to a **private Kaggle Dataset** when
  `COLLECTOR_SINK=kaggle` + `DATASET_SLUG` are set; that dataset is the canonical
  store collaborators read from inside Kaggle notebooks (zero egress).
- Never commit `.env`, `kaggle.json`, or anything under `collector_data/` /
  `collector_state/` — they are gitignored.
