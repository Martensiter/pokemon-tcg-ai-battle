# Pokemon TCG AI Battle Challenge — MCTS Agent + Episode Mining

> **Strategy Category writeup:** see [WRITEUP.md](WRITEUP.md) — full submission text for the Strategy Category competition.
>
> **Simulation Category submission:** `submission_hops_hybrid_v2.tar.gz` (built from this repo via `tools/make_submission.py`).

A determinized Information-Set MCTS agent with a self-trained numpy value network, driven by data mined from 6,533 real Kaggle episode JSONs. Four documented deck pivots took the agent from 600 (initial TrueSkill) to **948 in one hour** on the actual ladder.

## Headline results

| Iteration | Real-ladder score | Notes |
|---|---|---|
| Crustle wall (Pivot 1-2) | 745 | Sim-tested; counters known too widely |
| Lucario+Riolu (Pivot 3) | 604 | Right deck, wrong agent for its tempo |
| hops_hybrid (Pivot 4 baseline) | 948 (1h) | Identified via episode mining of top archetypes |
| **hops_hybrid_v2** (Pivot 4 tuned) | (climbing) | +Cramorant +Hilda +Boss's; 79% h2h vs baseline |

## Architecture

- `agent/mcts.py` — determinized flat-UCB MCTS over the engine's `search_begin`/`search_step` API
- `agent/determinize.py` — samples hidden info (opp deck/hand) consistent with what's visible
- `agent/policy.py` — heuristic action policy for rollouts and multi-select fallback
- `agent/value_net.py` — numpy MLP (32→64→64→1) for leaf evaluation; trained with torch, exported to `weights.npz`
- `agent/agent.py` — orchestrates: returns deck on first call, MCTS otherwise

Inference is **numpy-only** — no torch, no pandas, no network. Verified by [tests/submission_test.py](tests/submission_test.py).

## Repository structure

```
agent/                MCTS agent + value net (ships in submission)
cg/                   Engine SDK + binaries (Pokemon-distributed — NOT IN REPO; see Setup)
selfplay/             Self-play harness, data generation, training, tournaments
tools/                Deck builders, episode scanners, packaging, replay viewer
src/collector/        Production replay collector (24/7 ARM daemon; see below)
tests/                Smoke tests, exec-style submission verification
tests/collector/      Mock unit tests for the collector (no network/engine/torch)
WRITEUP.md            Strategy Category writeup (1813 words)
weights.npz           Trained value-net weights (numpy)
deck_*.csv            60-card decklists by archetype
```

## Setup (cannot run without these)

This repository requires **Pokemon-distributed materials** that are NOT shipped
here due to competition redistribution rules:

1. **Engine SDK + binaries.** Download the `sample_submission/cg/` folder from
   the [competition page](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data).
   Copy `cg.dll` (Windows) and `libcg.so` (Linux) into this repo's `cg/` directory.
   (The `cg/*.py` files in this repo are reproduced from the SDK header for IDE help.)

2. **Card data CSV.** Download `EN_Card_Data.csv` from the competition page and
   place it in the repo root. The agent does not require it at runtime; it is
   used by deck-building and analysis tools only.

3. **Python.** Tested on Python 3.12. Runtime needs only `numpy`. Training and
   tools need `torch` + standard library:
   ```bash
   pip install numpy torch
   ```

## Reproduction

```bash
# 1. Sanity — engine + battle loop + search API
python tests/smoke_test.py
python tests/search_test.py

# 2. Build decks (each writes deck_*.csv)
python tools/build_hops_hybrid_v2.py
python tools/build_anti_meta.py

# 3. Round-robin tournament across multiple decks
python selfplay/round_robin.py --agent mcts -n 10

# 4. Train value net (requires data.npz from selfplay/gen_data.py)
python selfplay/gen_data.py -n 6000 --decks pool --agent greedy
python selfplay/train_value.py --epochs 50

# 5. Package + verify a Kaggle submission
python tools/make_submission.py --deck deck_cand_hops_hybrid_v2.csv --name hops_hybrid_v2
python tests/submission_test.py submission_hops_hybrid_v2

# 6. Mine episode replays (requires daily episode dataset downloaded to ./episodes/)
python tools/scan_episodes.py --top-agents "hiroingk,Kadoraba"
```

## Replay collector (production episode mining)

`tools/scan_episodes.py` and `tools/import_episodes.py` were one-shot, ad-hoc
miners over a folder of pre-downloaded episode JSONs. `src/collector/` replaces
that with a **24/7, idempotent, resumable** collector that pulls real Kaggle
ladder replays via the **official Kaggle CLI only** (no HTML scraping) and emits
records the existing value-net training loop consumes unchanged.

**Input/output asymmetry by design:** submissions are the bottleneck (5/day);
replay *collection* is independent and the more the better — but the replay
endpoint is server-throttled, so the collector self-rate-limits (configurable
RPS + sleep between calls) and backs off exponentially on 429/5xx.

```
src/collector/
  config.py         dataclass config from env vars / .env (.env.example documents all knobs)
  kaggle_client.py  CLI wrapper: leaderboard / submissions / episodes / replay / logs
  ratelimit.py      RateLimiter + exponential backoff (RetryableError/FatalError)
  manifest.py       processed-episode manifest (idempotent, crash-resumable)
  parse.py          defensive replay parsing (never raises on unknown fields)
  convert.py        episode -> (X, y) value-net records, byte-compatible with selfplay/data.npz
  sink.py           abstract Sink + LocalSink + KaggleDatasetSink (GCS-swappable later)
  collector.py      discover -> fetch -> convert -> persist loop
  __main__.py       entry point (python -m collector)
```

### How collected data feeds the existing training

The collector reuses `agent.features.extract` (numpy-only) at each MAIN decision
of a downloaded replay and labels by the episode outcome — the *exact* convention
in `selfplay/gen_data.py` (1.0 win / 0.5 draw / 0.0 loss for the to-move player).
Chunks are written as `data_collected_*.npz` with arrays `X (N, FEATURE_DIM)` and
`y (N,)`, so they drop straight into the offline pipeline:

```bash
# offline training box (torch lives here, NOT on the collector device).
# tools/merge_collected.py bridges the collector's output dir into one dataset
# (selfplay/merge_data.py only globs inside selfplay/); it validates against the
# value net's FEATURE_DIM and is pure-numpy.
python tools/merge_collected.py --src collector_data/value --out selfplay/data_collected_all.npz
python selfplay/train_value.py --data selfplay/data_collected_all.npz --out agent/weights.npz
```

(To blend collected ladder data with existing self-play sets, pass multiple
sources: `--src collector_data/value selfplay/data.npz`.)

### Local run (uv, Python 3.11+)

```bash
uv venv && uv pip install -e ".[dev,kaggle]"
cp .env.example .env            # fill in KAGGLE_USERNAME / KAGGLE_KEY (never commit .env)
uv run python -m collector --dry-run     # print resolved config, no network
uv run python -m collector --self-test   # full pipeline on synthetic data — no creds/network (UAT)
uv run python -m collector --once        # single discovery+collection pass
uv run pytest tests/collector            # mock-only unit tests (no network, no engine, no torch)
```

**For collaborators / UAT:** [docs/UAT.md](docs/UAT.md) is a step-by-step
acceptance runbook (steps 1–3 need no credentials). [docs/ONBOARDING.md](docs/ONBOARDING.md)
covers adding GitHub + Kaggle Dataset collaborators. CI
(`.github/workflows/ci.yml`) runs the tests + offline self-test on every push/PR.

### Deploy on an ARM device (aarch64, no root/apt, nohup)

The collector is pure-Python + numpy — no torch, no engine binary, no native
build — so it runs on a locked-down aarch64 box:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # user-space install
uv venv && uv pip install -e ".[kaggle]"
export KAGGLE_USERNAME=... KAGGLE_KEY=...               # env only
export COLLECTOR_SINK=kaggle DATASET_SLUG=you/ptcg-ladder-replays
nohup uv run python -m collector >> collector.out 2>&1 &
```

It survives restarts (the manifest skips already-fetched episode ids), logs to
stdout **and** a rotating file, and persists progress continuously.

### Kaggle Dataset (canonical store)

Collected data is **never committed to the repo**. The canonical store is a
**private Kaggle Dataset** (collaborator-shared, zero egress from Kaggle
notebooks). With `COLLECTOR_SINK=kaggle`, the collector publishes a new version
each pass via `kaggle datasets version` (small chunks zipped; 1 file ≈ 2 GB cap).
Initialise it once from `collector/dataset-metadata.json` (set its `id` to your
slug): `kaggle datasets create -p collector_data`.

## Key files for the Strategy writeup

| File | Purpose |
|---|---|
| `WRITEUP.md` | Full Strategy Category submission text (1813 words) |
| `agent/mcts.py` | Determinized MCTS — methodology core |
| `agent/value_net.py` | Numpy value net inference |
| `selfplay/train_value.py` | Value-net training pipeline |
| `tools/scan_episodes.py` | Episode-mining tool that produced the tier list |
| `selfplay/round_robin.py` | Deck-tournament harness for pivot decisions |
| `deck_cand_hops_hybrid_v2.csv` | Final primary deck — Dudunsparce + Hop's hybrid, tuned |

## License

[MIT](LICENSE) — applies to all code, decks, weights, and documentation in this
repository. **Pokemon-distributed materials** (engine binary, card data, episode
replays) are NOT covered by this license and are not redistributed here.
