# Repository guide for coding agents

Fork of `henriquetakahiroito/pokemon-tcg-ai-battle` (MIT) for the Kaggle
competition *The Pokémon Company - PTCG AI Battle Challenge* (engine: cabt).
Keep the **MIT LICENSE and copyright notices intact** in every change.

## Two machines, one loop — input/output are asymmetric

Improvement is an **offline** loop, not online self-evolution:

```
collect real ladder replays  ->  offline re-train value net  ->  re-submit (≤5/day)
        (ARM device, 24/7)          (separate box, torch)         (the bottleneck)
```

- **Collection (input)** is independent of submission and the more the better,
  but the replay endpoint is **server-throttled** — always self-rate-limit and
  back off. Runs on a small **aarch64 ARM device**, 24/7, under `nohup`.
- **Training (output side)** stays **offline on another machine** with torch.
  Do **not** add online/streaming learning.

## Component map

| Path | Role | Touch? |
|---|---|---|
| `agent/` | Submission MCTS + numpy value net (`mcts`, `determinize`, `policy`, `value_net`, `features`, `agent`) | **Do not break / rewrite** |
| `selfplay/` | Self-play harness, `gen_data.py`, `train_value.py` (torch→`weights.npz`), tournaments | Offline; stable contract |
| `tools/` | Deck builders, episode scanners, packaging, replay viewer | Don't break the replay viewer |
| `src/collector/` | **Production replay collector** (this work) | Active area |
| `tests/` | Engine smoke tests + exec-style submission verification | Keep green |
| `tests/collector/` | Mock unit tests (no network/engine/torch) | Keep green |
| `cg/` | Engine SDK + binary — **Pokemon-distributed, NOT in repo** | Never vendor |

## Storage & secrets — hard rules

- **Never commit**: raw replays, large intermediate data, `.env`, `kaggle.json`,
  engine binary, card-data CSVs. Only code + small derived artifacts (and
  `weights.npz`) live in the repo. See `.gitignore`.
- Canonical data store is a **private Kaggle Dataset** (collaborator-shared,
  zero egress from Kaggle notebooks). Collector appends versions via
  `kaggle datasets version`; many small files are zipped (≈2 GB/file cap).
- The storage layer is an **abstract `Sink`** (`LocalSink`, `KaggleDatasetSink`)
  so GCS can be dropped in later. `LocalSink` is what offline training reads.
- All credentials come from the **environment only** (`KAGGLE_USERNAME`,
  `KAGGLE_KEY`); `.env.example` documents every knob. In CI/cloud they are
  injected; with none set, real calls are skipped and tests run on mocks.

## Collector constraints

- **Pure-Python / numpy-only.** No torch, no native builds, no engine binary in
  the collector path — it must run on a root-less aarch64 box. Package manager is
  **uv**, Python **3.11+**.
- **Every network call** has a timeout + retry; replay/log fetches are
  rate-limited (configurable RPS) with exponential backoff on 429/5xx.
- **Idempotent + resumable**: processed episode ids live in a manifest; restarts
  skip them. Logs go to stdout **and** a rotating file.
- **Official Kaggle CLI/API only** — never HTML-scrape (reCAPTCHA).
- **Defensive parsing**: schema is taken from existing code (`agent/`, `tools/`,
  `selfplay/`) and the official cabt docs as canonical; never hard-code a guessed
  schema and never crash on an unknown/missing field.

## Output compatibility (the point of the collector)

Converted records must feed the **existing** training unchanged: value-net chunks
are `data_collected_*.npz` with `X (N, FEATURE_DIM)` + `y (N,)`, labelled exactly
like `selfplay/gen_data.py` (1.0 win / 0.5 draw / 0.0 loss for the to-move
player), produced via `agent.features.extract`. They merge in with
`selfplay/merge_data.py --glob "data_collected_*.npz"` and train via
`selfplay/train_value.py`.

## Before committing

- No secret or raw data staged (`git status` — `.env`, `kaggle.json`,
  `collector_data/`, `collector_state/` are gitignored).
- `uv run pytest tests/collector` green; existing tests untouched.
- Type hints + docstrings on new code; config via dataclass + env vars.
