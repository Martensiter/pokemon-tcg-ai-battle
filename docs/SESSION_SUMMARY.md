# Session summary — production replay collector, built and deployed

## In one line
For the Kaggle *Pokémon TCG AI Battle* competition, we built — from scratch — a
system that **24/7 auto-collects real ladder replays, converts them into value-net
training data, and retrains a candidate model daily**, and we got it **running on a
real device (SwitchBot AI Hub)**.

## What was built (all merged to `main`, ~17 PRs, 83 tests green)
- **Collector** (`src/collector/`): official Kaggle CLI wrapper (leaderboard →
  submissions → episodes → replay → logs), self rate-limit + exponential backoff,
  idempotent/resumable manifest, defensive parsing of the real cabt (Kaggle env)
  replay shape, conversion to value-net records.
- **Output is value-net-compatible** (`data_collected_*.npz` with `X=(N, 32)`,
  `y`) — drops straight into the existing training. `tools/merge_collected.py`
  bridges the collector output into one dataset.
- **Torch-free training** (`selfplay/train_value_np.py`): retrains the value net
  in numpy only — so the whole loop can run on the ARM Hub (no torch).
- **Daily pipeline** (`tools/daily_pipeline.py`): merge → retrain → publish a
  candidate `weights.npz` to a private Kaggle Dataset.
- **Storage**: abstract Sink (Local + private Kaggle Dataset). No data in git.
- **Deploy & ops**: Docker (multi-arch), native uv, **reset-proof SwitchBot AI
  Hub** (`tools/hub/run.sh`, `docs/HUB.md`), an OpenClaw control API (`--serve`:
  status / collect), offline `--self-test`, CI, and docs (UAT / ONBOARDING /
  DEPLOY / HUB).

## What is actually running (live)
- ✅ Real ladder replays collected: **51 episodes → 5,642 training states + 51 raw
  replays**.
- ✅ **Private Kaggle Dataset** `ichitaro3/ptcg-ladder-replays` — shared with a
  collaborator, auto-versioned (data + candidate weights).
- ✅ **Running on the SwitchBot AI Hub** (aarch64). The container is ephemeral, but
  data / manifest / venv / cron all live on the persistent mount, and **OpenClaw
  cron auto-runs collection every 30 min + a daily retrain — surviving resets**.
- ✅ Remote status / on-demand collection via OpenClaw.

## The capability this unlocks (competition view)
A fully-automated **data flywheel**:

```
real ladder games → auto-collect → training data → daily retrain → (human-gated) resubmit
```

- **Kaggle Dataset** = the shared warehouse of data + candidate models (refreshed daily).
- **Agent** = the player that actually competes; you pick the best weights, package, and submit it.
- Strength verification + competition submission stay human-gated (they need the
  engine binary, and you don't want to auto-submit a regression).

## Hard problems solved on the way (real-device reality)
Identifying the real replay JSON shape; episode-id typing; replay file-download
behavior; the kaggle-CLI PATH; the Hub's ephemeral container (only
`/home/node/.openclaw` persists); OpenClaw's scope / bootstrap approval gates;
and driving collection through OpenClaw cron + exec.

## Honest "what's left"
- The improvement loop hasn't completed a full cycle yet (accumulate more data →
  retrain → verify → resubmit is still ahead, and human-gated).
- Verification + submission need the engine binary (off the Hub / manual).
- Minor: set the cron jobs' delivery to `none` (cosmetic; collection already works).

## Where things live
- Code: `github.com/Martensiter/pokemon-tcg-ai-battle` (branch `main`)
- Data: Kaggle **private** dataset `ichitaro3/ptcg-ladder-replays`
- On-device: `/home/node/.openclaw/extensions/ptcg-collector/` (Hub, persistent)
