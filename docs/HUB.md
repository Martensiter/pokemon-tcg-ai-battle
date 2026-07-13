# Running the collector on the SwitchBot AI Hub (OpenClaw), reset-proof

The Hub's container is **ephemeral**: a reset wipes `/app`, the home dir, and
`~/.local` (uv) — and kills all running processes. Only `/home/node/.openclaw`
survives (a persistent fuseblk mount; `cron/`, `credentials/`, etc. all persist).

So the durable design is:

- **Put everything under the persistent mount** — repo, venv, data, state, and a
  credentials file — at `/home/node/.openclaw/extensions/ptcg-collector/`. These
  survive resets. Run the collector via the venv's python directly (no uv needed
  at run time).
- **Schedule with OpenClaw cron** (`~/.openclaw/cron/jobs.json` persists). Cron
  runs an isolated agent turn that uses the `exec` tool to invoke our launcher.
  After a reset, the persisted cron fires again and collection resumes — no
  babysitting, no long-running daemon to keep alive. The persistent manifest
  makes every pass idempotent.

```
/home/node/.openclaw/extensions/ptcg-collector/
  .env          # credentials + config (OUTSIDE the repo, survives a re-clone)
  repo/         # the cloned repo + .venv (persistent)
  data/         # value/ meta/ raw/ weights/  (COLLECTOR_DATA_DIR)
  state/        # manifest + logs             (COLLECTOR_STATE_DIR)
```

## 1. One-time deploy (needs uv once)

```bash
ROOT=/home/node/.openclaw/extensions/ptcg-collector
mkdir -p "$ROOT"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; . "$HOME/.local/bin/env"; }
git clone https://github.com/Martensiter/pokemon-tcg-ai-battle.git "$ROOT/repo"
cd "$ROOT/repo" && uv venv && uv pip install -e ".[kaggle]"
```

Write the **persistent** credentials/config (outside the repo so a re-clone keeps it):

```bash
cat > "$ROOT/.env" <<EOF
KAGGLE_USERNAME=ichitaro3
KAGGLE_KEY=PASTE_REAL_KEY
DATASET_SLUG=ichitaro3/ptcg-ladder-replays
COLLECTOR_SINK=local
COLLECTOR_KEEP_RAW=true
COLLECTOR_CHUNK_SIZE=20
COLLECTOR_DATA_DIR=$ROOT/data
COLLECTOR_STATE_DIR=$ROOT/state
EOF
chmod 600 "$ROOT/.env"
```

`tools/hub/run.sh` is the launcher: it loads `$ROOT/.env`, runs the venv python,
and self-heals (re-clone/venv) only if the persistent copy is somehow missing.

## 2. Test the launcher manually (before scheduling)

```bash
R=/home/node/.openclaw/extensions/ptcg-collector/repo/tools/hub/run.sh
bash "$R" -m collector --once --rps 0.5          # one collection pass
ls /home/node/.openclaw/extensions/ptcg-collector/data/value/   # chunks appear
bash "$R" tools/daily_pipeline.py --publish      # retrain + publish a Dataset version
```

✅ If `pass_complete ... converted_rows>0` and a `data_collected_*.npz` appears,
the persistent setup works.

## 3. Schedule with OpenClaw cron (persists across resets)

> This Hub sets `tools.exec = {host: "gateway", security: "full"}` in
> `~/.openclaw/openclaw.json`, so exec already runs on the gateway with network
> access — **do NOT ask the model for `elevated=true`**. Boolean tool arguments
> break weaker models (e.g. Llama 4 Scout emits `"elevated": "false"` as a
> string, which Groq's tool validation rejects and the run silently does
> nothing). Keep the message to "command argument only".

```bash
R=/home/node/.openclaw/extensions/ptcg-collector/repo/tools/hub/run.sh
M=nvidia/meta-llama/llama-4-scout-17b-16e-instruct   # see model notes below

# collect every 30 min
openclaw cron add --name "ptcg-collect" --every 30m \
  --session isolated --light-context --tools exec \
  --model "$M" --no-deliver --best-effort-deliver \
  --message "Call the exec tool with ONLY the command argument, no other arguments. command: bash $R -m collector --once --rps 0.5 -- After it finishes, reply with only the last 3 lines of its output."

# retrain + publish daily at 04:00 UTC
openclaw cron add --name "ptcg-daily" --cron "0 4 * * *" --tz UTC \
  --session isolated --light-context --tools exec \
  --model "$M" --no-deliver --best-effort-deliver \
  --message "Call the exec tool with ONLY the command argument, no other arguments. command: bash $R tools/daily_pipeline.py --publish -- After it finishes, reply with only the last 3 lines of its output."

openclaw cron list
```

**Model choice matters (Groq free tier, shared key).** One cron agent turn is
>8k tokens, and Groq enforces per-model tokens-per-minute caps on a single
request: `llama-3.1-8b-instant`=6k, `gpt-oss-120b`=8k, `llama-3.3-70b`=12k
(but its 100k/day pool is burned by other tools on the same key),
`llama-4-scout-17b-16e-instruct`=**30k** — the only one with headroom, hence
`$M` above. A model must also be registered in `~/.openclaw/openclaw.json`
(both `models.providers.<p>.models` and `agents.defaults.models`) or the
per-job `--model` silently falls back to the default; `models.json` is
auto-regenerated, don't edit it.

## 4. Verify / operate

```bash
openclaw cron list
openclaw cron runs --id <job-id>          # run history
tail -n 30 /home/node/.openclaw/extensions/ptcg-collector/state/collector.log
uv run kaggle datasets files ichitaro3/ptcg-ladder-replays   # new versions appear
```

## Notes & troubleshooting

- **After a reset**, do nothing: the persisted cron jobs fire on schedule and
  `run.sh` uses the persistent venv/data. (If the venv itself was lost, `run.sh`
  re-creates it automatically on the next run.)
- **No network in scheduled runs** → check `tools.exec` in `~/.openclaw/openclaw.json`
  is `{host: "gateway", security: "full"}` (don't fix this via `elevated=true` in
  the message — see §3 on why boolean args break weaker models).
- **Job status ok but nothing happens** (summary like "I don't have access to
  exec") → the model failed the tool call. Check the model's tool-calling
  reliability and the rate-limit table in §3.
- **Credentials** live only in `$ROOT/.env` (persistent, `chmod 600`), never in
  git. The collector reads them from the environment that `run.sh` exports.
- **Data durability**: `data/` is on the persistent mount AND published to the
  private Kaggle Dataset each daily run, so it survives even SD-card loss.
- This replaces the `nohup` daemon approach for the Hub. On a normal always-on
  box (Pi/VM) you can still use the daemon + systemd (see DEPLOY.md).
