#!/usr/bin/env bash
# Persistent-mount launcher for the collector on the SwitchBot AI Hub (OpenClaw).
#
# Everything lives under /home/node/.openclaw  -- a persistent fuseblk mount that
# survives container resets. Only running processes + ephemeral /app and ~/.local
# (uv) are lost on a reset, so we run the collector via the venv's python directly
# (no uv needed at run time) and let an OpenClaw cron job re-invoke this script.
#
#   bash run.sh -m collector --once --rps 0.5        # one collection pass
#   bash run.sh tools/daily_pipeline.py --publish    # daily retrain + publish
#
# Config/credentials live OUTSIDE the repo at $ROOT/.env so they survive even a
# full re-clone. Self-heals (clone + venv) only if the persistent copy is missing.
set -euo pipefail

ROOT="/home/node/.openclaw/extensions/ptcg-collector"
REPO="$ROOT/repo"
PY="$REPO/.venv/bin/python"
REPO_URL="https://github.com/Martensiter/pokemon-tcg-ai-battle.git"

mkdir -p "$ROOT"

# --- self-heal (normally a no-op: the persistent mount keeps these) ----------
_need_uv() { command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }; }
if [ ! -d "$REPO/.git" ]; then
  _need_uv; git clone "$REPO_URL" "$REPO"
fi
if [ ! -x "$PY" ]; then
  _need_uv; ( cd "$REPO" && uv venv && uv pip install -e ".[kaggle]" )
fi

# --- credentials/config from the persistent .env (outside the repo) ----------
if [ -f "$ROOT/.env" ]; then
  set -a; . "$ROOT/.env"; set +a
fi

cd "$REPO"
exec "$PY" "$@"
