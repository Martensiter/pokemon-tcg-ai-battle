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

# Never block the unattended cron on a git credential prompt. The repo is public
# (anonymous fetch works, no token needed); if auth were ever required, git fails
# fast and run.sh falls back to the on-disk code instead of hanging forever.
export GIT_TERMINAL_PROMPT=0

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

# --- sync to the validated branch (best-effort; never block a run) -----------
# The Hub repo holds CODE only -- data/state/.env live OUTSIDE $REPO -- so a hard
# reset is safe and idempotent. This is how Mac-validated accuracy changes (new
# features, dims, tools) reach the 24/7 loop without a manual deploy. On network
# failure we keep running the on-disk code. Override branch via PTCG_BRANCH; set
# PTCG_NO_PULL=1 to pin the Hub to its current commit.
# Tracks `main` (the deployable line; changes land via auto-merged PRs, not direct
# pushes) -- override with PTCG_BRANCH to follow a feature branch for testing.
BRANCH="${PTCG_BRANCH:-main}"
if [ "${PTCG_NO_PULL:-0}" != "1" ]; then
  old="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
  if ( cd "$REPO" && git fetch --quiet origin "$BRANCH" \
        && git checkout --quiet -B "$BRANCH" "origin/$BRANCH" ); then
    new="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
    # Reinstall deps only when they actually changed (keeps the common run fast).
    if [ "$old" != "$new" ] && ! git -C "$REPO" diff --quiet "$old" "$new" -- pyproject.toml 2>/dev/null; then
      _need_uv; ( cd "$REPO" && uv pip install -e ".[kaggle]" )
    fi
    [ "$old" != "$new" ] && echo "run.sh: updated $BRANCH ${old:0:7} -> ${new:0:7}" >&2
  else
    echo "run.sh: branch sync skipped (offline?); running on-disk code" >&2
  fi
fi

if [ ! -x "$PY" ]; then
  _need_uv; ( cd "$REPO" && uv venv && uv pip install -e ".[kaggle]" )
fi

# --- credentials/config from the persistent .env (outside the repo) ----------
if [ -f "$ROOT/.env" ]; then
  set -a; . "$ROOT/.env"; set +a
fi

# Put the venv's bin on PATH so the collector's `kaggle` subprocess is found
# (we exec the venv python directly, not via `uv run`).
export PATH="$REPO/.venv/bin:$PATH"

cd "$REPO"
exec "$PY" "$@"
