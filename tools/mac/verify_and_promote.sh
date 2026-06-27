#!/usr/bin/env bash
# Mac engine-machine automation: verify the latest published candidate value-net
# against the current champion (agent/weights.npz) and PROMOTE it if it wins by
# the gate. This is the verification step of the improve loop -- the one piece
# that can't run on the aarch64 Hub (it needs the native engine).
#
# It NEVER submits. Submission stays a human step (Kaggle's `competitions submit`
# requires OAuth and you don't want to auto-ship a regression) -- see
# docs/SUBMIT_BASELINE.md.
#
# SELF-GATING: it no-ops cheaply (a few seconds) until enough states are
# collected, so it is safe to schedule NOW and only burns the multi-hour A/B
# self-play once the data is actually large enough that a candidate might pass.
#
# Install (on the Mac): see docs/MAC_AUTOMATION.md.
#
# Tunables (env or .env):
#   REPO         repo path                (default: ~/Downloads/AI/pokemon-tcg-ai-battle)
#   DATASET_SLUG kaggle dataset           (default: ichitaro3/ptcg-ladder-replays)
#   MIN_STATES   skip verify below this    (default: 30000)
#   GAMES        games per deck (10 decks) (default: 30  -> ~6h; 76s/game)
#   THRESHOLD    min candidate win rate    (default: 0.53)
set -euo pipefail

REPO="${REPO:-$HOME/Downloads/AI/pokemon-tcg-ai-battle}"
cd "$REPO"

# Load the same .env the collector uses (KAGGLE_USERNAME/KEY, DATASET_SLUG, ...).
if [ -f .env ]; then set -a; . ./.env; set +a; fi

DATASET="${DATASET_SLUG:-ichitaro3/ptcg-ladder-replays}"
MIN_STATES="${MIN_STATES:-30000}"
GAMES="${GAMES:-30}"
THRESHOLD="${THRESHOLD:-0.53}"

PY="${PY:-.venv/bin/python}"
KAG="${KAG:-.venv/bin/kaggle}"
LOG="${LOG:-verify_cron.log}"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

say "=== verify cron start (min_states=$MIN_STATES games=$GAMES threshold=$THRESHOLD) ==="

# 1) Pull the latest dataset (data + published candidate weights). Datasets API
#    works with legacy auth; keep kaggle<1.7 on this machine (see HANDOFF_MACBOOK).
rm -rf ds && mkdir ds
if ! "$KAG" datasets download -d "$DATASET" -p ds --unzip >>"$LOG" 2>&1; then
  say "dataset download failed -> skip"; exit 0
fi

CAND="ds/weights/weights_candidate.npz"
if [ ! -f "$CAND" ]; then say "no candidate weights in dataset yet -> skip"; exit 0; fi

# 2) Cheap self-gate: count collected states; skip the expensive A/B until ready.
STATES=$("$PY" - <<'PYEOF'
import glob, numpy as np
n = 0
for f in glob.glob("ds/value/data_collected_*.npz"):
    try:
        n += int(len(np.load(f)["y"]))
    except Exception:
        pass
print(n)
PYEOF
)
say "dataset states=$STATES"
if [ "$STATES" -lt "$MIN_STATES" ]; then
  say "insufficient data ($STATES < $MIN_STATES) -> skip verify (no CPU burned)"; exit 0
fi

# 3) A/B self-play (hours). --promote overwrites agent/weights.npz iff it passes.
say "running A/B verify (candidate vs champion) -- this can take hours"
if "$PY" tools/verify_candidate.py --new "$CAND" --games "$GAMES" \
        --threshold "$THRESHOLD" --promote >>"$LOG" 2>&1; then
  say "RESULT: PASS -> promoted to agent/weights.npz."
  say "NEXT (human): package + submit -> docs/SUBMIT_BASELINE.md (loop does NOT auto-submit)"
else
  say "RESULT: FAIL/insufficient -> champion unchanged (this is normal until data grows)"
fi
say "=== verify cron done ==="
