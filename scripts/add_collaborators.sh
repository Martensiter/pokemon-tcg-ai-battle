#!/usr/bin/env bash
# Invite one or more GitHub users as repo collaborators (push permission).
#
#   scripts/add_collaborators.sh USERNAME [USERNAME ...]
#
# Requires the `gh` CLI authenticated with admin on the repo. Permission level is
# overridable via PERMISSION env (pull|triage|push|maintain|admin; default push).
# Kaggle Dataset collaborators are web-UI only -- see docs/ONBOARDING.md.
set -euo pipefail

REPO="${REPO:-Martensiter/pokemon-tcg-ai-battle}"
PERMISSION="${PERMISSION:-push}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 USERNAME [USERNAME ...]" >&2
  echo "  (REPO=$REPO PERMISSION=$PERMISSION)" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found. Install from https://cli.github.com/ or use the web UI." >&2
  exit 1
fi

for user in "$@"; do
  echo ">> inviting '$user' to $REPO as '$PERMISSION' ..."
  gh api -X PUT "/repos/${REPO}/collaborators/${user}" -f permission="${PERMISSION}"
  echo "   invited (pending their acceptance)."
done

echo
echo "current collaborators:"
gh api "/repos/${REPO}/collaborators" --jq '.[].login' || true
