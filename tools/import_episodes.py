"""Auto-detect and summarize Kaggle episode replay JSONs dropped in ./episodes/.

Handles the two formats cabt-viewer's adapter describes:
  * Kaggle environment context: top-level dict with `steps` (list of [obs0, obs1] pairs).
  * Lower-level local runner JSON: top-level dict with a `visualize` array.

For each file we print:
  * winner, turn count, total decisions, file path
  * starting decks (60 card ids per player) — extracted from the very first agent
    submission, which in CABT is the 60-card deck list per main.py contract
  * which agent (player 0 / 1) won and by what reason

This is enough to mine which decks top agents are running. If your dataset has
hundreds of files, pass a glob to focus on one team's submissions.

  python tools/import_episodes.py                       # summarize everything in episodes/
  python tools/import_episodes.py --glob "*53802029*"   # only that submission id
  python tools/import_episodes.py --decks-out top_decks.csv  # also dump deck lists
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db  # noqa: E402

DB = get_db()


def _frames(blob):
    """Yield the list of CABT visualize frames regardless of which wrapper
    format Kaggle used."""
    if isinstance(blob, dict):
        if "visualize" in blob:
            return blob["visualize"]
        if "steps" in blob:
            # Kaggle environment shape: steps[i] is a list of agent obs dicts.
            # The CABT visualize array is embedded at steps[0][0].
            try:
                first = blob["steps"][0][0]
                if isinstance(first, dict) and "observation" in first:
                    obs = first["observation"]
                    if isinstance(obs, dict) and "visualize" in obs:
                        return obs["visualize"]
                    if isinstance(obs, list):
                        return obs
            except Exception:
                pass
    if isinstance(blob, list):
        return blob
    return []


def _deck_from_action(action):
    """A CABT first action for a player is the 60 card-id list (their deck)."""
    if isinstance(action, list) and len(action) == 60 and all(isinstance(x, int) for x in action):
        return list(action)
    return None


def _extract_decks(blob):
    """Best-effort: find each player's submitted deck in the early actions."""
    decks = {}
    if isinstance(blob, dict) and "steps" in blob:
        # steps[0] is the initial state; steps[1][i].action holds player i's first action.
        for step_i in range(min(3, len(blob["steps"]))):
            for pi, agent_state in enumerate(blob["steps"][step_i] or []):
                if not isinstance(agent_state, dict):
                    continue
                d = _deck_from_action(agent_state.get("action"))
                if d and pi not in decks:
                    decks[pi] = d
    return decks


def _winner_and_reason(frames):
    """Walk the visualize frames to find the RESULT log."""
    for f in frames:
        for lg in (f.get("logs") or []):
            # LogType.RESULT == 23; tolerate either ints or names
            if lg.get("type") in (23, "Result"):
                return lg.get("result"), lg.get("reason")
    # Fallback: final state's result field
    if frames:
        cur = (frames[-1] or {}).get("current") or {}
        return cur.get("result", -1), None
    return -1, None


def _final_turn(frames):
    for f in reversed(frames):
        cur = f.get("current") or {}
        t = cur.get("turn")
        if t:
            return t
    return 0


def _top_pokemon(deck):
    counter = Counter(deck)
    return [(n, DB.name(c)) for c, n in counter.most_common(4) if DB.card(c) and DB.card(c).cardType == 0]


def summarize(path):
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    frames = _frames(blob)
    if not frames:
        return {"path": path, "ok": False, "error": "no visualize frames"}
    decks = _extract_decks(blob)
    winner, reason = _winner_and_reason(frames)
    return {
        "path": path,
        "ok": True,
        "frames": len(frames),
        "turns": _final_turn(frames),
        "winner": winner,
        "reason": reason,
        "decks": decks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "episodes"))
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--decks-out", default=None,
                    help="optional CSV: one row per (file,player) with the 60-card deck")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "**", args.glob), recursive=True))
    if not files:
        print(f"no JSON files found under {args.dir} matching {args.glob}")
        print(f"\nDrop your Kaggle episode JSONs into:  {args.dir}")
        print("Then rerun this command.")
        return

    print(f"scanning {len(files)} files in {args.dir}")
    deck_rows = []
    wins = Counter()
    archetypes = Counter()
    for path in files:
        try:
            s = summarize(path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue
        if not s["ok"]:
            print(f"  [skip] {os.path.basename(path)}: {s['error']}")
            continue
        wins[s["winner"]] += 1
        for pi, deck in s["decks"].items():
            deck_rows.append((path, pi, deck))
            for n, nm in _top_pokemon(deck):
                archetypes[nm] += n

    print(f"\n=== overall ===")
    print(f"games parsed: {sum(wins.values())}  | winner counts: P0={wins.get(0,0)} P1={wins.get(1,0)} draws={wins.get(2,0)}")
    print(f"\n=== most-seen attackers (top 15) ===")
    for nm, n in archetypes.most_common(15):
        print(f"  {n:>4}x  {nm}")

    if args.decks_out:
        with open(args.decks_out, "w", encoding="utf-8") as f:
            f.write("file,player,deck_ids\n")
            for path, pi, deck in deck_rows:
                f.write(f"{os.path.basename(path)},{pi},{'|'.join(map(str,deck))}\n")
        print(f"\nwrote {args.decks_out} ({len(deck_rows)} decks)")


if __name__ == "__main__":
    main()
