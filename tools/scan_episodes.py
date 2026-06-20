"""Scan every episode JSON in episodes/archive/ and aggregate by agent.

Output for each agent:
  * games played, win rate
  * archetype shorthand for their decks (most-played Pokemon)
  * recurring tech cards
Output overall:
  * tier list of archetypes by play count + win rate
  * top tech cards by adoption rate (% of decks running them)

Fast-path: we only touch info.Agents, steps[1][i].action, and rewards — never
the full game frames. ~21 GB scans in a few minutes on a normal disk.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import time
import argparse
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db  # noqa: E402

DB = get_db()


def deck_archetype(deck: list[int]) -> str:
    """Best-effort archetype label: top 2 non-energy non-trainer cards."""
    pokemon = []
    for cid, n in Counter(deck).most_common():
        c = DB.card(cid)
        if c and c.cardType == 0:  # Pokemon
            pokemon.append((DB.name(cid), n))
            if len(pokemon) == 2:
                break
    if not pokemon:
        return "?"
    return " + ".join(f"{n}x {nm}" for nm, n in pokemon)


def quick_extract(path: str):
    """Pull only the fields we need from one episode JSON."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    agents = [a.get("Name", "?") for a in blob.get("info", {}).get("Agents", [])]
    rewards = blob.get("rewards") or [0, 0]
    steps = blob.get("steps") or []
    decks = []
    if len(steps) > 1:
        for pi in range(2):
            try:
                a = steps[1][pi].get("action")
                if isinstance(a, list) and len(a) == 60:
                    decks.append(a)
                else:
                    decks.append(None)
            except Exception:
                decks.append(None)
    while len(decks) < 2:
        decks.append(None)
    return agents, decks, rewards


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "episodes", "archive"))
    ap.add_argument("--top-agents", default=None,
                    help="comma-separated names to deep-dive (case-insensitive substring)")
    ap.add_argument("--max", type=int, default=None, help="cap for fast iteration")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if args.max:
        files = files[: args.max]
    if not files:
        sys.exit(f"no JSON files in {args.dir}")
    print(f"scanning {len(files)} episodes ...", flush=True)

    # per-agent aggregates
    agent_games = Counter()
    agent_wins = Counter()
    agent_archetypes = defaultdict(Counter)
    agent_tech = defaultdict(Counter)

    # overall aggregates
    archetype_games = Counter()
    archetype_wins = Counter()
    tech_in_decks = Counter()
    decks_seen = 0

    t0 = time.perf_counter()
    failed = 0
    for i, path in enumerate(files):
        if i and i % 500 == 0:
            print(f"  ...{i}/{len(files)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        try:
            agents, decks, rewards = quick_extract(path)
        except Exception:
            failed += 1
            continue
        for pi in range(2):
            if pi >= len(agents) or decks[pi] is None:
                continue
            who = agents[pi]
            deck = decks[pi]
            won = rewards[pi] == 1
            arch = deck_archetype(deck)
            agent_games[who] += 1
            if won:
                agent_wins[who] += 1
            agent_archetypes[who][arch] += 1
            archetype_games[arch] += 1
            if won:
                archetype_wins[arch] += 1
            decks_seen += 1
            # track every Pokemon/Trainer/Tool that appears (skip basic energies 1..9)
            for cid in set(deck):
                if cid > 9:
                    tech_in_decks[cid] += 1
                    agent_tech[who][cid] += 1

    dt = time.perf_counter() - t0
    print(f"\nparsed {len(files)-failed}/{len(files)} files ({failed} failed) in {dt:.1f}s; "
          f"{decks_seen} decks observed\n")

    # === ARCHETYPE TIER LIST ===
    print(f"=== top archetypes by play count (win rate) ===")
    for arch, n in archetype_games.most_common(20):
        wr = archetype_wins[arch] / n if n else 0
        print(f"  {n:>5}x  ({wr:>4.0%})  {arch}")

    # === TECH CARDS ===
    print(f"\n=== top tech cards by adoption rate (% of decks running them) ===")
    for cid, n in tech_in_decks.most_common(20):
        pct = n / decks_seen * 100
        print(f"  {pct:>5.1f}%  {DB.name(cid)} (id {cid})")

    # === AGENTS ===
    print(f"\n=== top agents by games played ===")
    for who, n in agent_games.most_common(25):
        wr = agent_wins[who] / n if n else 0
        top_arch = agent_archetypes[who].most_common(1)[0][0] if agent_archetypes[who] else "?"
        print(f"  {n:>4}g  ({wr:>4.0%})  {who:<28}  main: {top_arch}")

    # === DEEP DIVE ON SPECIFIC AGENTS ===
    if args.top_agents:
        print(f"\n=== DEEP DIVE: {args.top_agents} ===")
        for needle in args.top_agents.split(","):
            needle = needle.strip().lower()
            if not needle:
                continue
            matched = [a for a in agent_games if needle in a.lower()]
            if not matched:
                print(f"\n--- no agent matched '{needle}' ---")
                continue
            for who in matched:
                n = agent_games[who]
                wr = agent_wins[who] / n
                print(f"\n--- {who}  ({n} games, {wr:.0%} winrate) ---")
                print(f"  archetypes played:")
                for arch, k in agent_archetypes[who].most_common(5):
                    print(f"    {k:>3}g  {arch}")
                print(f"  signature tech (top 15 by usage):")
                for cid, k in agent_tech[who].most_common(15):
                    nm = DB.name(cid)
                    print(f"    {k:>3}x  {nm}")


if __name__ == "__main__":
    main()
