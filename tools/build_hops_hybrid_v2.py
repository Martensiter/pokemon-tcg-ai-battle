"""Hops + Dudunsparce hybrid v2 — tuned for current meta.

Changes vs the v1 baseline (build_hops_hybrid.py):
  + 2 Hop's Cramorant     (chip damage / single-prize utility attacker)
  + 2 Hilda               (search Evolution + Energy — finds Trevenant)
  + 1 Boss's Orders       (more gust)
  - 2 Brock's Scouting    (cut)
  - 1 Buddy-Buddy Poffin  (3 instead of 4)
  - 1 Pokegear 3.0        (3 instead of 4)
  - 1 Postwick            (3 instead of 4)

Theory: more attackers (Cramorant) + Evolution search (Hilda) vs slight
consistency drop on search/draw items. Validated 79% head-to-head vs v1
(n=14) and ~93% vs Lucario, 93% vs Alakazam under MCTS.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck

DECK = [
    # Pokemon (17)
    (65, 4),    # Dunsparce
    (66, 3),    # Dudunsparce
    (878, 4),   # Hop's Phantump
    (879, 2),   # Hop's Trevenant
    (311, 2),   # Hop's Cramorant   [NEW]
    (304, 2),   # Hop's Snorlax
    # Items (17)
    (1086, 3),  # Buddy-Buddy Poffin  [4 -> 3]
    (1152, 4),  # Poke Pad
    (1122, 3),  # Pokegear 3.0        [4 -> 3]
    (1115, 3),  # Hop's Bag
    (1171, 4),  # Hop's Choice Band
    # Supporters (12)
    (1227, 4),  # Lillie's Determination
    (1182, 3),  # Boss's Orders        [2 -> 3]
    (1225, 2),  # Hilda                [NEW]
    (1194, 2),  # Colress's Tenacity
    (1097, 3),  # Night Stretcher
    # Stadium (3)
    (1255, 3),  # Postwick             [4 -> 3]
    # Energy (9)
    (19, 4),    # Telepath Psychic Energy
    (11, 4),    # Mist Energy
    (12, 1),    # Legacy Energy (ACE SPEC)
]


def main():
    db = get_db()
    ids = []
    for cid, n in DECK:
        ids.extend([cid] * n)
    ok, probs = validate_deck(ids)
    ne = sum(1 for c in ids if db.is_energy(c))
    print(f"hops_hybrid_v2 deck: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
    if ok:
        out = os.path.join(ROOT, "deck_cand_hops_hybrid_v2.csv")
        with open(out, "w", newline="") as f:
            f.write("\n".join(str(c) for c in ids) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
