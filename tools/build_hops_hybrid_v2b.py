"""hops_hybrid_v2b — long-game consistency tune.

Targets the mirror / late-game deck-out losses. Dudunsparce's Run Away Draw
shuffles the Pokemon back into the deck, but each chain needs a fresh
Dudunsparce in hand. With only 3 copies, the chain runs dry in marathon
games (e.g. our 49-turn loss vs Junichiro Morita).

Change vs v2:
  + 1 Dudunsparce (3 -> 4)
  - 1 Hop's Snorlax (2 -> 1)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck

DECK = [
    (65, 4),    # Dunsparce
    (66, 4),    # Dudunsparce  [3 -> 4]
    (878, 4),   # Hop's Phantump
    (879, 2),   # Hop's Trevenant
    (311, 2),   # Hop's Cramorant
    (304, 1),   # Hop's Snorlax  [2 -> 1]
    (1086, 3),  # Buddy-Buddy Poffin
    (1152, 4),  # Poke Pad
    (1122, 3),  # Pokegear 3.0
    (1115, 3),  # Hop's Bag
    (1171, 4),  # Hop's Choice Band
    (1227, 4),  # Lillie's Determination
    (1182, 3),  # Boss's Orders
    (1225, 2),  # Hilda
    (1194, 2),  # Colress's Tenacity
    (1097, 3),  # Night Stretcher
    (1255, 3),  # Postwick
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
    print(f"hops_hybrid_v2b: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
    if ok:
        out = os.path.join(ROOT, "deck_cand_hops_hybrid_v2b.csv")
        with open(out, "w", newline="") as f:
            f.write("\n".join(str(c) for c in ids) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
