"""Hops + Dudunsparce hybrid archetype — episode-mining baseline.

Identified through episode mining: a Dudunsparce draw engine paired with Hop's
evolution attackers (Phantump/Trevenant/Snorlax) over a Mist + Telepath +
Legacy Energy special energy package. Setup is fast (Stage 1 only, no Rare
Candy needed) and Dudunsparce's draw-3 ability provides absurd card advantage.

This file is the baseline; see build_hops_hybrid_v2.py for the tuned version.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck
DECK = [
    (65, 4),    # Dunsparce
    (66, 3),    # Dudunsparce
    (878, 4),   # Hop's Phantump
    (879, 2),   # Hop's Trevenant
    (304, 2),   # Hop's Snorlax
    (1122, 4),  # Pokegear 3.0
    (1171, 4),  # Hop's Choice Band
    (1152, 4),  # Poke Pad
    (1086, 4),  # Buddy-Buddy Poffin
    (1097, 3),  # Night Stretcher
    (1115, 3),  # Hop's Bag
    (1210, 2),  # Brock's Scouting
    (1227, 4),  # Lillie's Determination
    (1182, 2),  # Boss's Orders
    (1194, 2),  # Colress's Tenacity
    (1255, 4),  # Postwick (stadium)
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
    print(f"hops_hybrid deck: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
    if ok:
        out = os.path.join(ROOT, "deck_cand_hops_hybrid.csv")
        with open(out, "w", newline="") as f:
            f.write("\n".join(str(c) for c in ids) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
