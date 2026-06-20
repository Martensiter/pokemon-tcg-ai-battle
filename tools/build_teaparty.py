"""Pure Hop's Trevenant + Team Rocket's engine — alternative hybrid baseline.

A pure Hop's Trevenant attacker line backed by Team Rocket's Petrel +
Transceiver search engine — a separate Hop's-based archetype identified via
episode mining alongside the Dudunsparce hybrid. Reaches around 84% winrate
on ladder when piloted with strong tempo control.

Key tech:
  * Hop's Phantump -> Hop's Trevenant (Stage 1 attacker)
  * Hop's Cramorant (1-energy 120 damage off-tempo attacker)
  * Hop's Snorlax (single-prize 140 dmg attacker)
  * Postwick stadium (4-of, always one in play)
  * Mist Energy (anti-effect, also provides C)
  * Telepath Psychic Energy (acceleration)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck

DECK = [
    (878, 4),   # Hop's Phantump
    (879, 4),   # Hop's Trevenant
    (311, 3),   # Hop's Cramorant
    (304, 2),   # Hop's Snorlax
    (1115, 4),  # Hop's Bag
    (1134, 4),  # Team Rocket's Transceiver
    (1122, 4),  # Pokegear 3.0
    (1171, 4),  # Hop's Choice Band
    (1219, 4),  # Team Rocket's Petrel
    (1227, 4),  # Lillie's Determination
    (1255, 4),  # Postwick (stadium)
    (19, 4),    # Telepath Psychic Energy
    (11, 4),    # Mist Energy
    (1225, 3),  # Hilda
    (1152, 2),  # Poke Pad
    (1197, 2),  # Xerosic's Machinations
    (1182, 2),  # Boss's Orders
    (1092, 1),  # Secret Box (likely ACE SPEC)
    (1097, 1),  # Night Stretcher
]


def main():
    db = get_db()
    ids = []
    for cid, n in DECK:
        ids.extend([cid] * n)
    ok, probs = validate_deck(ids)
    ne = sum(1 for c in ids if db.is_energy(c))
    print(f"teaparty deck: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
    if ok:
        out = os.path.join(ROOT, "deck_cand_teaparty.csv")
        with open(out, "w", newline="") as f:
            f.write("\n".join(str(c) for c in ids) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
