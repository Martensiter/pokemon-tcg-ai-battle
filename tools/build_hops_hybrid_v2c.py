"""hops_hybrid_v2c — prize-race optimization.

Targets the prize math vs ex-heavy fields and the mirror deck-out problem
simultaneously, without cutting Snorlax (the heavy single-prize closer).

Changes vs v2:
  + 1 Dudunsparce          (3 -> 4)  — longer Run Away Draw chain, anti-deckout
  + 1 Hop's Cramorant      (2 -> 3)  — more 1-prize 120-dmg attackers vs ex decks
  - 1 Colress's Tenacity   (2 -> 1)  — Pokegear + Lillie's cover supporter draw
  - 1 Hop's Bag            (3 -> 2)  — Buddy-Buddy Poffin remains as Basic search

Prize-race rationale: every Cramorant KO'd gives opponent only 1 prize, while
Cramorant's 1-energy 120 damage one-shots Abra (40 HP), Kadabra (80 HP),
and Riolu (80 HP). Snorlax (2x kept) remains the late-game closer.

ACE SPEC: Legacy Energy stays. Powerful Hand "places damage counters" rather
than dealing damage from an attack — defensive ACE SPECs (Survival Brace etc.)
do not trigger against it, so the type-flexible Legacy Energy is correct.
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
    (311, 3),   # Hop's Cramorant  [2 -> 3]
    (304, 2),   # Hop's Snorlax (kept at 2)
    (1086, 3),  # Buddy-Buddy Poffin
    (1152, 4),  # Poke Pad
    (1122, 3),  # Pokegear 3.0
    (1115, 2),  # Hop's Bag  [3 -> 2]
    (1171, 4),  # Hop's Choice Band
    (1227, 4),  # Lillie's Determination
    (1182, 3),  # Boss's Orders
    (1225, 2),  # Hilda
    (1194, 1),  # Colress's Tenacity  [2 -> 1]
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
    print(f"hops_hybrid_v2c: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
    if ok:
        out = os.path.join(ROOT, "deck_cand_hops_hybrid_v2c.csv")
        with open(out, "w", newline="") as f:
            f.write("\n".join(str(c) for c in ids) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
