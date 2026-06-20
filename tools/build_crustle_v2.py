"""Crustle v2 — tuned for the non-ex matchup (its closest matchup).

Changes vs v1 (deck_crustle.csv):
  + 2 Wo-Chien (#201): single-prize Grass attacker (130 for GGC). Lets us trade
    1-for-1 vs non-ex decks instead of feeding the 3-prize Mega Kangaskhan into a
    single-prize race.
  + 1 Spiky Energy (3 -> 4): more counter-chip on the wall.
  - 1 Cheren, 1 Super Potion, 1 basic Grass (room + energy stays at 15).
Everything else (Crustle wall, Mega Kang, Mist/Grow Grass, Hero's Cape ACE) is kept.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from agent.cards import get_db, validate_deck, ENERGY_ID_BY_TYPE  # noqa: E402
from cg.api import EnergyType  # noqa: E402

POKEMON = [(344, 4), (345, 3), (756, 2), (201, 2)]            # +2 Wo-Chien
SUPPORTERS = [(1182, 4), (1213, 3), (1227, 4), (1224, 2)]      # -1 Cheren
ITEMS = [(1086, 4), (1121, 4), (1102, 4), (1097, 3), (1147, 4), (1112, 1), (1159, 1)]  # -1 Super Potion
SPECIAL_ENERGY = [(11, 4), (14, 4), (18, 2)]                   # +1 Spiky
GRASS = EnergyType.GRASS


def build():
    ids = []
    for cid, n in POKEMON + SUPPORTERS + ITEMS + SPECIAL_ENERGY:
        ids.extend([cid] * n)
    ids.extend([ENERGY_ID_BY_TYPE[GRASS]] * (60 - len(ids)))   # 5 basic Grass
    return ids


def main():
    db = get_db()
    ids = build()
    ok, problems = validate_deck(ids)
    n_e = sum(1 for c in ids if db.is_energy(c))
    print("Crustle v2:", "OK" if ok else "ILLEGAL " + str(problems), f"| {len(ids)} cards, energy={n_e}")
    if not ok:
        sys.exit(1)
    out = os.path.join(ROOT, "deck_crustle_v2.csv")
    with open(out, "w", newline="") as f:
        f.write("\n".join(str(c) for c in ids) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
