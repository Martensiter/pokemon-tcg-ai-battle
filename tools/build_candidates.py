"""Build new candidate archetypes to search for a new best deck.

Each is Basic-attacker-centric (AI-pilotable) over the shared trainer engine.
We deliberately span the strategy space — a darkness ex beatdown, a psychic
ex+non-ex toolbox (non-ex bodies can break the Crustle wall), and a fighting
ex line — then a round-robin (round_robin.py) ranks them against our existing
decks + the official samples.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck, ENERGY_ID_BY_TYPE  # noqa: E402
from cg.api import EnergyType  # noqa: E402

ENGINE = [
    (1121, 4),  # Ultra Ball
    (1102, 4),  # Dusk Ball
    (1182, 4),  # Boss's Orders
    (1227, 4),  # Lillie's Determination
    (1224, 4),  # Cheren
    (1213, 3),  # Judge
    (1097, 3),  # Night Stretcher
    (1112, 2),  # Super Potion
    (1119, 2),  # Energy Search
]

CANDIDATES = {
    # Darkness ex beatdown: Yveltal ex (Basic, 210) + Okidogi ex + Munkidori snipe
    "dark_yveltal": dict(pokemon=[(1062, 4), (138, 2), (112, 2)], energy=EnergyType.DARKNESS),
    # Psychic toolbox: Latias ex (200) + non-ex Mesprit / Iron Boulder that ignore walls
    "psy_latias": dict(pokemon=[(184, 4), (216, 3), (971, 2)], energy=EnergyType.PSYCHIC),
    # Fighting ex line: Koraidon ex (200) + non-ex Koraidon + Stonjourner
    "fight_koraidon": dict(pokemon=[(979, 4), (62, 3), (682, 2)], energy=EnergyType.FIGHTING),
}


def build(pokemon, energy_type):
    ids = []
    for cid, n in pokemon + ENGINE:
        ids.extend([cid] * n)
    ids.extend([ENERGY_ID_BY_TYPE[energy_type]] * (60 - len(ids)))
    return ids


def main():
    db = get_db()
    for name, spec in CANDIDATES.items():
        ids = build(spec["pokemon"], spec["energy"])
        ok, problems = validate_deck(ids)
        ne = sum(1 for c in ids if db.is_energy(c))
        pk = ", ".join(f"{n}x {db.name(c)}" for c, n in spec["pokemon"])
        print(f"[{name:<14}] {pk}  | energy={ne} {spec['energy'].name}  -> {'OK' if ok else 'ILLEGAL '+str(problems)}")
        if ok:
            with open(os.path.join(ROOT, f"deck_cand_{name}.csv"), "w", newline="") as f:
                f.write("\n".join(str(c) for c in ids) + "\n")


if __name__ == "__main__":
    main()
