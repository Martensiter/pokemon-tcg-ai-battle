"""Build a small gauntlet of meta opponent decks for stress-testing Crustle.

Key point: Crustle #345 prevents damage from ALL Pokemon ex attacks regardless of
type, so any pure-ex deck is a bad test (Crustle walls it). The gauntlet therefore
spans the spectrum:
  * fire_ex   — pure ex aggro (Crustle's prey; sanity check)
  * nonex     — single-prize non-ex attackers that IGNORE the wall (Crustle's bane)
  * mixed     — realistic meta build: strong ex + a non-ex tech attacker
Each uses a shared, legal good-stuff trainer engine and mono-energy for consistency.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck, ENERGY_ID_BY_TYPE  # noqa: E402
from cg.api import EnergyType, CardType  # noqa: E402

# Shared trainer engine (no ACE SPEC -> safe to reuse across decks). 30 cards.
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

DECKS = {
    # all-ex Fire aggro: big damage, but every attacker is an ex -> Crustle walls it
    "fire_ex": dict(pokemon=[(46, 4), (259, 4), (357, 2)], energy=EnergyType.FIRE),
    # non-ex single-prize toolbox: ignores Crustle's ability entirely
    "nonex": dict(pokemon=[(953, 4), (304, 4), (175, 3)], energy=EnergyType.LIGHTNING),
    # realistic mixed: Lightning ex core + non-ex Zapdos to punch through walls
    "mixed": dict(pokemon=[(328, 3), (37, 2), (953, 4)], energy=EnergyType.LIGHTNING),
    # HARD: Alakazam — non-ex Stage 2; Powerful Hand = 1 energy, scales with hand size
    "alakazam": dict(pokemon=[(741, 4), (742, 1), (743, 3)], energy=EnergyType.PSYCHIC,
                     extra=[(1079, 4), (1086, 4)]),  # Rare Candy + Buddy-Buddy Poffin (Abra 50HP)
    # HARD: Team Rocket's Spidops — non-ex board-scaling single-prize attacker
    "rocket_spidops": dict(pokemon=[(400, 4), (401, 4)], energy=EnergyType.GRASS,
                           extra=[(1132, 4)]),  # Team Rocket's Great Ball
}


def build(pokemon, energy_type, extra=None):
    ids = []
    for cid, n in pokemon + ENGINE + (extra or []):
        ids.extend([cid] * n)
    ids.extend([ENERGY_ID_BY_TYPE[energy_type]] * (60 - len(ids)))
    return ids


def main():
    db = get_db()
    for name, spec in DECKS.items():
        ids = build(spec["pokemon"], spec["energy"], spec.get("extra"))
        ok, problems = validate_deck(ids)
        n_energy = sum(1 for c in ids if db.is_energy(c))
        pkmn = ", ".join(f"{n}x {db.name(c)}" for c, n in spec["pokemon"])
        status = "OK" if ok else "ILLEGAL: " + "; ".join(problems)
        print(f"[{name:<8}] {pkmn}  | energy={n_energy} {spec['energy'].name}  -> {status}")
        if ok:
            out = os.path.join(ROOT, f"deck_meta_{name}.csv")
            with open(out, "w", newline="") as f:
                f.write("\n".join(str(c) for c in ids) + "\n")


if __name__ == "__main__":
    main()
