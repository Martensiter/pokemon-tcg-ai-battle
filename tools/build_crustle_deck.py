"""Build the Crustle anti-ex wall/grind deck -> deck_crustle.csv.

Concept: Crustle #345 (Mysterious Rock Inn) takes zero damage from opponent
Pokemon ex attacks -> an (almost) unkillable single-prize wall in an ex-dominated
field. Hero's Cape (ACE SPEC, +100 HP) makes it a 250 HP wall. Mega Kangaskhan ex
(Basic, 300 HP, colorless) adds a draw engine (Run Errand) + a real finisher, on
the same Grass energy. Healing (Jumbo Ice Cream / Super Potion) erases chip damage
from the non-ex attackers the ability can't block; Boss's Orders + Judge supply
the gust-and-grind win condition.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck, ENERGY_ID_BY_TYPE  # noqa: E402
from cg.api import EnergyType, CardType  # noqa: E402

# (card_id, count). Energy auto-fills to 60.
POKEMON = [
    (344, 4),   # Dwebble        70 HP Basic (Buddy-Buddy Poffin target; Ascension self-evolves)
    (345, 3),   # Crustle        150 HP Stage1 - immune to ex attack damage; Superb Scissors 120
    (756, 2),   # Mega Kangaskhan ex  300 HP Basic - Run Errand draw + Rapid-Fire Combo 200+
]
SUPPORTERS = [
    (1182, 4),  # Boss's Orders — gust (win condition)
    (1213, 3),  # Judge — hand disruption
    (1227, 4),  # Lillie's Determination — shuffle + draw 6
    (1224, 3),  # Cheren — draw 3
]
ITEMS = [
    (1086, 4),  # Buddy-Buddy Poffin — search Basics <=70 HP (Dwebble!)
    (1121, 4),  # Ultra Ball — search any Pokemon
    (1102, 4),  # Dusk Ball — search a Pokemon
    (1097, 3),  # Night Stretcher — recover Pokemon / Energy
    (1147, 4),  # Jumbo Ice Cream — heal 80 (active with 3+ Energy: Crustle/Mega Kang qualify)
    (1112, 2),  # Super Potion — heal 60
    (1159, 1),  # Hero's Cape (ACE SPEC) — +100 HP -> 250 HP Crustle wall
]
# Special energy package (all provide {C}, so they also fuel Mega Kangaskhan CCC):
SPECIAL_ENERGY = [
    (11, 4),    # Mist Energy — prevents all attack EFFECTS on the holder (anti-status/gust)
    (14, 3),    # Spiky Energy — attacker takes 20 back when it damages the holder
    (18, 2),    # Grow Grass Energy — provides {G} and +20 HP (Crustle's Grass source)
]
ENERGY_TYPE = EnergyType.GRASS  # remaining slots filled with basic Grass


def build() -> list[int]:
    ids: list[int] = []
    for cid, n in POKEMON + SUPPORTERS + ITEMS + SPECIAL_ENERGY:
        ids.extend([cid] * n)
    fill = 60 - len(ids)
    if fill < 0:
        raise ValueError(f"non-energy cards exceed 60 ({len(ids)})")
    ids.extend([ENERGY_ID_BY_TYPE[ENERGY_TYPE]] * fill)
    return ids


def summarize(ids):
    db = get_db()
    counts = {}
    for c in ids:
        counts[c] = counts.get(c, 0) + 1
    order = {"Pokemon": 0, "Supporter": 1, "Item": 2, "Tool": 3, "Stadium": 4, "Energy": 5}
    rows = {k: [] for k in order}
    for cid, n in sorted(counts.items()):
        c = db.card(cid)
        ct = CardType(c.cardType)
        k = {CardType.POKEMON: "Pokemon", CardType.SUPPORTER: "Supporter",
             CardType.ITEM: "Item", CardType.TOOL: "Tool",
             CardType.STADIUM: "Stadium"}.get(ct, "Energy")
        tag = " (ACE SPEC)" if c.aceSpec else ""
        extra = f"HP{c.hp}" if ct == CardType.POKEMON else ""
        rows[k].append(f"    {n}x  {db.name(cid):<24} (id {cid}) {extra}{tag}")
    for k in order:
        if rows[k]:
            tot = sum(int(r.strip().split('x')[0]) for r in rows[k])
            print(f"  {k} ({tot}):")
            print("\n".join(rows[k]))


def main():
    ids = build()
    ok, problems = validate_deck(ids)
    print("=== Deck: Crustle anti-ex wall/grind (mono-Grass) ===")
    summarize(ids)
    print(f"\n  total = {len(ids)} cards")
    if not ok:
        print("\nLEGALITY PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("  legality: OK")

    out = os.path.join(ROOT, "deck_crustle.csv")
    with open(out, "w", newline="") as f:
        f.write("\n".join(str(c) for c in ids) + "\n")
    print(f"  wrote {out}")

    # Smoke: a full game must run with this deck on both seats.
    import random
    from cg.game import battle_start, battle_select, battle_finish

    def rnd(sel):
        n = len(sel["option"]); lo, hi = sel["minCount"], min(sel["maxCount"], n)
        if hi <= 0:
            return []
        k = random.randint(lo, hi) if hi >= lo else lo
        return random.sample(range(n), max(0, min(k, n)))

    obs, start = battle_start(ids, list(ids))
    if obs is None:
        print(f"  GAME FAILED: errorPlayer={start.errorPlayer} errorType={start.errorType}")
        sys.exit(1)
    steps = 0
    while True:
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            print(f"  game ran OK: winner={st['result']} steps={steps}")
            break
        sel = obs.get("select")
        if sel is None:
            break
        obs = battle_select(rnd(sel))
        steps += 1
        if steps > 20000:
            print("  did not terminate"); sys.exit(1)
    battle_finish()


if __name__ == "__main__":
    main()
