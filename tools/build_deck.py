"""Build the agent's deck (mono-Lightning Basic-ex aggro) and write deck.csv.

Archetype rationale: every attacker is a Basic Pokemon ex (no fragile evolution
lines for the search to navigate), single energy type for consistency, and a
heavy trainer engine for draw/search/gust. Counts are auto-completed with Basic
Lightning Energy to reach 60. The deck is validated against the engine and a
real game is run end-to-end.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck, ENERGY_ID_BY_TYPE  # noqa: E402
from cg.api import EnergyType, CardType  # noqa: E402

# (card_id, count). Energy is auto-filled to 60.
POKEMON = [
    (328, 3),   # Pikachu ex      HP190  Thunder LLC 220
    (37, 3),    # Iron Thorns ex  HP230  Volt Cyclone LCC 140 + ability
    (806, 4),   # Rotom ex        HP190  Thunderbolt LC 130 (cheap early attacker)
]
SUPPORTERS = [
    (1227, 4),  # Lillie's Determination — shuffle hand, draw 6
    (1224, 4),  # Cheren — draw 3
    (1182, 4),  # Boss's Orders — gust opponent's benched to active
    (1213, 3),  # Judge — both players draw 4 (disruption)
]
ITEMS = [
    (1121, 4),  # Ultra Ball — search any Pokemon (discard 2)
    (1122, 4),  # Pokegear 3.0 — find a Supporter
    (1097, 4),  # Night Stretcher — recover Pokemon/Energy from discard
    (1119, 4),  # Energy Search — get a Basic Energy
    (1102, 3),  # Dusk Ball — search a Pokemon from the bottom of the deck
    (1126, 1),  # Precious Trolley (ACE SPEC) — put Basic Pokemon onto Bench
]
ENERGY_TYPE = EnergyType.LIGHTNING


def build() -> list[int]:
    ids: list[int] = []
    for cid, n in POKEMON + SUPPORTERS + ITEMS:
        ids.extend([cid] * n)
    energy_id = ENERGY_ID_BY_TYPE[ENERGY_TYPE]
    fill = 60 - len(ids)
    if fill < 0:
        raise ValueError(f"non-energy cards exceed 60 ({len(ids)})")
    ids.extend([energy_id] * fill)
    return ids


def summarize(ids: list[int]):
    db = get_db()
    counts: dict[int, int] = {}
    for c in ids:
        counts[c] = counts.get(c, 0) + 1

    buckets = {"Pokemon": [], "Supporter": [], "Item": [], "Tool": [], "Energy": [], "Stadium": []}
    for cid, n in sorted(counts.items()):
        c = db.card(cid)
        ct = CardType(c.cardType)
        if ct == CardType.POKEMON:
            k = "Pokemon"
        elif ct == CardType.SUPPORTER:
            k = "Supporter"
        elif ct == CardType.ITEM:
            k = "Item"
        elif ct == CardType.TOOL:
            k = "Tool"
        elif ct == CardType.STADIUM:
            k = "Stadium"
        else:
            k = "Energy"
        extra = ""
        if ct == CardType.POKEMON:
            a = db.best_attack(cid)
            extra = f"HP{c.hp} dmg{a.damage if a else 0}"
        buckets[k].append(f"    {n}x  {db.name(cid):<26} (id {cid}) {extra}")

    for k, rows in buckets.items():
        if rows:
            tot = sum(int(r.strip().split('x')[0]) for r in rows)
            print(f"  {k} ({tot}):")
            print("\n".join(rows))


def main():
    ids = build()
    ok, problems = validate_deck(ids)
    print("=== Deck: mono-Lightning Basic-ex aggro ===")
    summarize(ids)
    print(f"\n  total = {len(ids)} cards")
    if not ok:
        print("\nLEGALITY PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("  legality: OK")

    out = os.path.join(ROOT, "deck.csv")
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
        print(f"  GAME FAILED to start: errorPlayer={start.errorPlayer} errorType={start.errorType}")
        sys.exit(1)
    steps = 0
    while True:
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            print(f"  game ran OK: winner={st['result']} steps={steps}")
            break
        sel = obs.get("select")
        if sel is None:
            print(f"  game ended (no select) steps={steps}")
            break
        obs = battle_select(rnd(sel))
        steps += 1
        if steps > 20000:
            print("  game did not terminate"); sys.exit(1)
    battle_finish()


if __name__ == "__main__":
    main()
