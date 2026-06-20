"""Explore the engine card DB to inform deck building.

Lists strong Basic Pokemon ex attackers (HP, best attack, energy cost, type),
plus the supporter / item / energy pools, joining engine data with CSV names.
"""
import os
import sys
import csv
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cg.api import all_card_data, all_attack, CardType, EnergyType  # noqa: E402


def load_names():
    names = {}
    with open(os.path.join(ROOT, "EN_Card_Data.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                names[int(row["Card ID"])] = row
            except (ValueError, KeyError):
                pass
    return names


def main():
    cards = {c.cardId: c for c in all_card_data()}
    attacks = {a.attackId: a for a in all_attack()}
    names = load_names()

    def nm(cid):
        c = cards.get(cid)
        if c and c.name:
            return c.name
        r = names.get(cid)
        return r["Card Name"] if r else f"#{cid}"

    def best_attack(c):
        best = None
        for aid in c.attacks:
            a = attacks.get(aid)
            if a is None:
                continue
            if best is None or a.damage > best.damage:
                best = a
        return best

    # ---- Basic Pokemon ex attackers ----
    print("=== Top Basic Pokemon ex by best attack damage ===")
    rows = []
    for cid, c in cards.items():
        if not c.basic or not c.ex or c.megaEx:
            continue
        a = best_attack(c)
        if a is None or a.damage <= 0:
            continue
        rows.append((a.damage, c.hp, len(a.energies), cid, c, a))
    rows.sort(key=lambda r: (-r[0], r[2], -r[1]))
    for dmg, hp, ne, cid, c, a in rows[:30]:
        etype = EnergyType(c.energyType).name
        ecost = "".join(EnergyType(e).name[0] for e in a.energies) or "-"
        print(f"  id={cid:<5} {nm(cid)[:26]:<26} HP{hp:<4} {etype:<9} "
              f"atk='{a.name[:18]:<18}' dmg={dmg:<4} cost[{ne}]={ecost}")

    # ---- efficient low-cost Basic ex (<=2 energy, good damage) ----
    print("\n=== Efficient Basic ex (<=2 energy, dmg>=120) ===")
    eff = [r for r in rows if r[2] <= 2 and r[0] >= 120]
    eff.sort(key=lambda r: (r[2], -r[0]))
    for dmg, hp, ne, cid, c, a in eff[:25]:
        etype = EnergyType(c.energyType).name
        print(f"  id={cid:<5} {nm(cid)[:26]:<26} HP{hp:<4} {etype:<9} dmg={dmg:<4} ne={ne} '{a.name[:18]}'")

    # ---- type distribution of strong basic ex ----
    print("\n=== Strong Basic ex by type (dmg>=120, ne<=2) ===")
    by_type = collections.Counter(EnergyType(r[4].energyType).name for r in eff)
    for t, c in by_type.most_common():
        print(f"   {t:<10} {c}")

    # ---- Supporters (draw/search) ----
    print("\n=== Supporters (sample) ===")
    sup = [(cid, c) for cid, c in cards.items() if c.cardType == CardType.SUPPORTER.value]
    for cid, c in sorted(sup)[:40]:
        r = names.get(cid)
        eff_txt = (r["Effect Explanation"][:70] if r and r.get("Effect Explanation") else "")
        print(f"  id={cid:<5} {nm(cid)[:24]:<24} {eff_txt}")

    # ---- Items ----
    print("\n=== Items (sample) ===")
    it = [(cid, c) for cid, c in cards.items() if c.cardType == CardType.ITEM.value]
    for cid, c in sorted(it)[:40]:
        r = names.get(cid)
        eff_txt = (r["Effect Explanation"][:70] if r and r.get("Effect Explanation") else "")
        print(f"  id={cid:<5} {nm(cid)[:24]:<24} {eff_txt}")

    # ---- Basic energies ----
    print("\n=== Basic energies ===")
    for cid, c in sorted(cards.items()):
        if c.cardType == CardType.BASIC_ENERGY.value:
            print(f"  id={cid:<4} {nm(cid):<22} type={EnergyType(c.energyType).name}")

    print(f"\ntotals: {len(cards)} cards; "
          f"basic_ex={sum(1 for c in cards.values() if c.basic and c.ex and not c.megaEx)}; "
          f"supporters={len(sup)}; items={len(it)}")


if __name__ == "__main__":
    main()
