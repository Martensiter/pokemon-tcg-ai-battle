"""Inspect candidate attackers' abilities/attacks and the support pool by keyword."""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from cg.api import all_card_data, all_attack, CardType, EnergyType  # noqa: E402

cards = {c.cardId: c for c in all_card_data()}
attacks = {a.attackId: a for a in all_attack()}
names = {}
with open(os.path.join(ROOT, "EN_Card_Data.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            names[int(row["Card ID"])] = row
        except (ValueError, KeyError):
            pass

def show_card(cid):
    c = cards.get(cid)
    if not c:
        print(f"  #{cid} not found"); return
    flags = "".join(k for k, v in [("Basic", c.basic), ("ex", c.ex), ("megaEx", c.megaEx)] if v)
    print(f"id={cid} {c.name}  HP{c.hp} {EnergyType(c.energyType).name} [{flags}] retreat={c.retreatCost}")
    for s in c.skills:
        print(f"    [Ability/Skill] {s.name}: {s.text[:120]}")
    for aid in c.attacks:
        a = attacks.get(aid)
        if a:
            cost = "".join(EnergyType(e).name[0] for e in a.energies) or "-"
            print(f"    [Attack] {a.name} cost={cost} dmg={a.damage}: {a.text[:90]}")

print("=== Candidate Lightning attackers ===")
for cid in [313, 210, 328, 37, 806, 1062, 547]:
    show_card(cid)

def search_support(keywords, ctypes, limit=40):
    kws = [k.lower() for k in keywords]
    out = []
    for cid, c in cards.items():
        if c.cardType not in ctypes:
            continue
        r = names.get(cid)
        txt = (r["Effect Explanation"] if r and r.get("Effect Explanation") else "") + " " + (c.name or "")
        low = txt.lower()
        if any(k in low for k in kws):
            out.append((cid, c.name or (r["Card Name"] if r else str(cid)), txt[:80]))
    out.sort()
    return out[:limit]

print("\n=== Draw / search supporters ===")
for cid, nm, txt in search_support(
        ["draw", "search your deck", "shuffle your hand"], {CardType.SUPPORTER.value}, 30):
    print(f"  id={cid:<5} {nm[:24]:<24} {txt}")

print("\n=== Gust / switch supporters ===")
for cid, nm, txt in search_support(
        ["switch in 1 of your opponent", "benched", "switch your active"], {CardType.SUPPORTER.value}, 15):
    print(f"  id={cid:<5} {nm[:24]:<24} {txt}")

print("\n=== Ball / search items ===")
for cid, nm, txt in search_support(
        ["search your deck for", "basic pok", "put it into your hand"], {CardType.ITEM.value}, 25):
    print(f"  id={cid:<5} {nm[:24]:<24} {txt}")

print("\n=== Energy acceleration (items/tools/supporters) ===")
for cid, nm, txt in search_support(
        ["attach", "energy from your", "energy card"],
        {CardType.ITEM.value, CardType.TOOL.value, CardType.SUPPORTER.value}, 30):
    print(f"  id={cid:<5} {nm[:24]:<24} {txt}")

print("\n=== Basic Lightning energy id ===")
for cid, c in sorted(cards.items()):
    if c.cardType == CardType.BASIC_ENERGY.value and c.energyType == EnergyType.LIGHTNING.value:
        print(f"  id={cid} {c.name}")
