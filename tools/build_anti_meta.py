"""Anti-meta deck targeting hiroingk's Alakazam and Blake's Mega Lucario.

Core thesis:
  * Iron Boulder (Psychic) Adjusted Horn deals 170, doubled by Mega Lucario ex's
    Psychic weakness -> exact OHKO of a 340 HP fully-evolved Mega Lucario.
  * Judge (shuffle hand, draw 4) hard-caps Alakazam's Powerful Hand at ~80 dmg.
  * Gravity Mountain (-30 HP to Stage 2) drops Alakazam from 140 -> 110 HP,
    well within Iron Boulder's reach without needing weakness.
  * Mesprit (160 dmg, 2 energy, single prize) is cheap chip / Boss's Orders gust
    target for sniping Riolu before it evolves into Mega Lucario.
  * Latias ex backs up the prize race when we need ex tempo; Skyliner ability
    gives every Basic free retreat (huge mobility tech).

Strategy: race Alakazam's setup, OHKO Mega Lucario the turn it evolves.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck, ENERGY_ID_BY_TYPE
from cg.api import EnergyType


def build():
    # PIVOT: Iron Boulder + Mesprit have conditional attacks that often do nothing.
    # Latias ex is unconditional 200 dmg PPC; vs Mega Lucario (Psychic weak) =
    # 400 = OHKO. Skyliner ability gives all Basic Pokemon free retreat.
    POK = [(184, 4), (971, 3)]
    # 184=Latias ex (primary, unconditional 200 dmg PPC),
    # 971=Iron Boulder (backup — fires when hand sizes match by chance)
    ITEMS = [(1086, 4), (1121, 4), (1152, 4), (1097, 3), (1102, 3), (1144, 2)]
    # Buddy-Buddy Poffin, Ultra Ball, Poké Pad x4 (top-meta std), Night Stretcher,
    # Dusk Ball, Strange Timepiece x2 (anti-Alakazam — devolves their Stage 2)
    SUP = [(1182, 4), (1213, 4), (1227, 4), (1192, 2), (1224, 3)]
    # Boss's Orders, Judge (Alakazam counter), Lillie's, Carmine, Cheren
    TOOL = [(1159, 1)]   # Hero's Cape ACE SPEC
    STADIUM = [(1252, 2)]  # Gravity Mountain (anti-Stage 2)
    SE = [(19, 4)]       # Telepath Psychic Energy
    ENERGY = EnergyType.PSYCHIC

    ids = []
    for cid, n in POK + ITEMS + SUP + TOOL + STADIUM + SE:
        ids.extend([cid] * n)
    ids.extend([ENERGY_ID_BY_TYPE[ENERGY]] * (60 - len(ids)))
    return ids


def main():
    db = get_db()
    ids = build()
    ok, probs = validate_deck(ids)
    ne = sum(1 for c in ids if db.is_energy(c))
    print(f"anti_meta_psychic: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
    if ok:
        out = os.path.join(ROOT, "deck_cand_anti_meta.csv")
        with open(out, "w", newline="") as f:
            f.write("\n".join(str(c) for c in ids) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
