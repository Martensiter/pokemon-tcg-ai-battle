"""Build the two top-meta Kaggle decks reverse-engineered from real episodes.

Kadoraba's Alakazam + Dudunsparce psychic combo deck (67% winrate, 64 games)
   * Abra/Kadabra/Alakazam Stage-2 line, Powerful Hand scales with hand size
   * Dunsparce/Dudunsparce draw engine (3 cards per turn via ability)
   * Special energy package: Enriching + Telepath Psychic
   * Tech: Lucky Helmet (counter), Air Balloon (free retreat), Enhanced Hammer

Blake Stagner's Mega Lucario ex + Riolu + Solrock deck (60% winrate, 48 games)
   * Riolu pre-evolve attacker + Mega Lucario ex finisher (270 dmg)
   * Hariyama Fighting partner (Wild Press 210)
   * Lunatone/Solrock duo (engine, Lunatone discards F energy for Solrock setup)
   * Stadium: Gravity Mountain (-30 HP Stage 2 — counters Alakazam!)
   * Tech: Premium Power Pro (+30 dmg), Fighting Gong, Hero's Cape
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck, ENERGY_ID_BY_TYPE
from cg.api import EnergyType

DECKS = {
    "alakazam_pro": dict(
        # Stage 2 Alakazam line + Dudunsparce non-ex partner (mirror of hiroingk/Kadoraba)
        # 741=Abra (Boosted Evolution), 742=Kadabra (Psychic Draw), 743=Alakazam (Psychic Draw + Powerful Hand)
        # 65=Dunsparce, 66=Dudunsparce, 140=Fezandipiti ex
        pokemon=[(741, 4), (742, 4), (743, 3), (65, 4), (66, 3), (140, 1)],
        # Poké Pad, Buddy-Buddy Poffin, Rare Candy, Night Stretcher, Enhanced Hammer, Lucky Helmet, Air Balloon, Ultra Ball
        items=[(1152, 4), (1086, 4), (1079, 4), (1097, 3), (1081, 2), (1156, 2), (1174, 2), (1121, 2)],
        # Boss's Orders, Lana's Aid, Hilda, Lillie's Determination, Carmine
        supporters=[(1182, 4), (1184, 2), (1225, 1), (1227, 4), (1192, 2)],
        # 1 Enriching Energy (ACE SPEC), 4 Telepath Psychic
        energy_special=[(13, 1), (19, 4)],
        energy=EnergyType.PSYCHIC,         # basic Psychic fills rest
    ),
    "lucario_riolu": dict(
        # Blake Stagner's Mega Lucario shell
        # 677=Riolu, 678=Mega Lucario ex, 673=Makuhita, 674=Hariyama, 675=Lunatone, 676=Solrock
        pokemon=[(677, 4), (678, 3), (673, 2), (674, 2), (675, 2), (676, 1)],
        # Poké Pad, Fighting Gong, Premium Power Pro, Buddy-Buddy Poffin, Night Stretcher, Dusk Ball, Ultra Ball, Switch
        items=[(1152, 4), (1142, 3), (1141, 3), (1086, 3), (1097, 2), (1102, 3), (1121, 2), (1123, 2)],
        # Boss's Orders, Lillie's Determination, Carmine
        supporters=[(1182, 4), (1227, 4), (1192, 2)],
        tools=[(1159, 1)],   # Hero's Cape ACE
        stadium=[(1252, 1)], # Gravity Mountain (-30 HP Stage 2 — counters Alakazam)
        energy=EnergyType.FIGHTING,
    ),
}


def build(spec):
    ids = []
    for cid, n in spec.get("pokemon", []):
        ids += [cid]*n
    for cid, n in spec.get("items", []):
        ids += [cid]*n
    for cid, n in spec.get("supporters", []):
        ids += [cid]*n
    for cid, n in spec.get("tools", []):
        ids += [cid]*n
    for cid, n in spec.get("stadium", []):
        ids += [cid]*n
    for cid, n in spec.get("energy_special", []):
        ids += [cid]*n
    ids += [ENERGY_ID_BY_TYPE[spec["energy"]]] * (60 - len(ids))
    return ids


def main():
    db = get_db()
    for name, spec in DECKS.items():
        ids = build(spec)
        ok, probs = validate_deck(ids)
        ne = sum(1 for c in ids if db.is_energy(c))
        print(f"[{name:<14}] cards={len(ids)} energy={ne} {spec['energy'].name}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
        if ok:
            out = os.path.join(ROOT, f"deck_cand_{name}.csv")
            with open(out, "w", newline="") as f:
                f.write("\n".join(str(c) for c in ids) + "\n")
            print(f"    wrote {out}")


if __name__ == "__main__":
    main()
