"""hops_hybrid_v3 tech variants — testing specific anti-meta cards.

v3a (surge_bargain):
  + 2 Lt. Surge's Bargain  (exploit naive Yes/No selection — Yes=free prize, No=draw 4)
  - 1 Colress's Tenacity
  - 1 Hop's Bag

v3b (disruption):
  + 2 Xerosic's Machinations  (cap opponent hand at 3 — kills Alakazam Powerful Hand)
  - 1 Colress's Tenacity
  - 1 Hop's Bag

v3c (combo):
  + 1 Lt. Surge's Bargain
  + 1 Xerosic's Machinations
  + 2 Crushing Hammer        (coin-flip energy denial)
  - 1 Colress's Tenacity
  - 1 Hop's Bag
  - 1 Postwick
  - 1 Pokegear 3.0
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck

# Base v2 (hops_hybrid_v2.py)
BASE = [
    (65, 4), (66, 3), (878, 4), (879, 2), (311, 2), (304, 2),
    (1086, 3), (1152, 4), (1122, 3), (1115, 3), (1171, 4),
    (1227, 4), (1182, 3), (1225, 2), (1194, 2), (1097, 3),
    (1255, 3),
    (19, 4), (11, 4), (12, 1),
]

VARIANTS = {
    "v3a_surge": {
        "+": [(1226, 2)],  # Lt. Surge's Bargain
        "-": [(1194, 1), (1115, 1)],  # Colress, Hop's Bag
    },
    "v3b_disrupt": {
        "+": [(1197, 2)],  # Xerosic's Machinations
        "-": [(1194, 1), (1115, 1)],
    },
    "v3c_combo": {
        "+": [(1226, 1), (1197, 1), (1120, 2)],  # Surge + Xerosic + 2 Crushing Hammer
        "-": [(1194, 1), (1115, 1), (1255, 1), (1122, 1)],
    },
}


def build(variant_name):
    counts = {cid: n for cid, n in BASE}
    spec = VARIANTS[variant_name]
    for cid, n in spec["-"]:
        counts[cid] = counts.get(cid, 0) - n
        if counts[cid] <= 0:
            del counts[cid]
    for cid, n in spec["+"]:
        counts[cid] = counts.get(cid, 0) + n
    ids = []
    for cid, n in counts.items():
        ids.extend([cid] * n)
    return ids


def main():
    db = get_db()
    for vname in VARIANTS:
        ids = build(vname)
        ok, probs = validate_deck(ids)
        ne = sum(1 for c in ids if db.is_energy(c))
        print(f"hops_hybrid_{vname}: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
        if ok:
            out = os.path.join(ROOT, f"deck_cand_hops_hybrid_{vname}.csv")
            with open(out, "w", newline="") as f:
                f.write("\n".join(str(c) for c in ids) + "\n")


if __name__ == "__main__":
    main()
