"""Assemble a self-contained, named submission bundle (and zip).

Copies only inference-time files: main.py, the chosen deck as deck.csv, cg/ (engine
SDK), and agent/ (incl. the active weights.npz). Excludes training code, the card
CSV, datasets, and weight backups. Use --deck to pick which deck ships and --name
to label the bundle, so the two hedge submissions can be built from one agent.

  python tools/make_submission.py --deck deck.csv          --name lightning
  python tools/make_submission.py --deck deck_crustle_v2.csv --name crustle
"""
import os
import sys
import shutil
import tarfile
import zipfile
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INCLUDE_DIRS = ["cg", "agent"]
PRUNE_NAMES = {"__pycache__"}
# never ship: pyc, weight backups, any stray data
PRUNE_SUFFIXES = (".pyc",)
PRUNE_PREFIXES = ("weights_v",)   # weights_v1_lightning.npz, weights_v2_diverse.npz, ...


def _copytree(src, dst):
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in PRUNE_NAMES]
        rel = os.path.relpath(root, src)
        target = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target, exist_ok=True)
        for fn in files:
            if fn.endswith(PRUNE_SUFFIXES) or fn.startswith(PRUNE_PREFIXES):
                continue
            shutil.copy2(os.path.join(root, fn), os.path.join(target, fn))


def build(deck_file: str, name: str):
    out = os.path.join(ROOT, f"submission_{name}")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    shutil.copy2(os.path.join(ROOT, "main.py"), os.path.join(out, "main.py"))
    src_deck = os.path.join(ROOT, deck_file)
    if not os.path.exists(src_deck):
        raise SystemExit(f"deck not found: {src_deck}")
    shutil.copy2(src_deck, os.path.join(out, "deck.csv"))   # always lands as deck.csv
    for d in INCLUDE_DIRS:
        _copytree(os.path.join(ROOT, d), os.path.join(out, d))

    has_w = os.path.exists(os.path.join(out, "agent", "weights.npz"))

    # Kaggle Simulation competitions expect a single tar.gz with the agent
    # files at the archive root (no extra wrapper directory). We also keep
    # writing a .zip for compatibility with anything that wants it.
    tgz_path = os.path.join(ROOT, f"submission_{name}.tar.gz")
    with tarfile.open(tgz_path, "w:gz") as tf:
        for root, _d, files in os.walk(out):
            for fn in files:
                full = os.path.join(root, fn)
                tf.add(full, arcname=os.path.relpath(full, out))
    zip_path = os.path.join(ROOT, f"submission_{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _d, files in os.walk(out):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, out))
    size_tgz = os.path.getsize(tgz_path) / 1e6
    print(f"[{name}] deck={deck_file} weights={'yes' if has_w else 'NO'} -> "
          f"submission_{name}.tar.gz ({size_tgz:.1f} MB)  +  submission_{name}.zip")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="deck.csv")
    ap.add_argument("--name", default="submission")
    args = ap.parse_args()
    build(args.deck, args.name)


if __name__ == "__main__":
    main()
