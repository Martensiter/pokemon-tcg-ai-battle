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


def _engine_kind(path: str) -> str:
    """Best-effort identify a shared library file. 'elf-x86_64' is what Kaggle needs."""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
    except OSError:
        return "missing"
    if head[:4] == b"\x7fELF":
        machine = head[18] | (head[19] << 8)        # e_machine (little-endian)
        return {0x3E: "elf-x86_64", 0xB7: "elf-arm64"}.get(machine, f"elf-machine-{machine:#06x}")
    if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
        return "mach-o-macos"
    return "unknown"


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


def build(deck_file: str, name: str, engine_so: str | None = None,
          extra_dirs: list[str] | None = None,
          allow_missing_engine: bool = False):
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

    # The agent loads cg/libcg.so at IMPORT time on Kaggle's grader (main.py ->
    # cg.sim -> ctypes LoadLibrary), so the submission MUST bundle the engine
    # binary or it crashes with "cannot open shared object file". cg/libcg.so is
    # gitignored (Pokemon-distributed) and on a Mac dev box may be the macOS
    # .dylib -- which is the WRONG arch for Kaggle (Linux x86_64). Resolve from
    # --engine-so if given (point it at sample_submission/cg/libcg.so), else the
    # repo's cg/libcg.so; then validate the arch.
    engine_dst = os.path.join(out, "cg", "libcg.so")
    if engine_so:
        if not os.path.exists(engine_so):
            raise SystemExit(f"--engine-so not found: {engine_so}")
        os.makedirs(os.path.dirname(engine_dst), exist_ok=True)
        shutil.copy2(engine_so, engine_dst)
    for xd in (extra_dirs or []):
        # vendor trees (e.g. orjson dual-ABI wheels) land at the bundle root so
        # `import orjson` resolves from main.py sys.path[0]
        for child in os.listdir(xd):
            s = os.path.join(xd, child)
            d = os.path.join(out, child)
            if os.path.isdir(s):
                _copytree(s, d)
            else:
                shutil.copy2(s, d)
    kind = _engine_kind(engine_dst)
    if kind == "missing":
        if not allow_missing_engine:
            raise SystemExit(
                "engine binary cg/libcg.so is MISSING from the bundle -- the agent would "
                "crash on Kaggle ('cannot open shared object file'). Put the Linux x86_64 "
                "engine in place first, e.g.:\n"
                "  python tools/make_submission.py --deck %s --name %s \\\n"
                "    --engine-so /path/to/sample_submission/cg/libcg.so\n"
                "(use --allow-missing-engine only to inspect bundle structure.)"
                % (deck_file, name))
        print("[WARN] no engine binary bundled (--allow-missing-engine): WILL crash on Kaggle.")
    elif kind != "elf-x86_64":
        print(f"[WARN] bundled cg/libcg.so looks like '{kind}', not Linux x86_64 ELF. "
              "Kaggle's grader is Linux x86_64 -- bundle sample_submission/cg/libcg.so "
              "(NOT the macOS .dylib or the arm64 .so). Override with --engine-so.")

    has_w = os.path.exists(os.path.join(out, "agent", "weights.npz"))
    has_engine = kind

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
    print(f"[{name}] deck={deck_file} weights={'yes' if has_w else 'NO'} "
          f"engine={has_engine} -> "
          f"submission_{name}.tar.gz ({size_tgz:.1f} MB)  +  submission_{name}.zip")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="deck.csv")
    ap.add_argument("--name", default="submission")
    ap.add_argument("--engine-so", default=None,
                    help="path to the LINUX x86_64 libcg.so to bundle as cg/libcg.so "
                         "(e.g. sample_submission/cg/libcg.so). Defaults to the repo's "
                         "cg/libcg.so; required if that is absent or the wrong arch.")
    ap.add_argument("--extra-dir", action="append", default=[],
                    help="copy directory contents into the bundle root (repeatable)")
    ap.add_argument("--allow-missing-engine", action="store_true",
                    help="build even without the engine binary (for inspection; "
                         "the resulting bundle WILL crash on Kaggle).")
    args = ap.parse_args()
    build(args.deck, args.name, engine_so=args.engine_so,
          allow_missing_engine=args.allow_missing_engine, extra_dirs=args.extra_dir)


if __name__ == "__main__":
    main()
