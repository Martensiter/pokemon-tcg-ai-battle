"""Verify the assembled submission/ is self-contained.

Blocks torch and pandas at import time, then loads `main` from the submission
folder only and drives a full game with `main.agent` controlling both seats.
A clean finish proves the submission needs nothing beyond cg/ + agent/ + numpy.

Run after tools/make_submission.py:  python tests/submission_test.py
"""
import os
import sys
import importlib.abc

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# target bundle: tests/submission_test.py [submission_dir_name]
_target = sys.argv[1] if len(sys.argv) > 1 else "submission"
SUB = _target if os.path.isabs(_target) else os.path.join(PROJECT, _target)


class _Block(importlib.abc.MetaPathFinder):
    BLOCKED = {"torch", "pandas"}

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"'{name}' must NOT be needed at inference")
        return None


def main():
    assert os.path.isdir(SUB), "run tools/make_submission.py first"
    # Resolve imports ONLY against the submission folder.
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != PROJECT]
    sys.path.insert(0, SUB)
    sys.meta_path.insert(0, _Block())

    os.environ.setdefault("PTCG_MOVE_BUDGET", "0.05")

    # ---- Kaggle exec()-style smoke ----
    # Kaggle's harness loads main.py via exec(code_object, env), so __file__ is
    # NOT defined in the executing globals. Catch any path/setup code that
    # implicitly relies on __file__ BEFORE we ship — production-bug repro.
    main_path = os.path.join(SUB, "main.py")
    with open(main_path) as f:
        code = compile(f.read(), "main.py", "exec")
    kaggle_globals = {}  # intentionally bare — no __file__, no __name__
    exec(code, kaggle_globals)
    assert callable(kaggle_globals.get("agent")), "exec() did not expose agent()"

    import main  # noqa
    from cg.game import battle_start, battle_select, battle_finish

    deck = None
    # main exposes agent; deck loaded via agent on select=None
    obs, start = battle_start(
        __import__("agent.base", fromlist=["read_deck"]).read_deck(),
        __import__("agent.base", fromlist=["read_deck"]).read_deck(),
    )
    assert obs is not None, "battle failed to start"

    steps = 0
    winner = 2
    while True:
        st = obs.get("current")
        if st and st.get("result", -1) != -1:
            winner = st["result"]
            break
        sel = obs.get("select")
        if sel is None:
            break
        choice = main.agent(obs)
        obs = battle_select(choice)
        steps += 1
        if steps > 30000:
            raise RuntimeError("did not terminate")
    battle_finish()

    assert "torch" not in sys.modules, "torch was imported!"
    assert "pandas" not in sys.modules, "pandas was imported!"
    print(f"submission self-contained OK: full game finished, winner={winner}, steps={steps}")
    print("numpy-only inference confirmed (torch/pandas blocked).")


if __name__ == "__main__":
    main()
