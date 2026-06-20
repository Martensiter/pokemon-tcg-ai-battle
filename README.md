# Pokemon TCG AI Battle Challenge — MCTS Agent + Episode Mining

> **Strategy Category writeup:** see [WRITEUP.md](WRITEUP.md) — full submission text for the Strategy Category competition.
>
> **Simulation Category submission:** `submission_hops_hybrid_v2.tar.gz` (built from this repo via `tools/make_submission.py`).

A determinized Information-Set MCTS agent with a self-trained numpy value network, driven by data mined from 6,533 real Kaggle episode JSONs. Four documented deck pivots took the agent from 600 (initial TrueSkill) to **948 in one hour** on the actual ladder.

## Headline results

| Iteration | Real-ladder score | Notes |
|---|---|---|
| Crustle wall (Pivot 1-2) | 745 | Sim-tested; counters known too widely |
| Lucario+Riolu (Pivot 3) | 604 | Right deck, wrong agent for its tempo |
| hops_hybrid (Pivot 4 baseline) | 948 (1h) | Identified via episode mining of top archetypes |
| **hops_hybrid_v2** (Pivot 4 tuned) | (climbing) | +Cramorant +Hilda +Boss's; 79% h2h vs baseline |

## Architecture

- `agent/mcts.py` — determinized flat-UCB MCTS over the engine's `search_begin`/`search_step` API
- `agent/determinize.py` — samples hidden info (opp deck/hand) consistent with what's visible
- `agent/policy.py` — heuristic action policy for rollouts and multi-select fallback
- `agent/value_net.py` — numpy MLP (32→64→64→1) for leaf evaluation; trained with torch, exported to `weights.npz`
- `agent/agent.py` — orchestrates: returns deck on first call, MCTS otherwise

Inference is **numpy-only** — no torch, no pandas, no network. Verified by [tests/submission_test.py](tests/submission_test.py).

## Repository structure

```
agent/                MCTS agent + value net (ships in submission)
cg/                   Engine SDK + binaries (Pokemon-distributed — NOT IN REPO; see Setup)
selfplay/             Self-play harness, data generation, training, tournaments
tools/                Deck builders, episode scanners, packaging, replay viewer
tests/                Smoke tests, exec-style submission verification
WRITEUP.md            Strategy Category writeup (1813 words)
weights.npz           Trained value-net weights (numpy)
deck_*.csv            60-card decklists by archetype
```

## Setup (cannot run without these)

This repository requires **Pokemon-distributed materials** that are NOT shipped
here due to competition redistribution rules:

1. **Engine SDK + binaries.** Download the `sample_submission/cg/` folder from
   the [competition page](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data).
   Copy `cg.dll` (Windows) and `libcg.so` (Linux) into this repo's `cg/` directory.
   (The `cg/*.py` files in this repo are reproduced from the SDK header for IDE help.)

2. **Card data CSV.** Download `EN_Card_Data.csv` from the competition page and
   place it in the repo root. The agent does not require it at runtime; it is
   used by deck-building and analysis tools only.

3. **Python.** Tested on Python 3.12. Runtime needs only `numpy`. Training and
   tools need `torch` + standard library:
   ```bash
   pip install numpy torch
   ```

## Reproduction

```bash
# 1. Sanity — engine + battle loop + search API
python tests/smoke_test.py
python tests/search_test.py

# 2. Build decks (each writes deck_*.csv)
python tools/build_hops_hybrid_v2.py
python tools/build_anti_meta.py

# 3. Round-robin tournament across multiple decks
python selfplay/round_robin.py --agent mcts -n 10

# 4. Train value net (requires data.npz from selfplay/gen_data.py)
python selfplay/gen_data.py -n 6000 --decks pool --agent greedy
python selfplay/train_value.py --epochs 50

# 5. Package + verify a Kaggle submission
python tools/make_submission.py --deck deck_cand_hops_hybrid_v2.csv --name hops_hybrid_v2
python tests/submission_test.py submission_hops_hybrid_v2

# 6. Mine episode replays (requires daily episode dataset downloaded to ./episodes/)
python tools/scan_episodes.py --top-agents "hiroingk,Kadoraba"
```

## Key files for the Strategy writeup

| File | Purpose |
|---|---|
| `WRITEUP.md` | Full Strategy Category submission text (1813 words) |
| `agent/mcts.py` | Determinized MCTS — methodology core |
| `agent/value_net.py` | Numpy value net inference |
| `selfplay/train_value.py` | Value-net training pipeline |
| `tools/scan_episodes.py` | Episode-mining tool that produced the tier list |
| `selfplay/round_robin.py` | Deck-tournament harness for pivot decisions |
| `deck_cand_hops_hybrid_v2.csv` | Final primary deck — Dudunsparce + Hop's hybrid, tuned |

## License

[MIT](LICENSE) — applies to all code, decks, weights, and documentation in this
repository. **Pokemon-distributed materials** (engine binary, card data, episode
replays) are NOT covered by this license and are not redistributed here.
