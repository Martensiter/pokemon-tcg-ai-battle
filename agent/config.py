"""Tunable parameters for the MCTS agent.

The per-move wall-clock budget is the key knob: MCTS is anytime, so this trades
strength for latency. Keep it conservative relative to whatever the Kaggle
evaluation enforces; it can be raised locally for stronger self-play.
"""
import os

# Per-decision search budget (seconds). Overridable via env for experiments.
# REVERTED from 1.2s -> 0.6s after ranked-data showed v2 (1.2s) underperformed
# v1 (0.6s) by 12-53 points across both deck variants on Kaggle. The value net
# is best-fit to ~0.6s search depth; deeper search explores branches it mis-
# evaluates, leading the agent into lines that roll-out well but actually fail.
MOVE_TIME_BUDGET = float(os.environ.get("PTCG_MOVE_BUDGET", "0.6"))

# Hard cap on simulations per move (safety even if the clock is generous).
MAX_SIMULATIONS = int(os.environ.get("PTCG_MAX_SIMS", "400"))

# Minimum simulations before we trust MCTS over the greedy fallback.
MIN_SIMULATIONS = 8

# Truncated-rollout depth (number of decisions simulated before leaf eval).
ROLLOUT_DEPTH = int(os.environ.get("PTCG_ROLLOUT_DEPTH", "24"))

# Exploration constant for UCB1 at the root.
UCB_C = 1.4

# Epsilon for the rollout policy (exploration noise inside playouts).
ROLLOUT_EPSILON = 0.15

# Number of distinct determinizations to cycle through per move. Each simulation
# samples hidden info; this caps how often we pay the (heavier) search_begin call
# by reusing a determinization for several rollouts.
DETERMINIZATIONS_PER_MOVE = int(os.environ.get("PTCG_DETS", "16"))

# Path to the learned value-net weights (Stage 5). If absent, use the heuristic.
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")

# Blend between heuristic and value net at leaves: 0 = pure heuristic, 1 = pure net.
VALUE_NET_WEIGHT = float(os.environ.get("PTCG_VNET_W", "0.7"))
