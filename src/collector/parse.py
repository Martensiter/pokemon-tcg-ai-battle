"""Defensive parsing of Kaggle episode / replay JSON.

The canonical reference for shapes is the existing repo code (``tools/import_
episodes.py``, ``tools/scan_episodes.py``) and the engine dataclasses in
``cg/api.py``. The competition explicitly warns that new enum members and object
attributes may be appended mid-competition, so every accessor here is best-effort
and must never raise on an unexpected/missing field -- unknown shapes yield empty
results, not exceptions.

Two wrapper formats are handled (as documented by the cabt viewer adapter):
  * Kaggle environment context: ``{"steps": [[obs0, obs1], ...], "rewards":[..],
    "info": {"Agents": [...]}}`` -- CABT ``visualize`` frames embedded at
    ``steps[0][0].observation.visualize``.
  * Lower-level local runner: top-level dict with a ``visualize`` array.

A "frame" is one board snapshot: ``{"current": State, "logs": [...],
"select": SelectData|None}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# LogType.RESULT == 23 (see cg/api.py). result: 0/1 winner index, 2 draw.
RESULT_LOG_TYPE = 23
# SelectContext.MAIN == 0 -- the strategic decisions the value net is trained on.
MAIN_CONTEXT = 0
DECK_SIZE = 60


@dataclass
class ParsedEpisode:
    """Normalised view of one episode, independent of wrapper format."""

    episode_id: str = ""
    agents: list[str] = field(default_factory=list)      # player index -> name
    rewards: list[float] = field(default_factory=list)   # player index -> reward
    winner: int = -1                                       # 0/1 winner, 2 draw, -1 unknown
    reason: Optional[int] = None
    decks: dict[int, list[int]] = field(default_factory=dict)  # player -> 60 ids
    frames: list[dict[str, Any]] = field(default_factory=list)
    n_frames: int = 0
    turns: int = 0
    ok: bool = False
    error: str = ""


def _as_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def unwrap(blob: Any) -> dict[str, Any]:
    """Peel the ``{"replay": ...}`` wrapper our client adds, if present."""
    if isinstance(blob, dict) and "replay" in blob and "steps" not in blob and "visualize" not in blob:
        inner = blob["replay"]
        if isinstance(inner, (dict, list)):
            return {"_payload": inner, "_meta": {k: v for k, v in blob.items() if k != "replay"}}
    return {"_payload": blob, "_meta": {}}


def _frames_from_steps(steps: list) -> list[dict[str, Any]]:
    """Build frames from the Kaggle env timeline.

    The real cabt replay is the Kaggle environment format: ``steps`` is a list of
    timesteps, each a list of per-agent entries ``{action, observation, status,
    ...}``. The agent that is to move carries the live ``observation`` with a
    ``current`` (State) and ``select`` (SelectData) -- exactly the obs dict the
    agent sees live. We turn each such observation into a frame
    ``{current, select, logs}``; the inactive seat (``current``/``select`` None)
    is naturally skipped downstream by :func:`iter_main_decisions`.
    """
    frames: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, list):
            continue
        for entry in step:
            e = _as_dict(entry)
            # Skip the non-acting seat (it may carry a stale ``current``); the
            # acting seat is the one we want, identified by status / live select.
            if str(e.get("status", "")).upper() == "INACTIVE":
                continue
            obs = e.get("observation")
            if not isinstance(obs, dict):
                continue
            cur = obs.get("current")
            if isinstance(cur, dict):
                frames.append({"current": cur, "select": obs.get("select"),
                               "logs": obs.get("logs")})
    return frames


def extract_frames(payload: Any) -> list[dict[str, Any]]:
    """Yield board-state frames regardless of wrapper format.

    Handles, in order: a top-level ``visualize`` array; a ``visualize`` array
    embedded in the first agent observation; and the Kaggle env timeline where
    each ``steps[i][seat].observation`` carries ``current``/``select`` (the real
    cabt replay shape).
    """
    if isinstance(payload, dict):
        vis = payload.get("visualize")
        if isinstance(vis, list):
            return [f for f in vis if isinstance(f, dict)]
        steps = payload.get("steps")
        if isinstance(steps, list) and steps:
            first = None
            try:
                first = steps[0][0]
            except (IndexError, TypeError, KeyError):
                first = None
            obs = _as_dict(first).get("observation")
            if isinstance(obs, dict) and isinstance(obs.get("visualize"), list):
                return [f for f in obs["visualize"] if isinstance(f, dict)]
            if isinstance(obs, list):
                return [f for f in obs if isinstance(f, dict)]
            # Kaggle env timeline: frames live in each agent's observation.
            frames = _frames_from_steps(steps)
            if frames:
                return frames
    if isinstance(payload, list):
        return [f for f in payload if isinstance(f, dict)]
    return []


def _deck_from_action(action: Any) -> Optional[list[int]]:
    """A CABT first action for a player is the 60 card-id list (their deck)."""
    if isinstance(action, list) and len(action) == DECK_SIZE and all(isinstance(x, int) for x in action):
        return list(action)
    return None


def extract_decks(payload: Any) -> dict[int, list[int]]:
    """Best-effort: find each player's submitted 60-card deck from early actions."""
    decks: dict[int, list[int]] = {}
    steps = _as_dict(payload).get("steps")
    if not isinstance(steps, list):
        return decks
    for step_i in range(min(3, len(steps))):
        for pi, agent_state in enumerate(_as_list(steps[step_i])):
            if pi in decks:
                continue
            d = _deck_from_action(_as_dict(agent_state).get("action"))
            if d is not None:
                decks[pi] = d
    return decks


def extract_agents(payload: Any) -> list[str]:
    """Agent display names by player index (from ``info.Agents``)."""
    info = _as_dict(_as_dict(payload).get("info"))
    agents = _as_list(info.get("Agents"))
    out: list[str] = []
    for a in agents:
        out.append(str(_as_dict(a).get("Name", "?")))
    return out


def extract_rewards(payload: Any) -> list[float]:
    """Per-player terminal rewards (``rewards`` array). ``[]`` if absent."""
    rewards = _as_dict(payload).get("rewards")
    if not isinstance(rewards, list):
        return []
    out: list[float] = []
    for r in rewards:
        try:
            out.append(float(r) if r is not None else 0.0)
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def winner_from_rewards(rewards: list[float]) -> int:
    """Winner index from rewards (reward == 1 -> win, per scan_episodes)."""
    if len(rewards) >= 2:
        if rewards[0] == rewards[1]:
            return 2
        return 0 if rewards[0] > rewards[1] else 1
    return -1


def winner_from_frames(frames: list[dict[str, Any]]) -> tuple[int, Optional[int]]:
    """Walk frames for the RESULT log; fall back to the final state's result."""
    for f in frames:
        for lg in _as_list(f.get("logs")):
            lg = _as_dict(lg)
            if lg.get("type") in (RESULT_LOG_TYPE, "Result"):
                return _as_int(lg.get("result"), -1), _as_int(lg.get("reason"), None)
    if frames:
        cur = _as_dict(frames[-1].get("current"))
        return _as_int(cur.get("result"), -1), None
    return -1, None


def final_turn(frames: list[dict[str, Any]]) -> int:
    for f in reversed(frames):
        t = _as_dict(f.get("current")).get("turn")
        if t:
            return _as_int(t, 0)
    return 0


def _as_int(x: Any, default: Optional[int]) -> Any:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def parse_episode(blob: Any, episode_id: str = "") -> ParsedEpisode:
    """Top-level entry: normalise a raw replay blob into :class:`ParsedEpisode`.

    Never raises on malformed input -- failures surface via ``ok=False`` and
    ``error``.
    """
    try:
        wrapped = unwrap(blob)
        payload = wrapped["_payload"]
        meta = wrapped["_meta"]
        ep_id = str(episode_id or meta.get("episode_id") or "")

        frames = extract_frames(payload)
        agents = extract_agents(payload)
        rewards = extract_rewards(payload)
        decks = extract_decks(payload)

        winner = winner_from_rewards(rewards)
        reason: Optional[int] = None
        if winner == -1:
            winner, reason = winner_from_frames(frames)

        ep = ParsedEpisode(
            episode_id=ep_id,
            agents=agents,
            rewards=rewards,
            winner=winner,
            reason=reason,
            decks=decks,
            frames=frames,
            n_frames=len(frames),
            turns=final_turn(frames),
            ok=bool(frames) or winner != -1,
        )
        if not ep.ok:
            ep.error = "no visualize frames and no result"
        return ep
    except Exception as e:  # noqa: BLE001  (parser must never raise)
        return ParsedEpisode(episode_id=str(episode_id), ok=False,
                             error=f"{type(e).__name__}: {e}")


def iter_main_decisions(ep: ParsedEpisode):
    """Yield ``(state_dict, to_move_index)`` for each MAIN decision frame.

    A frame qualifies when it carries a ``select`` with context MAIN, a live
    ``current`` state (game not finished), and a valid ``yourIndex``. This mirrors
    the sampling in ``selfplay/gen_data.py`` so the resulting features match what
    the value net already trains on.
    """
    for f in ep.frames:
        sel = f.get("select")
        if not isinstance(sel, dict):
            continue
        if sel.get("context") != MAIN_CONTEXT:
            continue
        state = f.get("current")
        if not isinstance(state, dict):
            continue
        if state.get("result", -1) not in (-1, None):
            continue
        me = state.get("yourIndex")
        if not isinstance(me, int) or me not in (0, 1):
            continue
        players = state.get("players")
        if not isinstance(players, list) or len(players) < 2:
            continue
        yield state, me
