"""Light accessors over the raw observation dict.

We operate on the raw dict (not the dataclass) in the hot path for speed, but
these helpers keep call sites readable. The dict layout mirrors the dataclasses
in cg/api.py (State / PlayerState / Pokemon / SelectData / Option).
"""
from __future__ import annotations


def me_index(state: dict) -> int:
    return state["yourIndex"]


def my_state(state: dict) -> dict:
    return state["players"][state["yourIndex"]]


def opp_state(state: dict) -> dict:
    return state["players"][1 - state["yourIndex"]]


def active_of(player: dict) -> dict | None:
    a = player.get("active") or []
    return a[0] if a and a[0] is not None else None


def in_play(player: dict) -> list[dict]:
    """Active + bench Pokemon (skipping face-down / empty active)."""
    out = []
    a = player.get("active") or []
    if a and a[0] is not None:
        out.append(a[0])
    out.extend(player.get("bench") or [])
    return out


def prize_remaining(player: dict) -> int:
    return len(player.get("prize") or [])


def total_energy(pokemon: dict) -> int:
    return len(pokemon.get("energies") or [])


def has_any_condition(player: dict) -> bool:
    return any(player.get(k) for k in ("poisoned", "burned", "asleep", "paralyzed", "confused"))


def n_conditions(player: dict) -> int:
    return sum(1 for k in ("poisoned", "burned", "asleep", "paralyzed", "confused") if player.get(k))
