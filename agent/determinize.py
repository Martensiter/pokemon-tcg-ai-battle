"""Sample a determinization of the hidden information for search_begin.

Imperfect information lives in: our own deck order + face-down prizes, and the
opponent's deck / prizes / hand / (face-down) active. We compute our hidden
multiset exactly (full deck minus everything visible) and sample the opponent's
hidden cards from a prior. The prior assumes the opponent runs a deck like ours
(guarantees legality + at least one Basic Pokemon, which the engine requires at
setup); it can be refined later from observed opponent cards.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .cards import get_db
from .observation import in_play

# A reliable Basic Pokemon id from our pool, used to satisfy the engine's
# "opponent deck needs a Basic" / face-down active requirements.
_FALLBACK_BASIC = 806   # Rotom ex
_FALLBACK_ENERGY = 4    # Basic {L} Energy


@dataclass
class Determinization:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


def _visible_mine(player: dict) -> Counter:
    vis: Counter = Counter()
    for c in (player.get("hand") or []):
        if c:
            vis[c["id"]] += 1
    for c in (player.get("discard") or []):
        if c:
            vis[c["id"]] += 1
    for p in in_play(player):
        vis[p["id"]] += 1
        for grp in ("energyCards", "tools", "preEvolution"):
            for c in (p.get(grp) or []):
                if c:
                    vis[c["id"]] += 1
    for c in (player.get("prize") or []):
        if c is not None:           # face-up prize (rare)
            vis[c["id"]] += 1
    return vis


def determinize(obs: dict, my_deck_ids: list[int], rng: random.Random,
                arch_prior: float = 0.0) -> Determinization:
    db = get_db()
    st = obs["current"]
    me = st["yourIndex"]
    opp = 1 - me
    mp = st["players"][me]
    op = st["players"][opp]

    # ---- our hidden cards (deck + face-down prizes) ----
    unknown = Counter(my_deck_ids) - _visible_mine(mp)
    pool = list(unknown.elements())
    rng.shuffle(pool)

    prize_list = mp.get("prize") or []
    need_prize_hidden = sum(1 for c in prize_list if c is None)
    need_deck = mp.get("deckCount", 0)
    while len(pool) < need_prize_hidden + need_deck:
        pool.append(_FALLBACK_ENERGY)

    your_prize = []
    for c in prize_list:
        your_prize.append(c["id"] if c is not None else pool.pop())
    your_deck = pool[:need_deck]

    # ---- opponent hidden cards ----
    facedown_active = bool(op.get("active") and op["active"][0] is None)
    need_opp = op.get("deckCount", 0) + len(op.get("prize") or []) + op.get("handCount", 0) \
        + (1 if facedown_active else 0)

    # Archetype prior (OPT-IN, fork roadmap (c)): match the opponent's visible
    # cards against mined meta archetypes and sample their hidden cards from
    # the matched list MINUS what we have already seen. No confident match
    # (early game, off-meta deck) -> mirror prior below, byte-identical.
    opp_pool = None
    arch_basic = 0
    if arch_prior > 0:
        from . import arch_prior as _ap
        vis_op = _visible_mine(op)
        m = _ap.match(vis_op)
        if m is not None:
            _, arch_deck, arch_basic = m
            remain = list((Counter(arch_deck) - vis_op).elements())
            rng.shuffle(remain)
            while len(remain) < need_opp:      # observed cards outside the list etc.
                pad = list(arch_deck)
                rng.shuffle(pad)
                remain.extend(pad)
            opp_pool = remain[:need_opp]

    if opp_pool is None:                       # mirror prior (stock behavior)
        template = list(my_deck_ids)
        reps = need_opp // len(template) + 1 if template else 1
        opp_pool = (template * reps)[:need_opp]
        rng.shuffle(opp_pool)

    d = op.get("deckCount", 0)
    opponent_deck = opp_pool[:d]
    if opponent_deck and not any(db.is_basic_pokemon(c) for c in opponent_deck):
        opponent_deck[0] = _FALLBACK_BASIC
    rest = opp_pool[d:]
    pc = len(op.get("prize") or [])
    opponent_prize = rest[:pc]
    rest = rest[pc:]
    hc = op.get("handCount", 0)
    opponent_hand = rest[:hc]
    opponent_active = [arch_basic or _FALLBACK_BASIC] if facedown_active else []

    return Determinization(
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=opponent_active,
    )
