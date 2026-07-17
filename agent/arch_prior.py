"""Archetype-aware opponent prior (fork roadmap item (c), WRITEUP.md).

The stock determinizer samples the opponent's hidden cards from a MIRROR prior
("assume they run our deck"). This module matches the opponent's VISIBLE cards
(board, discard, face-up prizes) against a library of meta archetypes mined
from the official daily episode datasets (agent/archetypes.json) and, on a
confident match, lets the determinizer sample their hidden deck/hand/prizes
from that archetype's remaining cards instead.

Matching is idf-weighted coverage: a card seen in few archetypes identifies
hard (headline pokemon), a card in every list (basic energy, staple trainers)
barely counts. No library, thin evidence, or an ambiguous score -> None, and
the caller falls back to the mirror prior (typical in the first turns).
"""
from __future__ import annotations

import json
import os
from collections import Counter

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archetypes.json")

# Gates for a confident match. MIN_DISTINCT: distinct observed ids before we
# try (turn ~2-3). MARGIN: best idf-score must beat the runner-up by this
# factor (small: COVERAGE is the real guard; this only breaks near-ties toward
# the more specific explanation). FLOOR: absolute score floor so N universal
# staples alone never match. COVERAGE: the matched list must explain this
# fraction of ALL observed cards -- an in-library opponent is 1.0 by
# construction, while an off-library deck (e.g. our own hops mirror) leaks
# unexplained signature cards and gets rejected instead of mis-modeled.
MIN_DISTINCT = 3
MARGIN = 1.05
FLOOR = 1.0
COVERAGE = 0.85

_LIB: list[tuple[str, Counter, list[int], int]] | None = None   # (name, counts, list60, best_basic)
_IDF: dict[int, float] = {}
_CACHE: dict[frozenset, tuple[str, list[int], int] | None] = {}


def _load() -> list:
    global _LIB, _IDF
    if _LIB is not None:
        return _LIB
    try:
        with open(_PATH) as f:
            data = json.load(f)
        from .cards import get_db
        db = get_db()
        lib = []
        for a in data["archetypes"]:
            deck = list(a["deck"])
            cnt = Counter(deck)
            basics = [(n, cid) for cid, n in cnt.items() if db.is_basic_pokemon(cid)]
            best_basic = max(basics)[1] if basics else 0
            lib.append((a["name"], cnt, deck, best_basic))
        df = Counter()
        for _, cnt, _, _ in lib:
            for cid in cnt:
                df[cid] += 1
        _IDF = {cid: 1.0 / (1.0 + n) for cid, n in df.items()}
        _LIB = lib
    except Exception:
        _LIB = []
    return _LIB


def match(observed: Counter) -> tuple[str, list[int], int] | None:
    """Best-matching archetype for the opponent's visible cards.

    Returns (name, deck_list_60, best_basic_id) or None when the evidence is
    thin/ambiguous. Cached per observed multiset (the determinizer calls this
    up to DETERMINIZATIONS_PER_MOVE times per move with identical input).
    """
    lib = _load()
    if not lib or len(observed) < MIN_DISTINCT:
        return None
    key = frozenset(observed.items())
    if key in _CACHE:
        return _CACHE[key]
    scored = []
    for name, cnt, deck, bb in lib:
        s = 0.0
        for cid, k in observed.items():
            ak = cnt.get(cid)
            if ak:
                s += min(k, ak) * _IDF.get(cid, 0.5)
        scored.append((s, name, deck, bb))
    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    res = None
    if best[0] >= FLOOR and (len(scored) == 1 or best[0] >= MARGIN * scored[1][0]):
        cnt = next(c for n, c, _, _ in lib if n == best[1])
        explained = sum(min(k, cnt.get(cid, 0)) for cid, k in observed.items())
        if explained >= COVERAGE * sum(observed.values()):
            res = (best[1], best[2], best[3])
    if len(_CACHE) > 4096:
        _CACHE.clear()
    _CACHE[key] = res
    return res
