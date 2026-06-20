"""Card database: a thin, cached wrapper over the engine's card/attack data.

Shared by the agent (evaluation, determinization) and the deck-building tools.
Joins engine `CardData`/`Attack` with optional CSV names. No pandas — stdlib only.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

from cg.api import (
    all_card_data, all_attack, CardData, Attack, CardType, EnergyType,
)

# Basic energy card ids (SVE set, from EN_Card_Data.csv ids 1..9).
# 1={G} 2={R} 3={W} 4={L} 5={P} 6={F} 7={D} 8={M} 9={C}
BASIC_ENERGY_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9}
ENERGY_ID_BY_TYPE = {
    EnergyType.GRASS: 1, EnergyType.FIRE: 2, EnergyType.WATER: 3,
    EnergyType.LIGHTNING: 4, EnergyType.PSYCHIC: 5, EnergyType.FIGHTING: 6,
    EnergyType.DARKNESS: 7, EnergyType.METAL: 8, EnergyType.COLORLESS: 9,
}


@dataclass
class AttackInfo:
    attackId: int
    name: str
    text: str
    damage: int
    energies: list  # list[EnergyType-int]
    cost: int       # number of energies required


class CardDB:
    """Cached view of all card + attack data from the engine."""

    def __init__(self):
        self._cards: dict[int, CardData] = {c.cardId: c for c in all_card_data()}
        self._attacks: dict[int, Attack] = {a.attackId: a for a in all_attack()}

    # ---- lookups ----
    def card(self, cid: int) -> CardData | None:
        return self._cards.get(cid)

    def name(self, cid: int) -> str:
        c = self._cards.get(cid)
        return c.name if c and c.name else f"#{cid}"

    def all_cards(self) -> dict[int, CardData]:
        return self._cards

    def attack(self, aid: int) -> AttackInfo | None:
        a = self._attacks.get(aid)
        if a is None:
            return None
        return AttackInfo(a.attackId, a.name, a.text, a.damage, list(a.energies), len(a.energies))

    def attacks_of(self, cid: int) -> list[AttackInfo]:
        c = self._cards.get(cid)
        if not c:
            return []
        out = []
        for aid in c.attacks:
            ai = self.attack(aid)
            if ai:
                out.append(ai)
        return out

    def best_attack(self, cid: int) -> AttackInfo | None:
        best = None
        for ai in self.attacks_of(cid):
            if best is None or ai.damage > best.damage:
                best = ai
        return best

    # ---- predicates ----
    def is_pokemon(self, cid: int) -> bool:
        c = self._cards.get(cid)
        return bool(c and c.cardType == CardType.POKEMON.value)

    def is_basic_pokemon(self, cid: int) -> bool:
        c = self._cards.get(cid)
        return bool(c and c.cardType == CardType.POKEMON.value and c.basic)

    def is_energy(self, cid: int) -> bool:
        c = self._cards.get(cid)
        return bool(c and c.cardType in (CardType.BASIC_ENERGY.value, CardType.SPECIAL_ENERGY.value))

    def is_basic_energy(self, cid: int) -> bool:
        c = self._cards.get(cid)
        return bool(c and c.cardType == CardType.BASIC_ENERGY.value)

    def is_ace_spec(self, cid: int) -> bool:
        c = self._cards.get(cid)
        return bool(c and c.aceSpec)

    def card_type(self, cid: int):
        c = self._cards.get(cid)
        return CardType(c.cardType) if c else None


@functools.lru_cache(maxsize=1)
def get_db() -> CardDB:
    """Process-wide singleton CardDB (engine data is immutable)."""
    return CardDB()


# ---------------------------------------------------------------------------
# Deck legality
# ---------------------------------------------------------------------------

def validate_deck(ids: list[int]) -> tuple[bool, list[str]]:
    """Check the core construction rules. Returns (ok, problems)."""
    db = get_db()
    problems: list[str] = []

    if len(ids) != 60:
        problems.append(f"deck must be exactly 60 cards, got {len(ids)}")

    # >=1 Basic Pokemon
    if not any(db.is_basic_pokemon(c) for c in ids):
        problems.append("deck must contain at least 1 Basic Pokemon")

    # <=4 copies of any non-basic-energy card id
    counts: dict[int, int] = {}
    for c in ids:
        counts[c] = counts.get(c, 0) + 1
    for cid, n in counts.items():
        if cid in BASIC_ENERGY_IDS:
            continue
        if db.is_basic_energy(cid):
            continue
        if n > 4:
            problems.append(f"card {cid} ({db.name(cid)}) appears {n} times (max 4)")

    # <=1 ACE SPEC total
    ace = sum(n for cid, n in counts.items() if db.is_ace_spec(cid))
    if ace > 1:
        problems.append(f"deck has {ace} ACE SPEC cards (max 1)")

    # all ids must exist
    for cid in counts:
        if db.card(cid) is None:
            problems.append(f"unknown card id {cid}")

    return (len(problems) == 0, problems)
