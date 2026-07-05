"""Heuristic action policy.

`option_scores` assigns a desirability score to every option of a decision;
`choose` turns those scores into a contract-valid selection (right count, unique,
in range). Used directly by the greedy baseline and as the rollout/prior policy
inside MCTS. It is deliberately fast and side-effect free (no engine calls).
"""
from __future__ import annotations

import random

from cg.api import OptionType, SelectContext
from .cards import get_db
from .observation import my_state, opp_state, active_of, total_energy
from .evaluate import attack_damage_estimate

# Contexts where selecting MORE is good (take the max count).
_BENEFICIAL_COUNT = {
    SelectContext.DRAW_COUNT.value,
    SelectContext.REMOVE_DAMAGE_COUNTER_COUNT.value,
    SelectContext.DAMAGE_COUNTER_COUNT.value,
    SelectContext.TO_HAND.value,
    SelectContext.HEAL.value,
    SelectContext.DAMAGE.value,
    SelectContext.DAMAGE_COUNTER.value,
    SelectContext.DAMAGE_COUNTER_ANY.value,
    SelectContext.REMOVE_DAMAGE_COUNTER.value,
    SelectContext.TO_BENCH.value,
}
# Contexts where selecting is a cost (take the min count).
_COSTLY_COUNT = {
    SelectContext.DISCARD.value,
    SelectContext.DISCARD_ENERGY.value,
    SelectContext.DISCARD_ENERGY_CARD.value,
    SelectContext.DISCARD_TOOL_CARD.value,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD.value,
    SelectContext.TO_DECK.value,
    SelectContext.TO_DECK_BOTTOM.value,
    SelectContext.TO_DECK_ENERGY.value,
    SelectContext.TO_PRIZE.value,
}


def _card_id_from_option(o: dict, state: dict) -> int | None:
    """Resolve the card id an option refers to, when locatable from the state."""
    area = o.get("area")
    idx = o.get("index")
    pi = o.get("playerIndex")
    if area is None or idx is None:
        return None
    try:
        player = state["players"][pi if pi is not None else state["yourIndex"]]
    except (KeyError, IndexError, TypeError):
        return None
    # AreaType: 2=HAND 3=DISCARD 5=BENCH 6=PRIZE ...
    area_key = {2: "hand", 3: "discard", 5: "bench"}.get(area)
    if area_key:
        arr = player.get(area_key) or []
        if 0 <= idx < len(arr) and arr[idx] is not None:
            return arr[idx].get("id")
    return None


def _score_attack(o: dict, state: dict) -> float:
    db = get_db()
    me = state["yourIndex"]
    mp = state["players"][me]
    op = state["players"][1 - me]
    my_act = active_of(mp)
    op_act = active_of(op)
    ai = db.attack(o.get("attackId")) if o.get("attackId") is not None else None
    base = ai.damage if ai else 0
    if my_act and op_act and base > 0:
        dmg = attack_damage_estimate(my_act["id"], op_act["id"], base)
        if dmg >= op_act.get("hp", 9999):
            return 100.0 + dmg / 100.0  # KO — best possible
        return 30.0 + dmg / 10.0
    # zero-damage / effect attacks: still usually worth doing
    return 20.0


def _score_play(o: dict, state: dict) -> float:
    db = get_db()
    mp = my_state(state)
    cid = _card_id_from_option(o, state)
    hand_n = mp.get("handCount", 0)
    bench_n = len(mp.get("bench") or [])
    if cid is None:
        return 5.0
    ct = db.card_type(cid)
    if ct is None:
        return 5.0
    name = ct.name
    if name == "POKEMON":
        # develop the bench, with diminishing returns
        return 12.0 - 2.0 * bench_n if bench_n < 5 else -1.0
    if name == "SUPPORTER":
        # draw/search supporters are better when our hand is thin
        return 9.0 + max(0, 6 - hand_n)
    if name == "ITEM":
        return 7.0
    if name == "TOOL":
        return 6.0
    if name == "STADIUM":
        return 5.5
    return 5.0


def _score_attach(o: dict, state: dict) -> float:
    """Energy attach: prefer loading the active attacker that still needs energy."""
    mp = my_state(state)
    in_play_area = o.get("inPlayArea")
    # AreaType ACTIVE=4, BENCH=5
    act = active_of(mp)
    if in_play_area == 4 and act is not None:
        need = 3 - total_energy(act)
        return 14.0 + max(0, need) * 2.0
    if in_play_area == 5:
        return 8.0
    return 7.0


def _score_main_option(o: dict, state: dict) -> float:
    t = o.get("type")
    if t == OptionType.ATTACK.value:
        return _score_attack(o, state)
    if t == OptionType.PLAY.value:
        return _score_play(o, state)
    if t == OptionType.ATTACH.value:
        return _score_attach(o, state)
    if t == OptionType.ABILITY.value:
        return 10.0
    if t == OptionType.EVOLVE.value:
        return 11.0
    if t == OptionType.RETREAT.value:
        return 2.0
    if t == OptionType.DISCARD.value:
        return 1.0
    if t == OptionType.END.value:
        return 3.0  # baseline: act if anything useful exists, else end
    return 4.0


def _score_yesno(o: dict, ctx: int) -> float:
    is_yes = (o.get("type") == OptionType.YES.value)
    # default YES for beneficial activations; specific contexts overridden below
    if ctx == SelectContext.MULLIGAN.value:
        return 1.0 if not is_yes else 0.0          # keep hand
    if ctx == SelectContext.IS_FIRST.value:
        return 1.0 if is_yes else 0.0              # go first
    # ACTIVATE / FIRST_EFFECT / COIN_HEAD / others: prefer YES
    return 1.0 if is_yes else 0.0


def _score_card_select(o: dict, ctx: int, state: dict) -> float:
    """Generic card-target scoring for the many CARD contexts."""
    db = get_db()
    cid = _card_id_from_option(o, state)
    # Setup / put-into-play: choose the best, most energy-efficient attacker.
    if ctx in (SelectContext.SETUP_ACTIVE_POKEMON.value, SelectContext.TO_ACTIVE.value):
        if cid is not None:
            a = db.best_attack(cid)
            if a:
                # lower cost online sooner; some weight on damage
                return 20.0 - 3.0 * a.cost + a.damage / 50.0
        return 5.0
    if ctx in (SelectContext.SETUP_BENCH_POKEMON.value, SelectContext.TO_BENCH.value,
               SelectContext.TO_FIELD.value):
        return 10.0
    # Damage / KO targeting: prefer the opponent's strongest (handled by sign of count)
    return 5.0


# --- v2 targeted scorers (audit-driven; see config.GREEDY_V2) ------------------
# Every scorer below only uses zones the observation actually reveals (hand /
# discard / board). Deck-area options are information-blind live (players[me]
# .deck is None even during searches) so those contexts are deliberately left
# on v1 behaviour.

_HEAL_CTX = {SelectContext.HEAL.value, SelectContext.REMOVE_DAMAGE_COUNTER.value}


def _v2_score(o: dict, ctx: int, state: dict) -> float | None:
    """Dispatch to a v2 scorer, or None to fall through to v1 scoring.

    SCOPED to DAMAGE_COUNTER_ANY ("place damage counters anywhere") only. The
    78-game replay audit proved v1 was already at/above the random baseline
    everywhere else (DISCARD 86%, DAMAGE_COUNTER 88%, TO_ACTIVE 41%,
    DISCARD_ENERGY 71%); dispatching v2 scorers there REGRESSED them
    (86->60, 88->41, 41->18). Only "place damage anywhere" was genuinely below
    random (15.6% vs 25.2%), and the KO-math scorer fixes it to 67.2%. Do NOT
    widen this without an audit proving the new context is broken first.
    """
    if ctx == SelectContext.DAMAGE_COUNTER_ANY.value:
        return _score_damage_target(o, ctx, state)
    return None


def _resolve_board_pokemon(o: dict, state: dict):
    """(pokemon dict | None, owner index) for an option that targets the board."""
    pi = o.get("playerIndex")
    if pi not in (0, 1):
        pi = state.get("yourIndex", 0)
    try:
        pl = state["players"][pi]
    except (KeyError, IndexError, TypeError):
        return None, pi
    area, idx = o.get("area"), o.get("index")
    if area == 4:  # ACTIVE
        a = pl.get("active") or []
        return (a[0] if a and a[0] is not None else None), pi
    if area == 5 and isinstance(idx, int):  # BENCH
        b = pl.get("bench") or []
        if 0 <= idx < len(b) and b[idx] is not None:
            return b[idx], pi
    return None, pi


def _score_damage_target(o: dict, ctx: int, state: dict) -> float:
    """Damage-counter placement / healing targets, by KO math and L1 value."""
    pkm, owner = _resolve_board_pokemon(o, state)
    if pkm is None:
        return 5.0
    db = get_db()
    me = state.get("yourIndex", 0)
    hp = pkm.get("hp", 999) or 999
    val = db.pokemon_value(pkm.get("id"))
    if ctx in _HEAL_CTX:
        # heal our own: the more valuable and the more damaged, the better
        dmg_taken = max(0, (pkm.get("maxHp", hp) or hp) - hp)
        side = 1.0 if owner == me else -1.0
        return side * (10.0 + val / 40.0 + dmg_taken / 20.0)
    # placing damage: pile onto the opponent's nearly-KO'd, high-value mons;
    # if forced to hit our own board, hurt the least valuable, healthiest one.
    if owner != me:
        return 20.0 + max(0.0, 40.0 - hp / 10.0) + val / 40.0
    return 5.0 - val / 100.0 + hp / 500.0


def option_scores(obs: dict) -> list[float]:
    sel = obs["select"]
    state = obs["current"]
    ctx = sel["context"]
    opts = sel["option"]
    from . import config as C
    v2 = float(getattr(C, "GREEDY_V2", 0.0)) > 0
    scores = []
    for o in opts:
        if v2:
            s2 = _v2_score(o, ctx, state)
            if s2 is not None:
                scores.append(s2)
                continue
        t = o.get("type")
        if t in (OptionType.YES.value, OptionType.NO.value):
            s = _score_yesno(o, ctx)
        elif t in (OptionType.CARD.value, OptionType.TOOL_CARD.value, OptionType.ENERGY_CARD.value):
            s = _score_card_select(o, ctx, state)
        elif t == OptionType.ENERGY.value:
            s = 5.0
        elif t == OptionType.NUMBER.value:
            s = float(o.get("number") or 0)
        elif t == OptionType.SPECIAL_CONDITION.value:
            s = 5.0
        elif t == OptionType.SKILL.value:
            s = 5.0
        else:
            s = _score_main_option(o, state)
        scores.append(s)
    return scores


def _choose_count(sel: dict) -> int:
    lo, hi = sel["minCount"], sel["maxCount"]
    n = len(sel["option"])
    hi = min(hi, n)
    if hi <= lo:
        return max(0, lo)
    ctx = sel["context"]
    if ctx in _BENEFICIAL_COUNT:
        return hi
    if ctx in _COSTLY_COUNT:
        return lo
    # default: act minimally but at least once when allowed
    return lo if lo > 0 else 1


def choose(obs: dict, rng: random.Random | None = None, epsilon: float = 0.0) -> list[int]:
    """Greedy (optionally epsilon-noisy) contract-valid selection."""
    sel = obs["select"]
    n = len(sel["option"])
    if n == 0:
        return []
    k = _choose_count(sel)
    if k <= 0:
        return []
    rng = rng or random
    if epsilon > 0 and rng.random() < epsilon:
        return rng.sample(range(n), min(k, n))
    scores = option_scores(obs)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return sorted(order[:k])
