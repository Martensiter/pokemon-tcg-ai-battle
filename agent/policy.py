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


def option_scores(obs: dict) -> list[float]:
    sel = obs["select"]
    state = obs["current"]
    ctx = sel["context"]
    opts = sel["option"]
    scores = []
    for o in opts:
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
