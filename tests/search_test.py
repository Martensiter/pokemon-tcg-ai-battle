"""De-risk the determinized search API (the core MCTS primitive).

Plays a random game until the player-to-move faces a MAIN decision, then calls
search_begin() with a (here cheating, since we know both decks) determinization,
walks a few search_step()s forward, and tears the search down.

Run from project root:  python tests/search_test.py
"""
import os
import sys
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.api import (  # noqa: E402
    all_card_data, to_observation_class, search_begin, search_step,
    search_end, SelectContext,
)


def read_deck(path):
    with open(path) as f:
        return [int(x) for x in f.read().split("\n") if x.strip()]


def random_legal_choice(sel):
    n = len(sel["option"])
    lo, hi = sel["minCount"], min(sel["maxCount"], n)
    if hi <= 0:
        return []
    k = random.randint(lo, hi) if hi >= lo else lo
    return random.sample(range(n), max(0, min(k, n)))


def main():
    cards = {c.cardId: c for c in all_card_data()}
    basics = [cid for cid, c in cards.items() if c.basic]

    deck = read_deck(os.path.join(ROOT, "deck_sample.csv"))
    obs, start = battle_start(deck, list(deck))
    assert obs is not None

    # advance to a mid-game MAIN decision
    target = None
    for _ in range(400):
        state = obs.get("current")
        if state and state.get("result", -1) != -1:
            break
        sel = obs.get("select")
        if sel is None:
            break
        if sel["context"] == SelectContext.MAIN.value and state["turn"] >= 2:
            target = obs
            break
        obs = battle_select(random_legal_choice(sel))

    if target is None:
        print("did not reach a MAIN decision; abort")
        battle_finish()
        return

    state = target["current"]
    me = state["yourIndex"]
    opp = 1 - me
    my = state["players"][me]
    op = state["players"][opp]

    print(f"reached MAIN: turn={state['turn']} me={me} "
          f"myDeck={my['deckCount']} myPrize={len(my['prize'])} "
          f"oppDeck={op['deckCount']} oppPrize={len(op['prize'])} oppHand={op['handCount']}")

    agent_obs = to_observation_class(target)

    # Determinization. We "cheat" here (we know the deck) just to prove the API works.
    your_deck = (deck * 2)[: my["deckCount"]]
    your_prize = (deck * 2)[: len(my["prize"])]
    opponent_deck = (deck * 2)[: op["deckCount"]]
    # guarantee >=1 basic in opp deck
    if opponent_deck and not any(cards[c].basic for c in opponent_deck if c in cards):
        opponent_deck[0] = basics[0]
    opponent_prize = (deck * 2)[: len(op["prize"])]
    opponent_hand = (deck * 2)[: op["handCount"]]
    opponent_active = []
    active = op["active"]
    if len(active) > 0 and active[0] is None:
        opponent_active = [basics[0]]

    root = search_begin(
        agent_obs, your_deck, your_prize, opponent_deck,
        opponent_prize, opponent_hand, opponent_active, manual_coin=False,
    )
    print(f"search_begin OK: searchId={root.searchId} "
          f"sel.context={SelectContext(root.observation.select.context).name} "
          f"nOpt={len(root.observation.select.option)}")

    # walk a few steps forward inside the search
    sid = root.searchId
    sel = root.observation.select
    for d in range(6):
        if sel is None:
            print(f"  depth {d}: terminal/no-select")
            break
        n = len(sel.option)
        lo, hi = sel.minCount, min(sel.maxCount, n)
        if hi <= 0:
            choice = []
        else:
            k = random.randint(lo, hi) if hi >= lo else lo
            choice = random.sample(range(n), max(0, min(k, n)))
        nxt = search_step(sid, choice)
        cur = nxt.observation.current
        res = cur.result if cur else -1
        ctx = SelectContext(nxt.observation.select.context).name if nxt.observation.select else "None"
        print(f"  depth {d}: chose {choice} -> next ctx={ctx} result={res}")
        sid = nxt.searchId
        sel = nxt.observation.select
        if res != -1:
            print("  reached terminal inside search")
            break

    search_end()
    battle_finish()
    print("search API smoke test PASSED")


if __name__ == "__main__":
    main()
