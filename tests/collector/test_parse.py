"""Defensive parsing tests: both wrapper formats + malformed input never raise."""
from __future__ import annotations

import conftest as cf

from collector.parse import parse_episode, iter_main_decisions, MAIN_CONTEXT


def test_parse_steps_wrapper():
    blob = cf.make_episode_steps(winner=0)
    ep = parse_episode(blob, episode_id="ep1")
    assert ep.ok
    assert ep.episode_id == "ep1"
    assert ep.agents == ["alice", "bob"]
    assert ep.winner == 0
    assert ep.decks[0] == list(range(1, 61))
    assert ep.n_frames == 5
    assert ep.turns >= 1


def test_parse_visualize_wrapper():
    blob = cf.make_episode_visualize(winner=1)
    ep = parse_episode(blob, episode_id="ep2")
    assert ep.ok
    # No rewards array here -> winner derived from RESULT log.
    assert ep.winner == 1


def test_winner_from_rewards_priority():
    blob = cf.make_episode_steps(winner=1)
    ep = parse_episode(blob)
    assert ep.winner == 1  # rewards [0,1]


def test_draw_detection():
    blob = cf.make_episode_steps(winner=2)
    ep = parse_episode(blob)
    assert ep.winner == 2


def test_iter_main_decisions_count():
    blob = cf.make_episode_steps(winner=0)
    ep = parse_episode(blob)
    decisions = list(iter_main_decisions(ep))
    # 4 live MAIN frames; terminal frame has result set and is excluded.
    assert len(decisions) == 4
    for state, me in decisions:
        assert me in (0, 1)
        assert state["result"] == -1


def test_non_main_context_excluded():
    frames = [cf.make_frame(turn=1, your_index=0, context=3)]  # SWITCH, not MAIN
    ep = parse_episode({"visualize": frames})
    assert list(iter_main_decisions(ep)) == []


def test_malformed_inputs_never_raise():
    for bad in (None, 42, "nope", [], {}, {"steps": "x"}, {"visualize": [1, 2, 3]},
                {"steps": [[{"observation": 5}]]}, {"rewards": "bad"}):
        ep = parse_episode(bad, episode_id="x")
        # Must not raise; ok may be False.
        assert ep.episode_id == "x"
        assert isinstance(ep.winner, int)


def test_unknown_fields_tolerated():
    blob = cf.make_episode_steps(winner=0)
    blob["info"]["Agents"][0]["NewField"] = "future"
    blob["steps"][0][0]["observation"]["visualize"][0]["current"]["brandNew"] = 99
    ep = parse_episode(blob)
    assert ep.ok and ep.winner == 0


def test_wrapped_payload_unwrap():
    # Mimic the {"episode_id":..,"replay":..} envelope the client returns.
    inner = cf.make_episode_steps(winner=0)
    ep = parse_episode({"episode_id": "e9", "replay": inner})
    # parse_episode is called by collector with payload.get("replay"); but ensure
    # passing the whole envelope still unwraps safely.
    assert ep.ok
