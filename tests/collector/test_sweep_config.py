"""Unit-test the config-sweep's pure logic (typed cloning + win rate). The A/B
self-play needs the engine and runs on the engine machine, not in CI."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.sweep_config import _clone_cfg, _config_attrs, win_rate


# A stand-in config with the same shape as agent/config.py (avoids importing the
# real one, which is fine here, but keeps the test self-contained and typed).
BASE = SimpleNamespace(
    MOVE_TIME_BUDGET=0.6, MAX_SIMULATIONS=400, DETERMINIZATIONS_PER_MOVE=16,
    VALUE_NET_WEIGHT=0.7, WEIGHTS_PATH="/x/weights.npz",
    _private=1, helper=lambda: 1,
)


def test_config_attrs_only_uppercase_constants():
    attrs = _config_attrs(BASE)
    assert "MOVE_TIME_BUDGET" in attrs and "VALUE_NET_WEIGHT" in attrs
    assert "_private" not in attrs and "helper" not in attrs   # skips private/callables


def test_clone_casts_to_base_type():
    v = _clone_cfg(BASE, {"VALUE_NET_WEIGHT": "0.5", "DETERMINIZATIONS_PER_MOVE": "8"})
    assert isinstance(v.VALUE_NET_WEIGHT, float) and v.VALUE_NET_WEIGHT == 0.5
    assert isinstance(v.DETERMINIZATIONS_PER_MOVE, int) and v.DETERMINIZATIONS_PER_MOVE == 8
    # untouched knobs are preserved
    assert v.MOVE_TIME_BUDGET == 0.6 and v.MAX_SIMULATIONS == 400
    assert v.WEIGHTS_PATH == "/x/weights.npz"


def test_clone_rejects_unknown_param():
    try:
        _clone_cfg(BASE, {"NOPE": 1})
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_win_rate():
    assert win_rate(6, 4) == 0.6
    assert win_rate(0, 0) == 0.0          # no decided games -> 0 (fail-safe)
    assert win_rate(5, 0) == 1.0
