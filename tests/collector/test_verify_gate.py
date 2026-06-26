"""Unit-test the verify gate (pure logic; the A/B games need the engine and are
exercised on the engine machine, not here)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.verify_candidate import passes


def test_pass_above_threshold():
    assert passes(55, 45, 0.53) is True       # 55% >= 53%
    assert passes(53, 47, 0.53) is True


def test_fail_below_threshold():
    assert passes(50, 50, 0.53) is False       # tie fails a >50% gate
    assert passes(40, 60, 0.53) is False


def test_zero_decided_fails():
    assert passes(0, 0, 0.53) is False          # all draws / no games -> fail safe
