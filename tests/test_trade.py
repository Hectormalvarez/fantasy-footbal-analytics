"""Tests for src/trade.py — multi-team trade evaluator & lineup impact engine."""

import pandas as pd
import pytest

from src.trade import simulate_roster_swap


# ---------------------------------------------------------------------------
# simulate_roster_swap
# ---------------------------------------------------------------------------
def test_simulate_swap_basic():
    """Basic 1-for-1 swap removes outgoing and adds incoming."""
    roster = ["p1", "p2", "p3", "p4"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p5"])
    assert "p5" in result
    assert "p2" not in result
    assert len(result) == len(roster)


def test_simulate_swap_preserves_other_players():
    """Players not involved in the trade remain on the roster."""
    roster = ["p1", "p2", "p3", "p4"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p5"])
    assert set(result) == {"p1", "p3", "p4", "p5"}


def test_simulate_swap_2_for_1():
    """2-for-1 consolidation: two outgoing, one incoming."""
    roster = ["p1", "p2", "p3", "p4"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2", "p3"], incoming_ids=["p5"])
    assert set(result) == {"p1", "p4", "p5"}
    assert len(result) == 3


def test_simulate_swap_1_for_2():
    """1-for-2 expansion: one outgoing, two incoming."""
    roster = ["p1", "p2", "p3"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p5", "p6"])
    assert set(result) == {"p1", "p3", "p5", "p6"}
    assert len(result) == 4


def test_simulate_swap_outgoing_not_on_roster():
    """Trading a player not on the roster raises ValueError."""
    roster = ["p1", "p2"]
    with pytest.raises(ValueError, match="not on roster"):
        simulate_roster_swap(roster, outgoing_ids=["p99"], incoming_ids=["p3"])


def test_simulate_swap_incoming_already_on_roster():
    """Acquiring a player already on the roster raises ValueError."""
    roster = ["p1", "p2", "p3"]
    with pytest.raises(ValueError, match="already on roster"):
        simulate_roster_swap(roster, outgoing_ids=["p1"], incoming_ids=["p2"])


def test_simulate_swap_noop():
    """Empty outgoing list returns the original roster unchanged."""
    roster = ["p1", "p2", "p3"]
    result = simulate_roster_swap(roster, outgoing_ids=[], incoming_ids=[])
    assert set(result) == {"p1", "p2", "p3"}


def test_simulate_swap_returns_list():
    """Result is a list of strings."""
    roster = ["p1", "p2"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p3"])
    assert isinstance(result, list)
    assert all(isinstance(pid, str) for pid in result)
