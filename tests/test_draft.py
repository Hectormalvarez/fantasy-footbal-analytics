"""Tests for src/draft.py — serpentine draft matrix and contingency sheets."""

from src.draft import get_snake_picks


def test_slot_1_round_1():
    """Slot 1 picks 1st overall in round 1."""
    picks = get_snake_picks(1, num_teams=14, rounds=1)
    assert picks == [(1, 1)]


def test_slot_14_round_1():
    """Slot 14 picks 14th overall in round 1."""
    picks = get_snake_picks(14, num_teams=14, rounds=1)
    assert picks == [(1, 14)]


def test_slot_1_round_2_reversed():
    """In round 2, slot 1 picks last (28th overall in a 14-team league)."""
    picks = get_snake_picks(1, num_teams=14, rounds=2)
    assert picks[1] == (2, 28)


def test_slot_14_round_2_turn():
    """Slot 14 gets the turn: picks 14 then 15."""
    picks = get_snake_picks(14, num_teams=14, rounds=2)
    assert picks == [(1, 14), (2, 15)]


def test_slot_14_first_6_picks():
    """Slot 14 serpentine: 14, 15, 42, 43, 70, 71."""
    picks = get_snake_picks(14, num_teams=14, rounds=6)
    expected = [(1, 14), (2, 15), (3, 42), (4, 43), (5, 70), (6, 71)]
    assert picks == expected


def test_slot_1_first_6_picks():
    """Slot 1 serpentine: 1, 28, 29, 56, 57, 84."""
    picks = get_snake_picks(1, num_teams=14, rounds=6)
    expected = [(1, 1), (2, 28), (3, 29), (4, 56), (5, 57), (6, 84)]
    assert picks == expected


def test_10_team_league():
    """Verify correct picks in a 10-team league."""
    picks = get_snake_picks(5, num_teams=10, rounds=3)
    expected = [(1, 5), (2, 16), (3, 25)]
    assert picks == expected
