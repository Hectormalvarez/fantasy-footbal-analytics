"""Tests for src/draft.py — serpentine draft matrix and contingency sheets."""

import pandas as pd

from src.draft import get_snake_picks, generate_contingency_sheet


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



# ---------------------------------------------------------------------------
# generate_contingency_sheet tests
# ---------------------------------------------------------------------------
def _board_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal draft board from row dicts."""
    defaults = {"vorp": 0.0, "vorp_rank": 1, "pos_rank": 1, "pos_label": "WR1",
                "search_rank": 1, "adp_delta": 0, "signal": "Fair Value"}
    for row in rows:
        for k, v in defaults.items():
            row.setdefault(k, v)
    return pd.DataFrame(rows)


def test_returns_dataframe():
    """generate_contingency_sheet returns a DataFrame."""
    board = _board_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 300.0,
         "search_rank": 14, "vorp": 100.0, "vorp_rank": 1},
    ])
    sheet = generate_contingency_sheet(board, draft_slot=14, num_teams=14, rounds=1)
    assert isinstance(sheet, pd.DataFrame)


def test_has_round_column():
    """Each row has a round column."""
    board = _board_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 300.0,
         "search_rank": 14, "vorp": 100.0, "vorp_rank": 1},
    ])
    sheet = generate_contingency_sheet(board, draft_slot=14, num_teams=14, rounds=2)
    assert "round" in sheet.columns


def test_has_tier_column():
    """Each row has a tier column."""
    board = _board_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 300.0,
         "search_rank": 14, "vorp": 100.0, "vorp_rank": 1},
    ])
    sheet = generate_contingency_sheet(board, draft_slot=14, num_teams=14, rounds=1)
    assert "tier" in sheet.columns
    assert set(sheet["tier"]).issubset({"Primary", "Secondary", "Value"})


def test_empty_when_no_targets():
    """If no players in ADP window, sheet is empty for that round."""
    board = _board_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 300.0,
         "search_rank": 200, "vorp": 100.0, "vorp_rank": 1},
    ])
    sheet = generate_contingency_sheet(board, draft_slot=14, num_teams=14, rounds=1,
                                       reach_buffer=4, fall_buffer=8)
    # Pick 14, window [10, 22], player has search_rank=200 -> no match
    assert len(sheet) == 0


def test_player_in_window_appears():
    """A player with search_rank within the ADP window appears in the sheet."""
    board = _board_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 300.0,
         "search_rank": 14, "vorp": 100.0, "vorp_rank": 1},
    ])
    sheet = generate_contingency_sheet(board, draft_slot=14, num_teams=14, rounds=1,
                                       reach_buffer=4, fall_buffer=8)
    # Pick 14, window [10, 22], player search_rank=14 -> in window
    assert len(sheet) > 0
    assert sheet.iloc[0]["player"] == "A"