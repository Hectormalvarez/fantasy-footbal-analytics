"""Tests for src/cli.py -- CLI entry point with subcommands."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.cli import build_parser, _format_contingency_sheet


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _board_df():
    """Return a minimal draft board DataFrame for handler tests."""
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p3", "p4"],
        "player_name": ["Jalen Hurts", "Saquon Barkley", "AJ Brown", "Dallas Goedert"],
        "position_proj": ["QB", "RB", "WR", "TE"],
        "proj_points": [25.0, 18.0, 16.0, 10.0],
        "vorp": [15.0, 12.0, 10.0, 4.0],
        "vorp_rank": [1, 2, 3, 4],
        "search_rank": [5, 3, 8, 20],
        "adp_delta": [4, 1, 5, 16],
        "signal": ["Fair Value", "Slight Value", "Slight Value", "Major Value"],
        "pos_label": ["QB1", "RB1", "WR1", "TE1"],
        "team": ["PHI", "PHI", "PHI", "PHI"],
        "norm_name": ["jalen hurts", "saquon barkley", "aj brown", "dallas goedert"],
        "baseline": [10.0, 6.0, 6.0, 6.0],
    })


# ---------------------------------------------------------------------------
# Parser tests -- general
# ---------------------------------------------------------------------------
def test_parser_requires_command():
    """No subcommand sets command to None (dispatch handled in main)."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_parser_no_command_has_func():
    """Parsing no subcommand produces Namespace without 'func'."""
    parser = build_parser()
    args = parser.parse_args([])
    assert not hasattr(args, "func")


# ---------------------------------------------------------------------------
# Parser tests -- draft subcommand
# ---------------------------------------------------------------------------
def test_draft_requires_slot():
    """draft subcommand requires --slot."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["draft"])


def test_draft_slot_int():
    """--slot is parsed as an integer."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "7"])
    assert args.slot == 7
    assert args.command == "draft"


def test_draft_default_rounds():
    """--rounds defaults to 15."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "1"])
    assert args.rounds == 15


def test_draft_custom_rounds():
    """--rounds can be set to a custom value."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "1", "--rounds", "10"])
    assert args.rounds == 10


def test_draft_refresh_flag():
    """--refresh is a boolean flag, default False."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "1"])
    assert args.refresh is False


def test_draft_refresh_set():
    """--refresh becomes True when present."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "1", "--refresh"])
    assert args.refresh is True


def test_draft_reach_buffer_default():
    """--reach-buffer defaults to 4."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "1"])
    assert args.reach_buffer == 4


def test_draft_fall_buffer_default():
    """--fall-buffer defaults to 8."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "1"])
    assert args.fall_buffer == 8


def test_draft_all_args():
    """All draft arguments can be specified together."""
    parser = build_parser()
    args = parser.parse_args([
        "draft", "--slot", "14", "--rounds", "12",
        "--refresh", "--reach-buffer", "3", "--fall-buffer", "10",
    ])
    assert args.slot == 14
    assert args.rounds == 12
    assert args.refresh is True
    assert args.reach_buffer == 3
    assert args.fall_buffer == 10


def test_draft_slot_boundary_low():
    """Slot 1 is valid."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "1"])
    assert args.slot == 1


def test_draft_slot_boundary_high():
    """Slot 14 is valid."""
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "14"])
    assert args.slot == 14


# ---------------------------------------------------------------------------
# Parser tests -- waivers subcommand
# ---------------------------------------------------------------------------
def test_waivers_requires_league_id():
    """waivers requires --league-id."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["waivers", "--roster-id", "1"])


def test_waivers_requires_roster_id():
    """waivers requires --roster-id."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["waivers", "--league-id", "12345"])


def test_waivers_parses_args():
    """waivers parses required args correctly."""
    parser = build_parser()
    args = parser.parse_args(["waivers", "--league-id", "12345", "--roster-id", "1"])
    assert args.league_id == "12345"
    assert args.roster_id == 1
    assert args.command == "waivers"


def test_waivers_default_faab():
    """--faab defaults to 100."""
    parser = build_parser()
    args = parser.parse_args(["waivers", "--league-id", "12345", "--roster-id", "1"])
    assert args.faab == 100


def test_waivers_custom_faab():
    """--faab can be set to a custom value."""
    parser = build_parser()
    args = parser.parse_args(["waivers", "--league-id", "12345", "--roster-id", "1", "--faab", "50"])
    assert args.faab == 50


def test_waivers_output_optional():
    """--output defaults to None."""
    parser = build_parser()
    args = parser.parse_args(["waivers", "--league-id", "12345", "--roster-id", "1"])
    assert args.output is None


def test_waivers_output_path():
    """--output accepts a file path."""
    parser = build_parser()
    args = parser.parse_args(["waivers", "--league-id", "12345", "--roster-id", "1", "--output", "/tmp/test.csv"])
    assert args.output == "/tmp/test.csv"


# ---------------------------------------------------------------------------
# Parser tests -- weekly subcommand
# ---------------------------------------------------------------------------
def test_weekly_requires_league_id():
    """weekly requires --league-id."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["weekly", "--roster-id", "1", "--week", "5"])


def test_weekly_requires_roster_id():
    """weekly requires --roster-id."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["weekly", "--league-id", "12345", "--week", "5"])


def test_weekly_requires_week():
    """weekly requires --week."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["weekly", "--league-id", "12345", "--roster-id", "1"])


def test_weekly_parses_args():
    """weekly parses required args correctly."""
    parser = build_parser()
    args = parser.parse_args(["weekly", "--league-id", "12345", "--roster-id", "1", "--week", "5"])
    assert args.league_id == "12345"
    assert args.roster_id == 1
    assert args.week == 5
    assert args.command == "weekly"


def test_weekly_default_refresh():
    """--refresh defaults to False."""
    parser = build_parser()
    args = parser.parse_args(["weekly", "--league-id", "12345", "--roster-id", "1", "--week", "5"])
    assert args.refresh is False


# ---------------------------------------------------------------------------
# Parser tests -- trade subcommand
# ---------------------------------------------------------------------------
def test_trade_requires_league_id():
    """trade requires --league-id."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "trade", "--team-a-roster-id", "1", "--team-b-roster-id", "2",
            "--team-a-sends", "p1", "--team-b-sends", "p2",
        ])


def test_trade_requires_team_a_roster_id():
    """trade requires --team-a-roster-id."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "trade", "--league-id", "12345", "--team-b-roster-id", "2",
            "--team-a-sends", "p1", "--team-b-sends", "p2",
        ])


def test_trade_requires_team_b_roster_id():
    """trade requires --team-b-roster-id."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "trade", "--league-id", "12345", "--team-a-roster-id", "1",
            "--team-a-sends", "p1", "--team-b-sends", "p2",
        ])


def test_trade_requires_team_a_sends():
    """trade requires --team-a-sends."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "trade", "--league-id", "12345", "--team-a-roster-id", "1",
            "--team-b-roster-id", "2", "--team-b-sends", "p2",
        ])


def test_trade_requires_team_b_sends():
    """trade requires --team-b-sends."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "trade", "--league-id", "12345", "--team-a-roster-id", "1",
            "--team-b-roster-id", "2", "--team-a-sends", "p1",
        ])


def test_trade_parses_args():
    """trade parses required args correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "trade", "--league-id", "12345",
        "--team-a-roster-id", "1", "--team-b-roster-id", "2",
        "--team-a-sends", "p1,p2", "--team-b-sends", "p3,p4",
    ])
    assert args.league_id == "12345"
    assert args.team_a_roster_id == 1
    assert args.team_b_roster_id == 2
    assert args.team_a_sends == "p1,p2"
    assert args.team_b_sends == "p3,p4"
    assert args.command == "trade"


# ---------------------------------------------------------------------------
# Handler execution tests -- mocked
# ---------------------------------------------------------------------------
@patch("src.cli._load_draft_board")
def test_handle_draft_calls_pipeline(mock_load):
    """handle_draft calls the draft pipeline with correct args."""
    mock_load.return_value = _board_df()
    from src.cli import handle_draft
    parser = build_parser()
    args = parser.parse_args(["draft", "--slot", "7"])
    handle_draft(args)
    mock_load.assert_called_once()


@patch("src.cli.fetch_league_rosters")
@patch("src.cli._load_draft_board")
def test_handle_waivers_calls_build(mock_load, mock_fetch):
    """handle_waivers calls build_waiver_recommendations."""
    mock_load.return_value = _board_df()
    mock_fetch.return_value = [
        {"roster_id": 1, "owner_id": "u1", "players": ["p1", "p2"]},
    ]
    from src.cli import handle_waivers
    parser = build_parser()
    args = parser.parse_args(["waivers", "--league-id", "12345", "--roster-id", "1"])
    handle_waivers(args)
    mock_fetch.assert_called_once_with("12345")


@patch("src.cli.fetch_league_matchups")
@patch("src.cli._load_draft_board")
def test_handle_weekly_calls_extract(mock_load, mock_fetch):
    """handle_weekly calls extract_weekly_matchup_roster."""
    mock_load.return_value = _board_df()
    mock_fetch.return_value = [{
        "matchup_id": 1, "roster_id": 1,
        "starters": ["p1", "p2"], "players": ["p1", "p2", "p3", "p4"],
        "players_points": {"p1": 25, "p2": 18, "p3": 16, "p4": 10},
    }]
    from src.cli import handle_weekly
    parser = build_parser()
    args = parser.parse_args(["weekly", "--league-id", "12345", "--roster-id", "1", "--week", "5"])
    handle_weekly(args)
    mock_fetch.assert_called_once_with("12345", 5)


@patch("src.cli.evaluate_trade")
@patch("src.cli.fetch_league_rosters")
@patch("src.cli._load_draft_board")
def test_handle_trade_calls_evaluate(mock_load, mock_fetch, mock_trade):
    """handle_trade calls evaluate_trade with correct args."""
    mock_load.return_value = _board_df()
    mock_fetch.return_value = [
        {"roster_id": 1, "owner_id": "u1", "players": ["p1", "p2"]},
        {"roster_id": 2, "owner_id": "u2", "players": ["p3", "p4"]},
    ]
    mock_trade.return_value = {
        "team_a": {"starting_delta": 2.0, "vorp_delta": 10.0},
        "team_b": {"starting_delta": -2.0, "vorp_delta": -10.0},
        "verdict_a": "Slight Upgrade",
        "verdict_b": "Decline / Value Loss",
    }
    from src.cli import handle_trade
    parser = build_parser()
    args = parser.parse_args([
        "trade", "--league-id", "12345",
        "--team-a-roster-id", "1", "--team-b-roster-id", "2",
        "--team-a-sends", "p1", "--team-b-sends", "p3",
    ])
    handle_trade(args)
    mock_trade.assert_called_once()


# ---------------------------------------------------------------------------
# main() dispatch tests
# ---------------------------------------------------------------------------
def test_main_no_args_exits():
    """main() with no args calls help and exits."""
    from src.cli import main
    with pytest.raises(SystemExit):
        main([])


@patch("src.cli._load_draft_board")
def test_main_draft_dispatches(mock_load):
    """main() with draft subcommand dispatches to handle_draft."""
    mock_load.return_value = _board_df()
    from src.cli import main
    # Should not raise
    main(["draft", "--slot", "1"])


# ---------------------------------------------------------------------------
# _format_contingency_sheet tests
# ---------------------------------------------------------------------------
def test_format_contains_header():
    """Formatted output includes the slot header."""
    sheet = pd.DataFrame([{
        "round": 1, "pick": 14, "tier": "Primary",
        "vorp_rank": 1, "pos_label": "WR1", "player": "Ja'Marr Chase",
        "team": "CIN", "position": "WR", "proj_pts": 310.0,
        "vorp": 165.0, "search_rank": 4, "adp_delta": 3.0,
        "signal": "Fair Value",
    }])
    text = _format_contingency_sheet(sheet, slot=14, num_teams=14)
    assert "Slot 14 of 14" in text
    assert "Round  1" in text


def test_format_empty_sheet():
    """An empty sheet still produces a valid header."""
    sheet = pd.DataFrame()
    text = _format_contingency_sheet(sheet, slot=1, num_teams=14)
    assert "Slot 1 of 14" in text
    assert "no targets found" in text


def test_format_tiers_shown():
    """Primary, Secondary, and Value tiers appear when populated."""
    sheet = pd.DataFrame([
        {"round": 1, "pick": 14, "tier": "Primary", "vorp_rank": 1,
         "pos_label": "WR1", "player": "A", "team": "", "position": "WR",
         "proj_pts": 300, "vorp": 100, "search_rank": 14, "adp_delta": 5,
         "signal": "Slight Value"},
        {"round": 1, "pick": 14, "tier": "Secondary", "vorp_rank": 2,
         "pos_label": "RB2", "player": "B", "team": "", "position": "RB",
         "proj_pts": 250, "vorp": 60, "search_rank": 12, "adp_delta": 3,
         "signal": "Fair Value"},
        {"round": 1, "pick": 14, "tier": "Value", "vorp_rank": 3,
         "pos_label": "QB3", "player": "C", "team": "", "position": "QB",
         "proj_pts": 200, "vorp": 0, "search_rank": 30, "adp_delta": 20,
         "signal": "Major Value"},
    ])
    text = _format_contingency_sheet(sheet, slot=14, num_teams=14)
    assert "[Primary]" in text
    assert "[Secondary]" in text
    assert "[Value]" in text
