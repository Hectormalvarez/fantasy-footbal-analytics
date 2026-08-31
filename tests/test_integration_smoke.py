"""Integration smoke tests for all in-season CLI subcommands."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.cli import build_parser
from src.reports import generate_weekly_digest, export_digest_markdown, export_power_rankings_csv
from src.waivers import build_waiver_recommendations
from src.matchups import extract_weekly_matchup_roster, optimize_starting_lineup
from src.trade import evaluate_trade
from src.simulation import simulate_team_matchup


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _board_df():
    """Minimal draft board with player_id column."""
    return pd.DataFrame({
        "player_id": ["7564", "8138", "6770", "4866", "12526",
                        "11628", "7553", "12501", "8134", "9493",
                        "4199", "11637", "4039", "6804", "11647"],
        "player_name": [
            "Ja'Marr Chase", "James Cook", "Joe Burrow",
            "Saquon Barkley", "Tetairoa McMillan", "Marvin Harrison",
            "Kyle Pitts", "Matthew Golden", "Khalil Shakir",
            "Puka Nacua", "Aaron Jones", "Keon Coleman",
            "Cooper Kupp", "Jordan Love", "Kimani Vidal",
        ],
        "position_proj": ["WR", "RB", "QB", "RB", "WR",
                           "WR", "TE", "WR", "WR", "WR",
                           "RB", "WR", "WR", "QB", "RB"],
        "proj_points": [310.0, 290.0, 280.0, 235.0, 195.0,
                         145.0, 110.0, 145.0, 145.0, 310.0,
                         140.0, 145.0, 145.0, 200.0, 140.0],
        "team": ["CIN", "BUF", "CIN", "NYG", "ARI",
                  "ARI", "ATL", "GB", "BUF", "LAR",
                  "NYG", "BUF", "LAR", "GB", "LAC"],
        "norm_name": [
            "ja'marr chase", "james cook", "joe burrow",
            "saquon barkley", "tetairoa mcmillan", "marvin harrison",
            "kyle pitts", "matthew golden", "khalil shakir",
            "puka nacua", "aaron jones", "keon coleman",
            "cooper kupp", "jordan love", "kimani vidal",
        ],
        "baseline": [250.0, 200.0, 200.0, 200.0, 150.0,
                      150.0, 100.0, 150.0, 150.0, 250.0,
                      150.0, 150.0, 150.0, 200.0, 150.0],
        "vorp": [60.0, 90.0, 80.0, 35.0, 45.0,
                  -5.0, 10.0, -5.0, -5.0, 60.0,
                  -10.0, -5.0, -5.0, 0.0, -10.0],
        "vorp_rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "pos_label": ["WR1", "RB1", "QB1", "RB2", "WR2",
                       "WR3", "TE1", "WR4", "WR5", "WR6",
                       "RB3", "WR7", "WR8", "QB2", "RB4"],
        "search_rank": [4, 8, 3, 5, 15, 20, 25, 30, 35, 6, 22, 28, 32, 12, 40],
        "adp_delta": [3, 6, 1, 1, 10, 14, 18, 22, 26, 4, 11, 16, 19, 2, 25],
        "signal": ["Fair Value", "Slight Value", "Fair Value",
                    "Fair Value", "Slight Value", "Slight Reach",
                    "Major Reach", "Major Reach", "Major Reach",
                    "Fair Value", "Slight Reach", "Major Reach",
                    "Major Reach", "Fair Value", "Major Reach"],
    })


def _matchups_raw():
    """Raw matchup data for roster 14 vs roster 8."""
    return [
        {
            "roster_id": 14, "matchup_id": 5,
            "starters": ["6770", "8138", "4866", "12526", "11628",
                          "7553", "12501", "8134", "11792", "SEA"],
            "players": ["11628", "11637", "11647", "11792", "12501",
                         "12526", "4039", "4199", "4866", "6770",
                         "6804", "7553", "8134", "8138", "SEA"],
            "players_points": {},
        },
        {
            "roster_id": 8, "matchup_id": 5,
            "starters": ["7547", "8112", "6797", "5892", "6794",
                          "1466", "5001", "9754", "11533", "DEN"],
            "players": ["7547", "8112", "6797", "5892", "6794",
                         "1466", "5001", "9754", "11533", "DEN"],
            "players_points": {},
        },
    ]


def _rosters_df():
    """Exploded roster DataFrame for 14 teams."""
    rows = []
    for rid in range(1, 15):
        for pid in [f"p{rid}_a", f"p{rid}_b", f"p{rid}_c"]:
            rows.append({"roster_id": rid, "owner_id": f"u{rid}", "player_id": pid})
    return pd.DataFrame(rows)


def _catalog_df():
    """Minimal catalog with injury fields."""
    return pd.DataFrame({
        "player_id": ["7564", "8138", "6770", "4866", "12526",
                        "11628", "7553", "12501", "8134", "9493"],
        "player_name": [
            "Ja'Marr Chase", "James Cook", "Joe Burrow",
            "Saquon Barkley", "Tetairoa McMillan", "Marvin Harrison",
            "Kyle Pitts", "Matthew Golden", "Khalil Shakir", "Puka Nacua",
        ],
        "position": ["WR", "RB", "QB", "RB", "WR", "WR", "TE", "WR", "WR", "WR"],
        "injury_status": [None, None, None, None, None, None, None, None, None, None],
        "injury_body_part": [None, None, None, None, None, None, None, None, None, None],
        "injury_notes": [None, None, None, None, None, None, None, None, None, None],
    })


# ---------------------------------------------------------------------------
# Parser smoke tests
# ---------------------------------------------------------------------------
def test_parser_weekly_args():
    """Weekly subcommand parses required args."""
    parser = build_parser()
    args = parser.parse_args([
        "weekly", "--league-id", "12345", "--roster-id", "1", "--week", "1",
    ])
    assert args.command == "weekly"
    assert args.league_id == "12345"
    assert args.roster_id == 1
    assert args.week == 1


def test_parser_waivers_args():
    """Waivers subcommand parses required args."""
    parser = build_parser()
    args = parser.parse_args([
        "waivers", "--league-id", "12345", "--roster-id", "1",
    ])
    assert args.command == "waivers"
    assert args.league_id == "12345"
    assert args.roster_id == 1
    assert args.faab == 100


def test_parser_report_args():
    """Report subcommand parses required args."""
    parser = build_parser()
    args = parser.parse_args([
        "report", "--league-id", "12345", "--roster-id", "1", "--week", "5",
    ])
    assert args.command == "report"
    assert args.league_id == "12345"
    assert args.roster_id == 1
    assert args.week == 5
    assert args.output_dir == "reports"


def test_parser_trade_args():
    """Trade subcommand parses required args."""
    parser = build_parser()
    args = parser.parse_args([
        "trade", "--league-id", "12345",
        "--team-a-roster-id", "1", "--team-b-roster-id", "2",
        "--team-a-sends", "p1", "--team-b-sends", "p2",
    ])
    assert args.command == "trade"
    assert args.team_a_roster_id == 1
    assert args.team_b_roster_id == 2


# ---------------------------------------------------------------------------
# Weekly handler pipeline test
# ---------------------------------------------------------------------------
def test_weekly_lineup_optimization():
    """Optimize starting lineup from roster player_ids."""
    board = _board_df()
    roster_ids = ["7564", "8138", "6770", "4866", "12526",
                  "11628", "7553", "12501", "8134"]
    proj = board[["player_id", "player_name", "position_proj", "proj_points"]].copy()
    proj.rename(columns={"position_proj": "position"}, inplace=True)
    result = optimize_starting_lineup(roster_ids, proj)
    assert "lineup" in result
    assert "total_proj_points" in result
    assert "bench" in result
    assert len(result["lineup"]) > 0
    assert result["total_proj_points"] > 0


def test_weekly_matchup_extraction():
    """Extract roster matchup from raw data."""
    board = _board_df()
    player_pool = board[["player_id", "player_name", "position_proj"]].copy()
    player_pool.rename(columns={"position_proj": "position"}, inplace=True)
    matchup_df = extract_weekly_matchup_roster(14, _matchups_raw(), player_pool)
    assert not matchup_df.empty
    assert "is_starter" in matchup_df.columns
    assert "opponent_roster_id" in matchup_df.columns
    assert matchup_df["opponent_roster_id"].iloc[0] == 8


def test_weekly_simulation_returns_valid_prob():
    """Monte Carlo simulation returns valid win probability."""
    board = _board_df()
    team_a = board[board["player_id"].isin(
        ["7564", "8138", "6770", "4866", "12526",
         "11628", "7553", "12501", "8134"]
    )][["player_id", "position_proj", "proj_points"]].copy()
    team_a.rename(columns={"position_proj": "position"}, inplace=True)
    team_b = board[board["player_id"].isin(
        ["7547", "8112", "6797"]
    )][["player_id", "position_proj", "proj_points"]].copy()
    team_b.rename(columns={"position_proj": "position"}, inplace=True)
    sim = simulate_team_matchup(team_a, team_b, iterations=1000)
    assert 0.0 <= sim["win_prob_a"] <= 1.0
    assert 0.0 <= sim["win_prob_b"] <= 1.0
    assert sim["iterations"] == 1000


# ---------------------------------------------------------------------------
# Waivers pipeline test
# ---------------------------------------------------------------------------
def test_waiver_recommendations_schema():
    """Waiver recommendations return expected columns."""
    board = _board_df()
    proj = board[["player_id", "player_name", "position_proj", "proj_points"]].copy()
    proj.rename(columns={"position_proj": "position"}, inplace=True)
    waiver_pool = proj.head(5).copy()
    roster_ids = ["7564", "8138", "6770"]
    result = build_waiver_recommendations(
        roster_id=14, rosters_df=_rosters_df(),
        draft_board_df=board, remaining_faab=100,
    )
    if not result.empty:
        expected_cols = {
            "player_id", "player_name", "position",
            "marginal_value", "conservative_bid",
            "market_bid", "aggressive_bid",
        }
        assert expected_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Trade pipeline test
# ---------------------------------------------------------------------------
def test_trade_evaluation_returns_dict():
    """Trade evaluation returns valid verdict structure."""
    board = _board_df()
    proj = board[["player_id", "player_name", "position_proj", "proj_points"]].copy()
    proj.rename(columns={"position_proj": "position"}, inplace=True)
    result = evaluate_trade(
        team_a_roster=["7564", "8138", "6770", "4866", "12526"],
        team_b_roster=["7547", "8112", "6797", "5892", "6794"],
        team_a_sends=["8138"],
        team_b_sends=["7547"],
        projections_df=proj,
        draft_board_df=board,
    )
    assert isinstance(result, dict)
    assert "team_a" in result
    assert "team_b" in result
    assert "verdict_a" in result
    assert "verdict_b" in result


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------
def test_digest_structure():
    """generate_weekly_digest returns all expected keys."""
    board = _board_df()
    with patch("src.reports.fetch_league_matchups") as mock_matchups:
        mock_matchups.return_value = _matchups_raw()
        digest = generate_weekly_digest(
            league_id="1386423085304938496",
            week=1, roster_id=14,
            rosters_df=_rosters_df(),
            draft_board_df=board,
            catalog_df=_catalog_df(),
        )
    expected_keys = {
        "league_id", "week", "roster_id", "optimal_lineup",
        "total_proj_points", "matchup", "opponent_roster_id",
        "waivers", "drop_candidates", "injury_warnings",
        "power_rankings", "roster_power_rank", "player_name_lookup",
    }
    assert expected_keys.issubset(set(digest.keys()))
    assert digest["week"] == 1
    assert digest["roster_id"] == 14


def test_markdown_export_creates_file():
    """export_digest_markdown creates a valid markdown file."""
    board = _board_df()
    with patch("src.reports.fetch_league_matchups") as mock_matchups:
        mock_matchups.return_value = _matchups_raw()
        digest = generate_weekly_digest(
            league_id="1386423085304938496",
            week=1, roster_id=14,
            rosters_df=_rosters_df(),
            draft_board_df=board,
            catalog_df=_catalog_df(),
        )
    with tempfile.TemporaryDirectory() as tmp:
        md_path = os.path.join(tmp, "week_1_roster_14.md")
        result = export_digest_markdown(digest, md_path)
        assert os.path.exists(result)
        with open(result) as f:
            content = f.read()
        assert "# Weekly Digest" in content
        assert "## Matchup Projection" in content
        assert "## Optimal Lineup" in content


def test_power_rankings_csv_export():
    """export_power_rankings_csv creates a valid CSV."""
    df = pd.DataFrame({
        "roster_id": [1, 2, 3],
        "true_talent_win_pct": [0.40, 0.35, 0.25],
        "power_rank": [1, 2, 3],
        "median_total": [1200.0, 1100.0, 1000.0],
        "floor_total": [1000.0, 900.0, 800.0],
        "ceiling_total": [1400.0, 1300.0, 1200.0],
    })
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "power_rankings.csv")
        result = export_power_rankings_csv(df, csv_path)
        assert os.path.exists(result)
        loaded = pd.read_csv(result)
        assert len(loaded) == 3
        assert "roster_id" in loaded.columns
        assert "power_rank" in loaded.columns


def test_report_handler_integration():
    """Full report handler creates both files."""
    board = _board_df()
    with tempfile.TemporaryDirectory() as tmp:
        with patch("src.cli._load_draft_board") as mock_board, \
             patch("src.cli._load_sleeper_catalog") as mock_cat, \
             patch("src.cli.fetch_league_rosters") as mock_rosters, \
             patch("src.reports.fetch_league_matchups") as mock_matchups:
            mock_board.return_value = board
            mock_cat.return_value = _catalog_df()
            mock_rosters.return_value = [
                {"roster_id": i, "owner_id": f"u{i}", "players": [f"p{i}_a"]}
                for i in range(1, 15)
            ]
            mock_matchups.return_value = _matchups_raw()
            parser = build_parser()
            args = parser.parse_args([
                "report", "--league-id", "1386423085304938496",
                "--roster-id", "14", "--week", "1",
                "--output-dir", tmp,
            ])
            args.func(args)
        assert os.path.exists(os.path.join(tmp, "week_1_roster_14.md"))
        assert os.path.exists(os.path.join(tmp, "week_1_power_rankings.csv"))
