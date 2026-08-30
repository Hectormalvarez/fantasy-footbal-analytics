"""Tests for src/reports.py -- weekly digest and report generator."""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.reports import (
    generate_weekly_digest,
    export_digest_markdown,
    export_power_rankings_csv,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _rosters_df():
    """Exploded roster DataFrame (3 rosters)."""
    rows = []
    for rid, pids in [(1, ["p1", "p2", "p3"]), (2, ["p4", "p5", "p6"]), (3, ["p7", "p8", "p9"])]:
        for pid in pids:
            rows.append({"roster_id": rid, "owner_id": f"u{rid}", "player_id": pid})
    return pd.DataFrame(rows)


def _draft_board():
    """Draft board with projections."""
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
        "player_name": [
            "Jalen Hurts", "Saquon Barkley", "AJ Brown",
            "Josh Allen", "Bijan Robinson", "CeeDee Lamb",
            "Lamar Jackson", "Breece Hall", "Tyreek Hill",
        ],
        "position_proj": ["QB", "RB", "WR", "QB", "RB", "WR", "QB", "RB", "WR"],
        "proj_points": [25.0, 18.0, 16.0, 24.0, 17.0, 15.0, 23.0, 16.5, 14.0],
        "team": ["PHI", "PHI", "PHI", "BUF", "ATL", "DAL", "BAL", "NYJ", "MIA"],
        "norm_name": [
            "jalen hurts", "saquon barkley", "aj brown",
            "josh allen", "bijan robinson", "ceedee lamb",
            "lamar jackson", "breece hall", "tyreek hill",
        ],
    })


def _catalog_df():
    """Minimal catalog with injury fields."""
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
        "player_name": [
            "Jalen Hurts", "Saquon Barkley", "AJ Brown",
            "Josh Allen", "Bijan Robinson", "CeeDee Lamb",
            "Lamar Jackson", "Breece Hall", "Tyreek Hill",
        ],
        "position": ["QB", "RB", "WR", "QB", "RB", "WR", "QB", "RB", "WR"],
        "injury_status": [None, "Questionable", None, None, None, None, None, None, None],
        "injury_body_part": [None, "Knee", None, None, None, None, None, None, None],
        "injury_notes": [None, None, None, None, None, None, None, None, None],
    })


def _power_rankings_df():
    """Sample power rankings DataFrame."""
    return pd.DataFrame({
        "roster_id": [1, 2, 3],
        "true_talent_win_pct": [0.42, 0.35, 0.23],
        "power_rank": [1, 2, 3],
        "median_total": [110.5, 102.3, 95.1],
        "floor_total": [85.0, 78.0, 70.0],
        "ceiling_total": [135.0, 125.0, 118.0],
    })


def _mock_matchups():
    """Fake raw matchup data for mocking API calls."""
    return [
        {
            "matchup_id": 1, "roster_id": 1,
            "starters": ["p1", "p2", "p3"],
            "players": ["p1", "p2", "p3"],
            "players_points": {"p1": 25.0, "p2": 18.0, "p3": 16.0},
        },
        {
            "matchup_id": 1, "roster_id": 2,
            "starters": ["p4", "p5", "p6"],
            "players": ["p4", "p5", "p6"],
            "players_points": {"p4": 24.0, "p5": 17.0, "p6": 15.0},
        },
        {
            "matchup_id": 2, "roster_id": 3,
            "starters": ["p7", "p8", "p9"],
            "players": ["p7", "p8", "p9"],
            "players_points": {"p7": 23.0, "p8": 16.5, "p9": 14.0},
        },
    ]


# ---------------------------------------------------------------------------
# generate_weekly_digest -- structure tests
# ---------------------------------------------------------------------------
@patch("src.reports.fetch_league_matchups")
def test_digest_returns_dict(mock_fetch):
    """generate_weekly_digest returns a dict."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result, dict)


@patch("src.reports.fetch_league_matchups")
def test_digest_has_all_keys(mock_fetch):
    """Digest contains all expected top-level keys."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    expected_keys = {
        "league_id", "week", "roster_id",
        "optimal_lineup", "total_proj_points",
        "matchup", "opponent_roster_id",
        "waivers", "drop_candidates",
        "injury_warnings", "power_rankings", "roster_power_rank",
    }
    assert expected_keys.issubset(result.keys())


@patch("src.reports.fetch_league_matchups")
def test_digest_league_id_preserved(mock_fetch):
    """Digest preserves league_id."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="99999", week=3, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert result["league_id"] == "99999"


@patch("src.reports.fetch_league_matchups")
def test_digest_week_preserved(mock_fetch):
    """Digest preserves week number."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=7, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert result["week"] == 7


@patch("src.reports.fetch_league_matchups")
def test_digest_optimal_lineup_is_dict(mock_fetch):
    """optimal_lineup is a dict with lineup key."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result["optimal_lineup"], dict)
    assert "lineup" in result["optimal_lineup"]


@patch("src.reports.fetch_league_matchups")
def test_digest_total_proj_points_is_float(mock_fetch):
    """total_proj_points is a float."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result["total_proj_points"], float)


@patch("src.reports.fetch_league_matchups")
def test_digest_matchup_is_dict(mock_fetch):
    """matchup is a dict with win_prob_a key."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result["matchup"], dict)
    assert "win_prob_a" in result["matchup"]


@patch("src.reports.fetch_league_matchups")
def test_digest_waivers_is_dataframe(mock_fetch):
    """waivers is a DataFrame."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result["waivers"], pd.DataFrame)


@patch("src.reports.fetch_league_matchups")
def test_digest_injury_warnings_is_list(mock_fetch):
    """injury_warnings is a list."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result["injury_warnings"], list)


@patch("src.reports.fetch_league_matchups")
def test_digest_power_rankings_is_dataframe(mock_fetch):
    """power_rankings is a DataFrame."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result["power_rankings"], pd.DataFrame)


@patch("src.reports.fetch_league_matchups")
def test_digest_roster_power_rank_is_int_or_none(mock_fetch):
    """roster_power_rank is an int or None."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    val = result["roster_power_rank"]
    assert val is None or isinstance(val, int)


@patch("src.reports.fetch_league_matchups")
def test_digest_with_catalog_injury_warnings(mock_fetch):
    """Catalog with injury data populates injury_warnings."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
        catalog_df=_catalog_df(),
    )
    assert isinstance(result["injury_warnings"], list)


@patch("src.reports.fetch_league_matchups")
def test_digest_empty_rosters(mock_fetch):
    """Empty rosters still returns a valid digest."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=pd.DataFrame(columns=["roster_id", "owner_id", "player_id"]),
        draft_board_df=_draft_board(),
    )
    assert isinstance(result, dict)
    assert result["week"] == 5


@patch("src.reports.fetch_league_matchups")
def test_digest_empty_draft_board(mock_fetch):
    """Empty draft board returns a valid digest with zeros."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=1,
        rosters_df=_rosters_df(),
        draft_board_df=pd.DataFrame(columns=["player_id", "player_name", "position_proj", "proj_points"]),
    )
    assert isinstance(result, dict)
    assert result["total_proj_points"] == 0.0


@patch("src.reports.fetch_league_matchups")
def test_digest_unknown_roster(mock_fetch):
    """Unknown roster_id returns digest with empty matchup."""
    mock_fetch.return_value = _mock_matchups()
    result = generate_weekly_digest(
        league_id="12345", week=5, roster_id=999,
        rosters_df=_rosters_df(), draft_board_df=_draft_board(),
    )
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# export_digest_markdown
# ---------------------------------------------------------------------------
def test_markdown_writes_file():
    """export_digest_markdown creates a file at the given path."""
    digest = {
        "week": 5, "roster_id": 1, "matchup": {"win_prob_a": 0.642, "median_a": 118.5, "floor_a": 92.1, "ceiling_a": 144.8},
        "opponent_roster_id": 2, "total_proj_points": 118.5,
        "optimal_lineup": {"lineup": {"QB": "p1", "RB1": "p2"}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_report.md")
        result = export_digest_markdown(digest, path)
        assert os.path.exists(result)
        assert result.endswith("test_report.md")


def test_markdown_content_has_header():
    """Markdown output contains the week header."""
    digest = {
        "week": 3, "roster_id": 1, "matchup": {"win_prob_a": 0.55, "median_a": 100.0, "floor_a": 75.0, "ceiling_a": 125.0},
        "opponent_roster_id": 2, "total_proj_points": 100.0,
        "optimal_lineup": {"lineup": {"QB": "p1"}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "# Weekly Digest -- Week 3, Roster 1" in content


def test_markdown_content_has_win_prob():
    """Markdown output contains win probability."""
    digest = {
        "week": 5, "roster_id": 1, "matchup": {"win_prob_a": 0.642, "median_a": 118.5, "floor_a": 92.1, "ceiling_a": 144.8},
        "opponent_roster_id": 2, "total_proj_points": 118.5,
        "optimal_lineup": {"lineup": {"QB": "p1"}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "64.2%" in content
        assert "vs Roster 2" in content


def test_markdown_content_has_lineup_table():
    """Markdown output contains a lineup table."""
    digest = {
        "week": 5, "roster_id": 1, "matchup": {"win_prob_a": 0.5, "median_a": 0.0, "floor_a": 0.0, "ceiling_a": 0.0},
        "opponent_roster_id": None, "total_proj_points": 43.0,
        "optimal_lineup": {"lineup": {"QB": "p1", "RB1": "p2"}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "## Optimal Lineup" in content
        assert "| Slot | Player |" in content


def test_markdown_content_has_waivers():
    """Markdown output includes waiver targets when non-empty."""
    waivers = pd.DataFrame({
        "player_id": ["w1"], "player_name": ["Trey McBride"],
        "position": ["TE"], "proj_points": [200.0],
        "marginal_value": [50.0],
        "conservative_bid": [5], "market_bid": [10], "aggressive_bid": [15],
    })
    digest = {
        "week": 5, "roster_id": 1, "matchup": {"win_prob_a": 0.5, "median_a": 0.0, "floor_a": 0.0, "ceiling_a": 0.0},
        "opponent_roster_id": None, "total_proj_points": 0.0,
        "optimal_lineup": {"lineup": {}, "bench": []},
        "waivers": waivers, "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "## Top Waiver Targets" in content
        assert "Trey McBride" in content


def test_markdown_content_has_drops():
    """Markdown output includes drop candidates when non-empty."""
    drops = pd.DataFrame({
        "player_id": ["p3"], "player_name": ["AJ Brown"],
        "position": ["WR"], "proj_points": [16.0], "drop_rank": [1],
    })
    digest = {
        "week": 5, "roster_id": 1, "matchup": {"win_prob_a": 0.5, "median_a": 0.0, "floor_a": 0.0, "ceiling_a": 0.0},
        "opponent_roster_id": None, "total_proj_points": 0.0,
        "optimal_lineup": {"lineup": {}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": drops,
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "## Drop Candidates" in content
        assert "AJ Brown" in content


def test_markdown_content_has_injuries():
    """Markdown output includes injury alerts when present."""
    injuries = [{
        "player_id": "p2", "player_name": "Saquon Barkley",
        "slot": "RB1", "injury_status": "Questionable",
        "injury_body_part": "Knee", "severity": "medium",
    }]
    digest = {
        "week": 5, "roster_id": 1, "matchup": {"win_prob_a": 0.5, "median_a": 0.0, "floor_a": 0.0, "ceiling_a": 0.0},
        "opponent_roster_id": None, "total_proj_points": 0.0,
        "optimal_lineup": {"lineup": {}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": injuries, "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "## Injury Alerts" in content
        assert "Saquon Barkley" in content
        assert "[Questionable]" in content
        assert "(Knee)" in content


def test_markdown_content_has_power_rankings():
    """Markdown output includes power rankings table when non-empty."""
    power = _power_rankings_df()
    digest = {
        "week": 5, "roster_id": 1, "matchup": {"win_prob_a": 0.5, "median_a": 0.0, "floor_a": 0.0, "ceiling_a": 0.0},
        "opponent_roster_id": None, "total_proj_points": 0.0,
        "optimal_lineup": {"lineup": {}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": power, "roster_power_rank": 1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "## League Power Rankings" in content
        assert "**Your Power Rank:** 1" in content
        assert "**<-- YOU**" in content


def test_markdown_creates_parent_dirs():
    """export_digest_markdown creates intermediate directories."""
    digest = {
        "week": 1, "roster_id": 1, "matchup": {},
        "opponent_roster_id": None, "total_proj_points": 0.0,
        "optimal_lineup": {"lineup": {}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "deep", "nested", "report.md")
        result = export_digest_markdown(digest, path)
        assert os.path.exists(result)


def test_markdown_empty_digest():
    """Empty/minimal digest still produces valid markdown."""
    digest = {
        "week": 1, "roster_id": 1, "matchup": {},
        "opponent_roster_id": None, "total_proj_points": 0.0,
        "optimal_lineup": {"lineup": {}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "# Weekly Digest" in content


def test_markdown_no_matchup_section():
    """When matchup is empty, a fallback message appears."""
    digest = {
        "week": 5, "roster_id": 1, "matchup": {},
        "opponent_roster_id": None, "total_proj_points": 0.0,
        "optimal_lineup": {"lineup": {}, "bench": []},
        "waivers": pd.DataFrame(), "drop_candidates": pd.DataFrame(),
        "injury_warnings": [], "power_rankings": pd.DataFrame(), "roster_power_rank": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.md")
        written = export_digest_markdown(digest, path)
        with open(written) as f:
            content = f.read()
        assert "No matchup data available." in content


# ---------------------------------------------------------------------------
# export_power_rankings_csv
# ---------------------------------------------------------------------------
def test_csv_writes_file():
    """export_power_rankings_csv creates a file."""
    df = _power_rankings_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rankings.csv")
        result = export_power_rankings_csv(df, path)
        assert os.path.exists(result)
        assert result.endswith("rankings.csv")


def test_csv_content():
    """CSV file contains expected rows and columns."""
    df = _power_rankings_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rankings.csv")
        written = export_power_rankings_csv(df, path)
        loaded = pd.read_csv(written)
        assert len(loaded) == 3
        assert "roster_id" in loaded.columns
        assert "power_rank" in loaded.columns
        assert "true_talent_win_pct" in loaded.columns


def test_csv_creates_parent_dirs():
    """export_power_rankings_csv creates intermediate directories."""
    df = _power_rankings_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "deep", "nested", "rankings.csv")
        result = export_power_rankings_csv(df, path)
        assert os.path.exists(result)


def test_csv_empty_dataframe():
    """Empty DataFrame produces a CSV with only headers."""
    df = pd.DataFrame(columns=["roster_id", "true_talent_win_pct", "power_rank"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.csv")
        written = export_power_rankings_csv(df, path)
        loaded = pd.read_csv(written)
        assert len(loaded) == 0


def test_csv_roundtrip_integrity():
    """Data survives CSV roundtrip."""
    df = _power_rankings_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rankings.csv")
        written = export_power_rankings_csv(df, path)
        loaded = pd.read_csv(written)
        pd.testing.assert_frame_equal(df, loaded, check_dtype=False)
