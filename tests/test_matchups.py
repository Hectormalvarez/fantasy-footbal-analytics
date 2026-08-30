"""Tests for src/matchups.py — weekly sit/start lineup optimizer and matchup engine."""

import pandas as pd
import pytest

from src.matchups import (
    extract_weekly_matchup_roster,
    optimize_starting_lineup,
    compare_sit_start,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _player_pool_df():
    """Return a sample player pool DataFrame."""
    return pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
            "player_name": [
                "Jalen Hurts",
                "Saquon Barkley",
                "AJ Brown",
                "Dallas Goedert",
                "DeVonta Smith",
                "Jaylen Waddle",
                "Taysom Hill",
            ],
            "position": ["QB", "RB", "WR", "TE", "WR", "WR", "TE"],
        }
    )


def _projections_df():
    """Return a sample projections DataFrame."""
    return pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
            "player_name": [
                "Jalen Hurts",
                "Saquon Barkley",
                "AJ Brown",
                "Dallas Goedert",
                "DeVonta Smith",
                "Jaylen Waddle",
                "Taysom Hill",
            ],
            "position": ["QB", "RB", "WR", "TE", "WR", "WR", "TE"],
            "proj_points": [25.0, 18.0, 16.0, 10.0, 14.0, 12.0, 8.0],
        }
    )


def _week_matchups_raw():
    """Return raw Sleeper matchup data for Week 5."""
    return [
        {
            "matchup_id": 1,
            "roster_id": 1,
            "starters": ["p1", "p2", "p3", "p5", "p4"],
            "players": ["p1", "p2", "p3", "p4", "p5", "p7"],
            "points": 110.5,
            "players_points": {
                "p1": 28.0,
                "p2": 18.5,
                "p3": 22.0,
                "p4": 10.5,
                "p5": 14.0,
                "p7": 8.0,
            },
        },
        {
            "matchup_id": 1,
            "roster_id": 2,
            "starters": ["p8", "p9", "p10"],
            "players": ["p8", "p9", "p10"],
            "points": 85.0,
            "players_points": {
                "p8": 30.0,
                "p9": 25.0,
                "p10": 15.0,
            },
        },
    ]


# ---------------------------------------------------------------------------
# extract_weekly_matchup_roster
# ---------------------------------------------------------------------------
def test_extract_matchup_returns_dataframe():
    """extract_weekly_matchup_roster returns a DataFrame."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    assert isinstance(result, pd.DataFrame)


def test_extract_matchup_columns():
    """Result contains expected columns."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    expected = {
        "player_id",
        "player_name",
        "position",
        "is_starter",
        "opponent_roster_id",
        "actual_points",
        "proj_points",
    }
    assert expected == set(result.columns)


def test_extract_matchup_player_count():
    """Result has one row per player on the roster."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    assert len(result) == 6  # roster 1 has 6 players


def test_extract_matchup_starters():
    """Players in the starters list are marked is_starter=True."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    starters = result[result["is_starter"] == True]
    assert set(starters["player_id"]) == {"p1", "p2", "p3", "p4", "p5"}


def test_extract_matchup_bench():
    """Players NOT in the starters list are marked is_starter=False."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    bench = result[result["is_starter"] == False]
    assert "p7" in bench["player_id"].values


def test_extract_matchup_opponent():
    """opponent_roster_id points to the other roster sharing the same matchup_id."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    assert (result["opponent_roster_id"] == 2).all()


def test_extract_matchup_actual_points():
    """actual_points are populated from the players_points dict."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    hurts = result[result["player_id"] == "p1"].iloc[0]
    assert hurts["actual_points"] == 28.0


def test_extract_matchup_not_in_player_pool():
    """Players missing from player_pool_df get name='Unknown' and position='UNKNOWN'."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=pd.DataFrame(columns=["player_id", "player_name", "position"]),
    )
    assert (result["player_name"] == "Unknown").all()
    assert (result["position"] == "UNKNOWN").all()


def test_extract_matchup_roster_not_found():
    """Unknown roster_id returns empty DataFrame."""
    result = extract_weekly_matchup_roster(
        roster_id=999,
        week_matchups_raw=_week_matchups_raw(),
        player_pool_df=_player_pool_df(),
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_extract_matchup_empty_matchups():
    """Empty matchup list returns empty DataFrame."""
    result = extract_weekly_matchup_roster(
        roster_id=1,
        week_matchups_raw=[],
        player_pool_df=_player_pool_df(),
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# optimize_starting_lineup
# ---------------------------------------------------------------------------
def test_optimize_returns_dict():
    """optimize_starting_lineup returns a dict."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
    )
    assert isinstance(result, dict)


def test_optimize_keys():
    """Result dict contains expected keys."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
    )
    assert "lineup" in result
    assert "total_proj_points" in result
    assert "bench" in result


def test_optimize_fills_qb_slot():
    """QB slot is assigned to a QB-eligible player."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
    )
    assert result["lineup"]["QB"] == "p1"


def test_optimize_fills_rb_slots():
    """RB slots are assigned to RB-eligible players."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
    )
    assert result["lineup"]["RB1"] == "p2"


def test_optimize_fills_wr_slots():
    """WR slots are assigned to WR-eligible players."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
    )
    wr_slots = [result["lineup"].get("WR1"), result["lineup"].get("WR2")]
    assert "p3" in wr_slots
    assert "p5" in wr_slots


def test_optimize_fills_flex_with_best_remaining():
    """FLEX slot is filled with highest-projected RB/WR/TE not already starting."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
        roster_slots=["QB", "RB1", "WR1", "TE", "FLEX"],
    )
    # After QB=p1, RB1=p2, WR1=p3, TE=p4 → p5 (WR, 14.0) fills FLEX
    assert result["lineup"]["FLEX"] == "p5"


def test_optimize_total_points():
    """total_proj_points is the sum of all starter projections."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
    )
    expected_total = 25.0 + 18.0 + 16.0 + 14.0 + 10.0
    assert result["total_proj_points"] == expected_total


def test_optimize_bench():
    """Bench list contains player_ids not in the starting lineup."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        projections_df=_projections_df(),
        roster_slots=["QB", "RB1", "WR1", "TE"],
    )
    # Only 4 starting slots but 7 players → 3 on bench
    assert "p7" in result["bench"]
    assert "p6" in result["bench"]


def test_optimize_custom_slots():
    """Custom roster_slots override the default configuration."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5"],
        projections_df=_projections_df(),
        roster_slots=["QB", "FLEX"],
    )
    assert result["lineup"]["QB"] == "p1"
    assert result["lineup"]["FLEX"] == "p2"


def test_optimize_empty_roster():
    """Empty roster returns empty lineup and zero points."""
    result = optimize_starting_lineup(
        rostered_player_ids=[],
        projections_df=_projections_df(),
    )
    assert result["lineup"] == {}
    assert result["total_proj_points"] == 0
    assert result["bench"] == []


def test_optimize_no_projection():
    """Players missing from projections get 0 projected points."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "unknown_player"],
        projections_df=_projections_df(),
    )
    assert result["total_proj_points"] == 25.0 + 18.0


def test_optimize_bench_sorted_descending():
    """Bench players are sorted by projected points descending."""
    result = optimize_starting_lineup(
        rostered_player_ids=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        projections_df=_projections_df(),
        roster_slots=["QB", "RB1", "WR1", "TE"],
    )
    # Starting: p1(25), p2(18), p3(16), p4(10) → bench: p5(14), p6(12), p7(8) desc
    assert result["bench"] == ["p5", "p6", "p7"]


# ---------------------------------------------------------------------------
# compare_sit_start
# ---------------------------------------------------------------------------
def test_compare_returns_dict():
    """compare_sit_start returns a dict."""
    result = compare_sit_start("p1", "p2", _projections_df())
    assert isinstance(result, dict)


def test_compare_keys():
    """Result dict contains expected keys."""
    result = compare_sit_start("p1", "p2", _projections_df())
    assert "player_a" in result
    assert "player_b" in result
    assert "delta" in result
    assert "recommendation" in result


def test_compare_player_fields():
    """Each player sub-dict has id, name, position, proj_points, floor, ceiling."""
    result = compare_sit_start("p1", "p2", _projections_df())
    for key in ("player_a", "player_b"):
        p = result[key]
        assert "id" in p
        assert "name" in p
        assert "position" in p
        assert "proj_points" in p
        assert "floor" in p
        assert "ceiling" in p


def test_compare_delta():
    """delta is player_a proj minus player_b proj."""
    result = compare_sit_start("p1", "p2", _projections_df())
    assert result["delta"] == pytest.approx(25.0 - 18.0)


def test_compare_recommendation_start_a():
    """When player_a is projected significantly higher, recommendation favors A."""
    result = compare_sit_start("p1", "p2", _projections_df())
    assert result["recommendation"] == "Start Jalen Hurts"


def test_compare_recommendation_start_b():
    """When player_b is projected significantly higher, recommendation favors B."""
    result = compare_sit_start("p2", "p1", _projections_df())
    assert result["recommendation"] == "Start Jalen Hurts"


def test_compare_recommendation_tossup():
    """When delta is within tolerance, recommendation is 'Toss-up'."""
    result = compare_sit_start("p5", "p6", _projections_df())
    assert result["recommendation"] == "Toss-up"


def test_compare_floor_ceiling():
    """Floor < proj_points < ceiling for each player."""
    result = compare_sit_start("p1", "p2", _projections_df())
    for key in ("player_a", "player_b"):
        p = result[key]
        assert p["floor"] < p["proj_points"]
        assert p["proj_points"] < p["ceiling"]


def test_compare_player_not_found():
    """Missing player_id returns player fields with 0 projection."""
    result = compare_sit_start("p1", "unknown", _projections_df())
    assert result["player_b"]["proj_points"] == 0
    assert result["delta"] == 25.0


def test_compare_both_missing():
    """Both missing returns zero delta and toss-up."""
    result = compare_sit_start("unknown1", "unknown2", _projections_df())
    assert result["delta"] == 0
    assert result["recommendation"] == "Toss-up"
