"""Tests for src/waivers.py — waiver wire evaluator and FAAB allocation."""

import pandas as pd
import pytest

from src.waivers import calculate_marginal_roster_value, rank_drop_candidates


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _proj_df():
    """Return a sample projections DataFrame with player_id."""
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
            "proj_points": [340.0, 280.0, 250.0, 150.0, 210.0, 180.0, 120.0],
        }
    )


def _waiver_df():
    """Return a sample waiver pool DataFrame."""
    return pd.DataFrame(
        {
            "player_id": ["w1", "w2", "w3", "w4"],
            "player_name": [
                "Trey McBride",
                "Chuba Hubbard",
                "Rashid Shaheed",
                "Justin Fields",
            ],
            "position": ["TE", "RB", "WR", "QB"],
            "proj_points": [200.0, 195.0, 170.0, 290.0],
        }
    )


# ---------------------------------------------------------------------------
# calculate_marginal_roster_value
# ---------------------------------------------------------------------------
def test_marginal_value_returns_dataframe():
    """calculate_marginal_roster_value returns a DataFrame."""
    proj = _proj_df()
    waiver = _waiver_df()
    roster_ids = ["p1", "p2", "p3", "p4", "p5"]
    result = calculate_marginal_roster_value(roster_ids, waiver, proj)
    assert isinstance(result, pd.DataFrame)


def test_marginal_value_columns():
    """Result contains expected columns."""
    proj = _proj_df()
    waiver = _waiver_df()
    roster_ids = ["p1", "p2", "p3", "p4", "p5"]
    result = calculate_marginal_roster_value(roster_ids, waiver, proj)
    expected = {
        "player_id",
        "player_name",
        "position",
        "proj_points",
        "baseline",
        "marginal_value",
    }
    assert expected == set(result.columns)


def test_marginal_value_correct_values():
    """Baseline is the minimum at each position; marginal = target - baseline."""
    proj = _proj_df()
    waiver = _waiver_df()
    # Roster: QB Hurts(340), RB Barkley(280), WR Brown(250) + Smith(210),
    #         TE Goedert(150)
    # Baselines: QB=340, RB=280, WR=210 (min of 250,210), TE=150
    roster_ids = ["p1", "p2", "p3", "p4", "p5"]
    result = calculate_marginal_roster_value(roster_ids, waiver, proj)

    # Trey McBride: TE baseline=150, proj=200 → marginal=50
    mcbride = result[result["player_id"] == "w1"].iloc[0]
    assert mcbride["baseline"] == 150.0
    assert mcbride["marginal_value"] == 50.0

    # Chuba Hubbard: RB baseline=280, proj=195 → marginal=-85 (excluded)
    assert "w2" not in result["player_id"].values

    # Rashid Shaheed: WR baseline=210, proj=170 → marginal=-40 (excluded)
    assert "w3" not in result["player_id"].values

    # Justin Fields: QB baseline=340, proj=290 → marginal=-50 (excluded)
    assert "w4" not in result["player_id"].values


def test_marginal_value_filters_negative():
    """Targets worse than the baseline are excluded."""
    proj = _proj_df()
    waiver = _waiver_df()
    roster_ids = ["p1", "p2", "p3", "p4", "p5"]
    result = calculate_marginal_roster_value(roster_ids, waiver, proj)
    assert (result["marginal_value"] > 0).all()


def test_marginal_value_empty_roster():
    """Empty roster returns empty DataFrame."""
    waiver = _waiver_df()
    result = calculate_marginal_roster_value([], waiver, _proj_df())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_marginal_value_empty_waiver():
    """Empty waiver pool returns empty DataFrame."""
    proj = _proj_df()
    result = calculate_marginal_roster_value(["p1"], pd.DataFrame(columns=proj.columns), proj)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_marginal_value_new_position_baseline_zero():
    """Position not on roster gets baseline 0, so full proj_points counts."""
    proj = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "player_name": ["Hurts", "Barkley"],
            "position": ["QB", "RB"],
            "proj_points": [340.0, 280.0],
        }
    )
    waiver = pd.DataFrame(
        {
            "player_id": ["w1"],
            "player_name": ["McBride"],
            "position": ["TE"],
            "proj_points": [200.0],
        }
    )
    result = calculate_marginal_roster_value(["p1", "p2"], waiver, proj)
    row = result.iloc[0]
    assert row["baseline"] == 0.0
    assert row["marginal_value"] == 200.0


# ---------------------------------------------------------------------------
# rank_drop_candidates
# ---------------------------------------------------------------------------
def test_drop_candidates_returns_dataframe():
    """rank_drop_candidates returns a DataFrame."""
    proj = _proj_df()
    roster_ids = ["p1", "p2", "p3", "p4", "p5"]
    result = rank_drop_candidates(roster_ids, proj)
    assert isinstance(result, pd.DataFrame)


def test_drop_candidates_sorted_ascending():
    """Drop candidates are sorted by proj_points ascending (worst first)."""
    proj = _proj_df()
    roster_ids = ["p1", "p2", "p3", "p4", "p5"]
    result = rank_drop_candidates(roster_ids, proj)
    assert list(result["proj_points"]) == sorted(result["proj_points"])


def test_drop_candidates_columns():
    """Result contains expected columns."""
    proj = _proj_df()
    result = rank_drop_candidates(["p1"], proj)
    expected = {"player_id", "player_name", "position", "proj_points", "drop_rank"}
    assert expected == set(result.columns)


def test_drop_candidates_drop_rank():
    """drop_rank is a sequential integer starting at 1."""
    proj = _proj_df()
    roster_ids = ["p1", "p2", "p3", "p4", "p5"]
    result = rank_drop_candidates(roster_ids, proj)
    assert list(result["drop_rank"]) == list(range(1, len(result) + 1))


def test_drop_candidates_empty_roster():
    """Empty roster returns empty DataFrame."""
    result = rank_drop_candidates([], _proj_df())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_drop_candidates_empty_projections():
    """Empty projections returns empty DataFrame."""
    empty_proj = pd.DataFrame(columns=["player_id", "player_name", "position", "proj_points"])
    result = rank_drop_candidates(["p1"], empty_proj)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
