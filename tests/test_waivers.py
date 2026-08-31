"""Tests for src/waivers.py — waiver wire evaluator and FAAB allocation."""

import pandas as pd
import pytest

from src.waivers import (
    calculate_marginal_roster_value,
    rank_drop_candidates,
    recommend_faab_bids,
    build_waiver_recommendations,
)


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


# ---------------------------------------------------------------------------
# recommend_faab_bids
# ---------------------------------------------------------------------------
def _upgrades_df():
    """Return a sample upgrades DataFrame (output of marginal_roster_value)."""
    return pd.DataFrame(
        {
            "player_id": ["w1", "w5"],
            "player_name": ["Trey McBride", "Rachaad White"],
            "position": ["TE", "RB"],
            "proj_points": [200.0, 170.0],
            "baseline": [150.0, 180.0],
            "marginal_value": [50.0, -10.0],
        }
    )


def _positive_upgrades_df():
    """Return only positive marginal-value targets."""
    return pd.DataFrame(
        {
            "player_id": ["w1", "w6"],
            "player_name": ["Trey McBride", "Rhamondre Stevenson"],
            "position": ["TE", "RB"],
            "proj_points": [200.0, 220.0],
            "baseline": [150.0, 180.0],
            "marginal_value": [50.0, 40.0],
        }
    )


def test_faab_bids_returns_dataframe():
    """recommend_faab_bids returns a DataFrame."""
    result = recommend_faab_bids(_positive_upgrades_df(), 100)
    assert isinstance(result, pd.DataFrame)


def test_faab_bids_columns():
    """Result contains expected columns."""
    result = recommend_faab_bids(_positive_upgrades_df(), 100)
    expected = {
        "player_id",
        "player_name",
        "position",
        "proj_points",
        "marginal_value",
        "conservative_bid",
        "market_bid",
        "aggressive_bid",
    }
    assert expected == set(result.columns)


def test_faab_bids_respect_remaining_budget():
    """No bid tier exceeds remaining_faab."""
    result = recommend_faab_bids(_positive_upgrades_df(), 100)
    for col in ("conservative_bid", "market_bid", "aggressive_bid"):
        assert (result[col] <= 100).all()


def test_faab_bids_enforce_min_bid():
    """All bids are at least min_bid."""
    result = recommend_faab_bids(_positive_upgrades_df(), 100, min_bid=5)
    for col in ("conservative_bid", "market_bid", "aggressive_bid"):
        assert (result[col] >= 5).all()


def test_faab_bids_zero_budget():
    """With $0 remaining, every bid is $0."""
    result = recommend_faab_bids(_positive_upgrades_df(), 0, min_bid=0)
    for col in ("conservative_bid", "market_bid", "aggressive_bid"):
        assert (result[col] == 0).all()


def test_faab_bids_empty_upgrades():
    """Empty upgrades returns empty DataFrame with correct columns."""
    result = recommend_faab_bids(
        pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "position",
                "proj_points",
                "baseline",
                "marginal_value",
            ]
        ),
        100,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_faab_bids_tier_ordering():
    """Conservative <= Market <= Aggressive for every row."""
    result = recommend_faab_bids(_positive_upgrades_df(), 100)
    assert (result["conservative_bid"] <= result["market_bid"]).all()
    assert (result["market_bid"] <= result["aggressive_bid"]).all()


def test_faab_bids_proportional():
    """Higher marginal value yields a higher bid at each tier."""
    result = recommend_faab_bids(_positive_upgrades_df(), 100)
    row_high = result[result["player_id"] == "w1"].iloc[0]
    row_low = result[result["player_id"] == "w6"].iloc[0]
    assert row_high["aggressive_bid"] > row_low["aggressive_bid"]


# ---------------------------------------------------------------------------
# build_waiver_recommendations
# ---------------------------------------------------------------------------
def _integration_rosters_df():
    """Sample rosters_df for integration tests (2 managers, 3 teams)."""
    return pd.DataFrame(
        {
            "roster_id": [1, 1, 1, 1, 2, 2, 2, 3],
            "owner_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2", "u3"],
            "player_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
        }
    )


def _integration_draft_board():
    """Draft board with projections, VORP, and positions."""
    return pd.DataFrame(
        {
            "player_id": [
                "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
                "w1", "w2", "w3",
            ],
            "player_name": [
                "Hurts", "Barkley", "Brown", "Goedert",
                "Chase", "Ekeler", "Lamb", "Kelce",
                "McBride", "Hubbard", "Shaheed",
            ],
            "position_proj": [
                "QB", "RB", "WR", "TE",
                "WR", "RB", "WR", "TE",
                "TE", "RB", "WR",
            ],
            "proj_points": [
                340, 280, 250, 150,
                300, 200, 270, 130,
                200, 195, 170,
            ],
            "vorp": [340, 280, 250, 150, 300, 200, 270, 130, 200, 195, 170],
        }
    )


def test_build_waiver_recs_returns_dataframe():
    """build_waiver_recommendations returns a DataFrame."""
    result = build_waiver_recommendations(
        roster_id=1,
        rosters_df=_integration_rosters_df(),
        draft_board_df=_integration_draft_board(),
        remaining_faab=100,
    )
    assert isinstance(result, pd.DataFrame)


def test_build_waiver_recs_columns():
    """Result has the combined column set."""
    result = build_waiver_recommendations(
        roster_id=1,
        rosters_df=_integration_rosters_df(),
        draft_board_df=_integration_draft_board(),
        remaining_faab=100,
    )
    expected = {
        "player_id",
        "player_name",
        "position",
        "proj_points",
        "marginal_value",
        "conservative_bid",
        "market_bid",
        "aggressive_bid",
    }
    assert expected == set(result.columns)


def test_build_waiver_recs_excludes_rostered():
    """Targets already on any roster are excluded."""
    result = build_waiver_recommendations(
        roster_id=1,
        rosters_df=_integration_rosters_df(),
        draft_board_df=_integration_draft_board(),
        remaining_faab=100,
    )
    all_rostered = {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"}
    assert not set(result["player_id"]).intersection(all_rostered)


def test_build_waiver_recs_includes_free_agents():
    """Free agents from the draft board appear in the result."""
    result = build_waiver_recommendations(
        roster_id=1,
        rosters_df=_integration_rosters_df(),
        draft_board_df=_integration_draft_board(),
        remaining_faab=100,
    )
    assert "w1" in result["player_id"].values


def test_build_waiver_recs_empty_roster_id():
    """Unknown roster_id returns empty DataFrame."""
    result = build_waiver_recommendations(
        roster_id=999,
        rosters_df=_integration_rosters_df(),
        draft_board_df=_integration_draft_board(),
        remaining_faab=100,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_build_waiver_recs_no_positive_targets():
    """When no FA improves over the roster baseline, result is empty."""
    # All waiver players project worse than rostered players
    draft_board = _integration_draft_board()
    draft_board.loc[draft_board["player_id"].str.startswith("w"), "proj_points"] = 10
    result = build_waiver_recommendations(
        roster_id=1,
        rosters_df=_integration_rosters_df(),
        draft_board_df=draft_board,
        remaining_faab=100,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_build_waiver_recs_faab_bounded():
    """All bids respect the remaining_faab cap."""
    result = build_waiver_recommendations(
        roster_id=1,
        rosters_df=_integration_rosters_df(),
        draft_board_df=_integration_draft_board(),
        remaining_faab=50,
    )
    for col in ("conservative_bid", "market_bid", "aggressive_bid"):
        assert (result[col] <= 50).all()
