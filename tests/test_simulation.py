"""Tests for src/simulation.py -- Monte Carlo simulation engine."""

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    simulate_player_weekly_distribution,
    simulate_team_matchup,
    calculate_league_power_rankings,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _starters_df_a():
    """Team A starters."""
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p3"],
        "position": ["QB", "RB", "WR"],
        "proj_points": [25.0, 15.0, 12.0],
    })


def _starters_df_b():
    """Team B starters."""
    return pd.DataFrame({
        "player_id": ["p4", "p5", "p6"],
        "position": ["QB", "RB", "WR"],
        "proj_points": [20.0, 18.0, 10.0],
    })


def _rosters_df():
    """League rosters (3 teams)."""
    return pd.DataFrame({
        "roster_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "player_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
    })


def _projections_df():
    """Player projections."""
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
        "position": ["QB", "RB", "WR", "QB", "RB", "WR", "QB", "RB", "WR"],
        "proj_points": [25.0, 15.0, 12.0, 20.0, 18.0, 10.0, 18.0, 14.0, 11.0],
    })


# ---------------------------------------------------------------------------
# simulate_player_weekly_distribution
# ---------------------------------------------------------------------------
def test_distribution_shape():
    """Output has correct number of iterations."""
    result = simulate_player_weekly_distribution(20.0, iterations=5000)
    assert result.shape == (5000,)


def test_distribution_returns_numpy():
    """Output is a numpy array."""
    result = simulate_player_weekly_distribution(20.0)
    assert isinstance(result, np.ndarray)


def test_distribution_mean_near_projection():
    """Mean of distribution is near the projected points."""
    result = simulate_player_weekly_distribution(20.0, iterations=50000, random_seed=42)
    assert np.mean(result) == pytest.approx(20.0, abs=0.5)


def test_distribution_respects_floor():
    """No values below floor_ratio * projected_points."""
    result = simulate_player_weekly_distribution(20.0, floor_ratio=0.70, iterations=10000)
    assert np.min(result) >= 20.0 * 0.70 - 0.01


def test_distribution_respects_ceiling():
    """No values above ceiling_ratio * projected_points."""
    result = simulate_player_weekly_distribution(20.0, ceiling_ratio=1.30, iterations=10000)
    assert np.max(result) <= 20.0 * 1.30 + 0.01


def test_distribution_positive_values():
    """All simulated points are non-negative."""
    result = simulate_player_weekly_distribution(20.0, iterations=10000)
    assert np.all(result >= 0)


def test_distribution_zero_projection():
    """Zero projection returns all zeros."""
    result = simulate_player_weekly_distribution(0.0, iterations=1000)
    assert np.all(result == 0.0)


def test_distribution_negative_projection():
    """Negative projection returns all zeros."""
    result = simulate_player_weekly_distribution(-5.0, iterations=1000)
    assert np.all(result == 0.0)


def test_distribution_seed_determinism():
    """Same seed produces identical output."""
    r1 = simulate_player_weekly_distribution(20.0, iterations=1000, random_seed=42)
    r2 = simulate_player_weekly_distribution(20.0, iterations=1000, random_seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_distribution_different_seeds():
    """Different seeds produce different output."""
    r1 = simulate_player_weekly_distribution(20.0, iterations=1000, random_seed=42)
    r2 = simulate_player_weekly_distribution(20.0, iterations=1000, random_seed=99)
    assert not np.array_equal(r1, r2)


def test_distribution_custom_ratios():
    """Custom floor/ceiling ratios are respected."""
    result = simulate_player_weekly_distribution(
        20.0, floor_ratio=0.50, ceiling_ratio=1.50, iterations=10000
    )
    assert np.min(result) >= 20.0 * 0.50 - 0.01
    assert np.max(result) <= 20.0 * 1.50 + 0.01


# ---------------------------------------------------------------------------
# simulate_team_matchup
# ---------------------------------------------------------------------------
def test_matchup_returns_dict():
    """simulate_team_matchup returns a dict."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b())
    assert isinstance(result, dict)


def test_matchup_keys():
    """Result contains all expected keys."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b())
    expected_keys = {
        "win_prob_a", "win_prob_b", "median_a", "median_b",
        "floor_a", "floor_b", "ceiling_a", "ceiling_b",
        "avg_margin", "iterations",
    }
    assert expected_keys.issubset(result.keys())


def test_matchup_win_prob_bounds():
    """Win probabilities are between 0 and 1."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b())
    assert 0.0 <= result["win_prob_a"] <= 1.0
    assert 0.0 <= result["win_prob_b"] <= 1.0


def test_matchup_win_probs_sum():
    """Win probabilities sum to approximately 1.0."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b())
    assert result["win_prob_a"] + result["win_prob_b"] == pytest.approx(1.0, abs=0.01)


def test_matchup_symmetry_identical_rosters():
    """Identical rosters produce ~50/50 win probability."""
    df_a = _starters_df_a()
    result = simulate_team_matchup(df_a, df_a, iterations=10000, random_seed=42)
    assert result["win_prob_a"] == pytest.approx(0.5, abs=0.05)
    assert result["win_prob_b"] == pytest.approx(0.5, abs=0.05)


def test_matchup_better_team_wins_more():
    """Team with higher projections wins more often."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b(), iterations=10000)
    # Team A (25+15+12=52) > Team B (20+18+10=48)
    assert result["win_prob_a"] > 0.5


def test_matchup_floor_below_median():
    """10th percentile is below median."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b())
    assert result["floor_a"] <= result["median_a"]
    assert result["floor_b"] <= result["median_b"]


def test_matchup_ceiling_above_median():
    """90th percentile is above median."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b())
    assert result["ceiling_a"] >= result["median_a"]
    assert result["ceiling_b"] >= result["median_b"]


def test_matchup_seed_determinism():
    """Same seed produces identical output."""
    r1 = simulate_team_matchup(_starters_df_a(), _starters_df_b(), iterations=1000, random_seed=42)
    r2 = simulate_team_matchup(_starters_df_a(), _starters_df_b(), iterations=1000, random_seed=42)
    assert r1 == r2


def test_matchup_empty_team_a():
    """Empty Team A returns default 50/50."""
    empty = pd.DataFrame(columns=["player_id", "position", "proj_points"])
    result = simulate_team_matchup(empty, _starters_df_b())
    assert result["win_prob_a"] == 0.5
    assert result["win_prob_b"] == 0.5


def test_matchup_empty_team_b():
    """Empty Team B returns default 50/50."""
    empty = pd.DataFrame(columns=["player_id", "position", "proj_points"])
    result = simulate_team_matchup(_starters_df_a(), empty)
    assert result["win_prob_a"] == 0.5
    assert result["win_prob_b"] == 0.5


def test_matchup_both_empty():
    """Both empty returns default 50/50."""
    empty = pd.DataFrame(columns=["player_id", "position", "proj_points"])
    result = simulate_team_matchup(empty, empty)
    assert result["win_prob_a"] == 0.5


def test_matchup_iterations_recorded():
    """iterations key matches input."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b(), iterations=7777)
    assert result["iterations"] == 7777


def test_matchup_positive_projections():
    """All median/floor/ceiling values are non-negative."""
    result = simulate_team_matchup(_starters_df_a(), _starters_df_b())
    for key in ["median_a", "median_b", "floor_a", "floor_b", "ceiling_a", "ceiling_b"]:
        assert result[key] >= 0


# ---------------------------------------------------------------------------
# calculate_league_power_rankings
# ---------------------------------------------------------------------------
def test_power_rankings_returns_dataframe():
    """calculate_league_power_rankings returns a DataFrame."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    assert isinstance(result, pd.DataFrame)


def test_power_rankings_columns():
    """Result has all expected columns."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    expected = {"roster_id", "true_talent_win_pct", "power_rank",
                "median_total", "floor_total", "ceiling_total"}
    assert expected.issubset(result.columns)


def test_power_rankings_one_row_per_roster():
    """One row per unique roster."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    assert len(result) == 3


def test_power_rankings_win_pct_bounds():
    """Win percentages are between 0 and 1."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    assert (result["true_talent_win_pct"] >= 0).all()
    assert (result["true_talent_win_pct"] <= 1).all()


def test_power_rankings_ranks_sequential():
    """Power ranks are 1 to N."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    ranks = sorted(result["power_rank"].tolist())
    assert ranks == [1, 2, 3]


def test_power_rankings_best_team_rank1():
    """Team with highest total projection gets rank 1."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    rank1 = result[result["power_rank"] == 1].iloc[0]
    # Team 1: 25+15+12=52 (highest)
    assert rank1["roster_id"] == 1


def test_power_rankings_seed_determinism():
    """Same seed produces identical output."""
    r1 = calculate_league_power_rankings(_rosters_df(), _projections_df(), iterations=500, random_seed=42)
    r2 = calculate_league_power_rankings(_rosters_df(), _projections_df(), iterations=500, random_seed=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_power_rankings_empty_rosters():
    """Empty rosters returns empty DataFrame."""
    empty = pd.DataFrame(columns=["roster_id", "player_id"])
    result = calculate_league_power_rankings(empty, _projections_df())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_power_rankings_empty_projections():
    """Empty projections returns empty DataFrame."""
    empty = pd.DataFrame(columns=["player_id", "position", "proj_points"])
    result = calculate_league_power_rankings(_rosters_df(), empty)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_power_rankings_single_roster():
    """Single roster returns empty DataFrame (no matchups)."""
    single = pd.DataFrame({"roster_id": [1, 1], "player_id": ["p1", "p2"]})
    result = calculate_league_power_rankings(single, _projections_df())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_power_rankings_floor_below_ceiling():
    """Floor is below ceiling for each roster."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    for _, row in result.iterrows():
        assert row["floor_total"] <= row["ceiling_total"]


def test_power_rankings_win_pcts_approx_sum():
    """Win percentages roughly sum to 1.0 (within rounding)."""
    result = calculate_league_power_rankings(_rosters_df(), _projections_df())
    total = result["true_talent_win_pct"].sum()
    assert total == pytest.approx(1.0, abs=0.05)
