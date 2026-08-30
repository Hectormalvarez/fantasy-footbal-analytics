"""Tests for src/dvp.py -- DvP matchup multipliers and projection adjuster."""

import pandas as pd
import pytest

from src.dvp import (
    calculate_defensive_rankings,
    compute_matchup_multiplier,
    adjust_projections_for_matchup,
)
from src.matchups import enhance_projections_with_dvp


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _weekly_stats_df():
    """Game-level stats: two teams, two positions, three weeks."""
    rows = []
    # DAL allows lots of WR points (bad defense) every week
    for w in range(1, 4):
        rows += [
            {"opponent_team": "DAL", "position": "WR", "total_ppr": 30.0, "week": w},
            {"opponent_team": "DAL", "position": "RB", "total_ppr": 20.0, "week": w},
        ]
    # NYG allows few WR points (good defense) every week
    for w in range(1, 4):
        rows += [
            {"opponent_team": "NYG", "position": "WR", "total_ppr": 15.0, "week": w},
            {"opponent_team": "NYG", "position": "RB", "total_ppr": 10.0, "week": w},
        ]
    # CAR is in the middle
    for w in range(1, 4):
        rows += [
            {"opponent_team": "CAR", "position": "WR", "total_ppr": 22.0, "week": w},
            {"opponent_team": "CAR", "position": "RB", "total_ppr": 15.0, "week": w},
        ]
    return pd.DataFrame(rows)


def _projections_df():
    """Sample player projections (per-game baseline)."""
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p3", "p4"],
        "player_name": ["QB A", "WR B", "RB C", "WR D"],
        "position": ["QB", "WR", "RB", "WR"],
        "team": ["PHI", "PHI", "PHI", "NYG"],
        "proj_points": [20.0, 15.0, 12.0, 14.0],
    })


def _schedule_df():
    """Weekly schedule."""
    return pd.DataFrame({
        "team": ["PHI", "NYG", "DAL", "CAR"],
        "opponent_team": ["DAL", "PHI", "NYG", "PHI"],
    })


def _dvp_ranks_df():
    """Pre-built DvP rankings for three teams."""
    return pd.DataFrame({
        "team": ["NYG", "CAR", "DAL", "NYG", "DAL", "CAR"],
        "position": ["WR", "WR", "WR", "RB", "RB", "RB"],
        "fp_allowed_per_game": [15.0, 22.0, 30.0, 10.0, 20.0, 15.0],
        "defense_rank": [1, 16, 30, 1, 2, 3],
    })


# ---------------------------------------------------------------------------
# calculate_defensive_rankings
# ---------------------------------------------------------------------------
def test_rankings_returns_dataframe():
    """calculate_defensive_rankings returns a DataFrame."""
    result = calculate_defensive_rankings(_weekly_stats_df())
    assert isinstance(result, pd.DataFrame)


def test_rankings_columns():
    """Result contains expected columns."""
    result = calculate_defensive_rankings(_weekly_stats_df())
    assert set(result.columns) == {
        "team", "position", "fp_allowed_per_game", "defense_rank",
    }


def test_rankings_covers_all_positions():
    """Each position present in the stats appears in the output."""
    result = calculate_defensive_rankings(_weekly_stats_df())
    positions = set(result["position"])
    assert "WR" in positions
    assert "RB" in positions


def test_rankings_toughest_has_lowest_rank():
    """Team allowing fewest FP gets rank 1."""
    result = calculate_defensive_rankings(_weekly_stats_df())
    wr = result[result["position"] == "WR"].sort_values("defense_rank")
    assert wr.iloc[0]["team"] == "NYG"
    assert wr.iloc[0]["defense_rank"] == 1


def test_rankings_easiest_has_highest_rank():
    """Team allowing most FP gets the highest rank."""
    result = calculate_defensive_rankings(_weekly_stats_df())
    wr = result[result["position"] == "WR"].sort_values("defense_rank", ascending=False)
    assert wr.iloc[0]["team"] == "DAL"
    assert wr.iloc[0]["defense_rank"] == 3


def test_rankings_fp_per_game_accuracy():
    """FP allowed per game matches manual calculation."""
    result = calculate_defensive_rankings(_weekly_stats_df())
    nyg_wr = result[(result["team"] == "NYG") & (result["position"] == "WR")]
    assert nyg_wr.iloc[0]["fp_allowed_per_game"] == pytest.approx(15.0)


def test_rankings_empty_input():
    """Empty input returns empty DataFrame."""
    result = calculate_defensive_rankings(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_rankings_without_week_column():
    """Works without a week column (fallback games count)."""
    df = _weekly_stats_df().drop(columns=["week"])
    result = calculate_defensive_rankings(df)
    assert len(result) > 0
    assert "defense_rank" in result.columns


def test_rankings_with_raw_stat_columns():
    """Computes total_ppr from raw stat columns when not provided."""
    df = pd.DataFrame({
        "opponent_team": ["DAL", "DAL"],
        "position": ["WR", "RB"],
        "passing_yards": [0, 0],
        "passing_tds": [0, 0],
        "passing_interceptions": [0, 0],
        "rushing_yards": [0, 100],
        "rushing_tds": [0, 1],
        "receiving_yards": [100, 0],
        "receiving_tds": [1, 0],
        "receptions": [5, 0],
        "sack_fumbles_lost": [0, 0],
        "rushing_fumbles_lost": [0, 0],
        "receiving_fumbles_lost": [0, 0],
    })
    result = calculate_defensive_rankings(df)
    assert len(result) > 0
    assert "fp_allowed_per_game" in result.columns


def test_rankings_missing_columns_raises():
    """Missing raw stat columns raises ValueError."""
    df = pd.DataFrame({"opponent_team": ["DAL"], "position": ["WR"]})
    with pytest.raises(ValueError, match="missing required columns"):
        calculate_defensive_rankings(df)


def test_rankings_within_position_ordering():
    """Ranks are ascending within each position."""
    result = calculate_defensive_rankings(_weekly_stats_df())
    for pos in ["WR", "RB"]:
        subset = result[result["position"] == pos]["defense_rank"].tolist()
        assert subset == sorted(subset)


# ---------------------------------------------------------------------------
# compute_matchup_multiplier
# ---------------------------------------------------------------------------
def test_multiplier_rank1_toughest():
    """Rank 1 returns 1.0 - max_penalty."""
    assert compute_matchup_multiplier(1) == pytest.approx(0.85)


def test_multiplier_rank32_easiest():
    """Rank 32 returns 1.0 + max_boost."""
    assert compute_matchup_multiplier(32) == pytest.approx(1.15)


def test_multiplier_rank16_near_neutral():
    """Rank 16 maps to approximately 1.00 (slightly below)."""
    val = compute_matchup_multiplier(16)
    assert val == pytest.approx(0.995, abs=0.01)


def test_multiplier_rank17_near_neutral():
    """Rank 17 maps to approximately 1.00 (slightly above)."""
    val = compute_matchup_multiplier(17)
    assert val == pytest.approx(1.005, abs=0.01)


def test_multiplier_monotonically_increasing():
    """Multiplier increases as rank increases (easier matchup)."""
    vals = [compute_matchup_multiplier(r) for r in range(1, 33)]
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1]


def test_multiplier_custom_bounds():
    """Custom max_boost and max_penalty are respected."""
    assert compute_matchup_multiplier(1, max_boost=0.20, max_penalty=0.10) == pytest.approx(0.90)
    assert compute_matchup_multiplier(32, max_boost=0.20, max_penalty=0.10) == pytest.approx(1.20)


def test_multiplier_clamped_below_1():
    """Rank below 1 is clamped to 1."""
    assert compute_matchup_multiplier(0) == compute_matchup_multiplier(1)
    assert compute_matchup_multiplier(-5) == compute_matchup_multiplier(1)


def test_multiplier_clamped_above_32():
    """Rank above 32 is clamped to 32."""
    assert compute_matchup_multiplier(50) == compute_matchup_multiplier(32)
    assert compute_matchup_multiplier(100) == compute_matchup_multiplier(32)


def test_multiplier_zero_bounds_gives_neutral():
    """Zero boost and zero penalty always returns 1.0."""
    assert compute_matchup_multiplier(1, max_boost=0, max_penalty=0) == 1.0
    assert compute_matchup_multiplier(32, max_boost=0, max_penalty=0) == 1.0


# ---------------------------------------------------------------------------
# adjust_projections_for_matchup
# ---------------------------------------------------------------------------
def test_adjust_returns_dataframe():
    """adjust_projections_for_matchup returns a DataFrame."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    assert isinstance(result, pd.DataFrame)


def test_adjust_added_columns():
    """Result has the four new DvP columns."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    for col in ["opponent", "defense_rank", "multiplier", "adjusted_proj_points"]:
        assert col in result.columns


def test_adjust_opponent_assigned():
    """Each player gets their opponent from the schedule."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    phi_wr = result[(result["player_id"] == "p2")]
    assert phi_wr.iloc[0]["opponent"] == "DAL"


def test_adjust_multiplier_applied():
    """adjusted_proj_points = proj_points * multiplier."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    for _, row in result.iterrows():
        expected = round(row["proj_points"] * row["multiplier"], 2)
        assert row["adjusted_proj_points"] == pytest.approx(expected)


def test_adjust_easy_matchup_boosts():
    """Player facing a weak defense (high rank) gets boosted projection."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    # PHI WR faces DAL (rank 3 for WR, easiest) -> boost
    phi_wr = result[result["player_id"] == "p2"].iloc[0]
    assert phi_wr["multiplier"] > 1.0
    assert phi_wr["adjusted_proj_points"] > phi_wr["proj_points"]


def test_adjust_tough_matchup_penalizes():
    """Player facing a strong defense (low rank) gets reduced projection."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    # NYG WR faces PHI -> PHI not in dvp_ranks -> fallback rank 16 -> ~neutral
    # But let's test with a schedule where NYG faces NYG-rank-1 defense
    sched = pd.DataFrame({"team": ["PHI"], "opponent_team": ["NYG"]})
    dvp = pd.DataFrame({
        "team": ["NYG"], "position": ["WR"],
        "fp_allowed_per_game": [15.0], "defense_rank": [1],
    })
    proj = pd.DataFrame({
        "player_id": ["p1"], "player_name": ["WR A"],
        "position": ["WR"], "team": ["PHI"], "proj_points": [15.0],
    })
    result = adjust_projections_for_matchup(proj, sched, dvp)
    assert result.iloc[0]["multiplier"] < 1.0
    assert result.iloc[0]["adjusted_proj_points"] < 15.0


def test_adjust_bye_week_neutral():
    """Player with no schedule entry (bye) gets multiplier 1.0."""
    proj = pd.DataFrame({
        "player_id": ["p1"], "player_name": ["WR A"],
        "position": ["WR"], "team": ["BYE"], "proj_points": [15.0],
    })
    result = adjust_projections_for_matchup(proj, _schedule_df(), _dvp_ranks_df())
    assert result.iloc[0]["multiplier"] == pytest.approx(1.0)
    assert result.iloc[0]["adjusted_proj_points"] == pytest.approx(15.0)


def test_adjust_no_matching_rank_fallback():
    """Player whose opponent has no DvP rank gets neutral multiplier."""
    proj = pd.DataFrame({
        "player_id": ["p1"], "player_name": ["WR A"],
        "position": ["WR"], "team": ["PHI"], "proj_points": [15.0],
    })
    sched = pd.DataFrame({"team": ["PHI"], "opponent_team": ["SEA"]})
    # No SEA in dvp_ranks_df
    result = adjust_projections_for_matchup(proj, sched, _dvp_ranks_df())
    assert result.iloc[0]["multiplier"] == pytest.approx(1.0, abs=0.01)


def test_adjust_empty_projections():
    """Empty projections returns empty result with DvP columns."""
    result = adjust_projections_for_matchup(
        pd.DataFrame(columns=["player_id", "position", "team", "proj_points"]),
        _schedule_df(), _dvp_ranks_df(),
    )
    assert len(result) == 0
    for col in ["opponent", "defense_rank", "multiplier", "adjusted_proj_points"]:
        assert col in result.columns


def test_adjust_empty_schedule():
    """Empty schedule treats all players as bye (multiplier 1.0)."""
    result = adjust_projections_for_matchup(
        _projections_df(), pd.DataFrame(columns=["team", "opponent_team"]), _dvp_ranks_df(),
    )
    for val in result["multiplier"]:
        assert val == pytest.approx(1.0, abs=0.01)


def test_adjust_multi_position_scaling():
    """Different positions get different multipliers based on DvP."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    # PHI RB faces DAL (rank 2 for RB), PHI WR faces DAL (rank 3 for WR)
    phi_rb = result[result["player_id"] == "p3"].iloc[0]
    phi_wr = result[result["player_id"] == "p2"].iloc[0]
    # Different positions -> different DvP ranks -> different multipliers
    assert phi_rb["multiplier"] != phi_wr["multiplier"]


def test_adjust_preserves_original_columns():
    """Original projection columns are preserved."""
    result = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    for col in ["player_id", "player_name", "position", "team", "proj_points"]:
        assert col in result.columns
    assert len(result) == len(_projections_df())


# ---------------------------------------------------------------------------
# enhance_projections_with_dvp (matchups.py wrapper)
# ---------------------------------------------------------------------------
def test_enhance_wrapper_returns_dataframe():
    """enhance_projections_with_dvp returns a DataFrame."""
    result = enhance_projections_with_dvp(
        _projections_df(), _dvp_ranks_df(), _schedule_df(),
    )
    assert isinstance(result, pd.DataFrame)


def test_enhance_wrapper_has_dvp_columns():
    """Wrapper result includes DvP adjustment columns."""
    result = enhance_projections_with_dvp(
        _projections_df(), _dvp_ranks_df(), _schedule_df(),
    )
    for col in ["opponent", "defense_rank", "multiplier", "adjusted_proj_points"]:
        assert col in result.columns


def test_enhance_wrapper_matches_core_function():
    """Wrapper produces the same result as calling adjust_projections_for_matchup directly."""
    from src.dvp import adjust_projections_for_matchup
    from src.matchups import enhance_projections_with_dvp
    core = adjust_projections_for_matchup(
        _projections_df(), _schedule_df(), _dvp_ranks_df(),
    )
    wrapper = enhance_projections_with_dvp(
        _projections_df(), _dvp_ranks_df(), _schedule_df(),
    )
    pd.testing.assert_frame_equal(core, wrapper)
