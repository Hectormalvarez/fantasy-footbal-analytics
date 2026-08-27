"""Tests for src/scoring.py — Full-PPR scoring logic."""

import pandas as pd

from src.scoring import calculate_ppr_points, project_17_game_ppr

# ---------------------------------------------------------------------------
# Shared fixture: zeroed-out stat row
# ---------------------------------------------------------------------------
ZERO_ROW = {
    "passing_yards": [0],
    "passing_tds": [0],
    "passing_interceptions": [0],
    "rushing_yards": [0],
    "rushing_tds": [0],
    "receiving_yards": [0],
    "receiving_tds": [0],
    "receptions": [0],
    "sack_fumbles_lost": [0],
    "rushing_fumbles_lost": [0],
    "receiving_fumbles_lost": [0],
}


def _row(**overrides) -> pd.DataFrame:
    """Return a single-row DataFrame with optional stat overrides."""
    data = {k: v[0] for k, v in ZERO_ROW.items()}
    data.update(overrides)
    return pd.DataFrame({k: [v] for k, v in data.items()})


def test_passing_touchdown_only():
    """1 passing TD = 4.0 pts."""
    df = _row(passing_tds=1)
    assert calculate_ppr_points(df).iloc[0] == 4.0


def test_rushing_touchdown_only():
    """1 rushing TD = 6.0 pts."""
    df = _row(rushing_tds=1)
    assert calculate_ppr_points(df).iloc[0] == 6.0


def test_receiving_touchdown_only():
    """1 receiving TD = 6.0 pts."""
    df = _row(receiving_tds=1)
    assert calculate_ppr_points(df).iloc[0] == 6.0


def test_interception_penalty():
    """1 INT = -2.0 pts."""
    df = _row(passing_interceptions=1)
    assert calculate_ppr_points(df).iloc[0] == -2.0


def test_fumble_lost_penalty():
    """1 sack fumble lost = -2.0 pts."""
    df = _row(sack_fumbles_lost=1)
    assert calculate_ppr_points(df).iloc[0] == -2.0


def test_reception_ppr():
    """5 receptions = 5.0 pts (PPR)."""
    df = _row(receptions=5)
    assert calculate_ppr_points(df).iloc[0] == 5.0


def test_passing_yards():
    """400 passing yards = 16.0 pts."""
    df = _row(passing_yards=400)
    assert calculate_ppr_points(df).iloc[0] == 16.0


def test_rushing_yards():
    """100 rushing yards = 10.0 pts."""
    df = _row(rushing_yards=100)
    assert calculate_ppr_points(df).iloc[0] == 10.0


def test_receiving_yards():
    """100 receiving yards = 10.0 pts."""
    df = _row(receiving_yards=100)
    assert calculate_ppr_points(df).iloc[0] == 10.0


def test_composite_qb_line():
    """Realistic QB stat line: 300 yds, 3 TD, 1 INT, 20 rush yds, 1 rush TD, -2 fumbles."""
    df = _row(
        passing_yards=300, passing_tds=3, passing_interceptions=1,
        rushing_yards=20, rushing_tds=1,
        sack_fumbles_lost=1,
    )
    # 300*.04 + 3*4 + 1*(-2) + 20*.1 + 1*6 + 1*(-2) = 12 + 12 - 2 + 2 + 6 - 2 = 28.0
    assert calculate_ppr_points(df).iloc[0] == 28.0


def test_composite_wr_line():
    """Realistic WR stat line: 8 rec, 120 yds, 1 TD."""
    df = _row(receptions=8, receiving_yards=120, receiving_tds=1)
    # 8*1 + 120*.1 + 1*6 = 8 + 12 + 6 = 26.0
    assert calculate_ppr_points(df).iloc[0] == 26.0


def test_multi_row_batch():
    """Scoring works across multiple rows (batch calculation)."""
    df = pd.DataFrame({
        "passing_yards":        [0, 200],
        "passing_tds":          [0, 2],
        "passing_interceptions":[0, 1],
        "rushing_yards":        [50, 0],
        "rushing_tds":          [1, 0],
        "receiving_yards":      [0, 0],
        "receiving_tds":        [0, 0],
        "receptions":           [0, 0],
        "sack_fumbles_lost":    [0, 0],
        "rushing_fumbles_lost": [1, 0],
        "receiving_fumbles_lost":[0, 0],
    })
    pts = calculate_ppr_points(df)
    # Row 0: 50*.1 + 1*6 + 1*(-2) = 5 + 6 - 2 = 9.0
    assert pts.iloc[0] == 9.0
    # Row 1: 200*.04 + 2*4 + 1*(-2) = 8 + 8 - 2 = 14.0
    assert pts.iloc[1] == 14.0



# ---------------------------------------------------------------------------
# project_17_game_ppr tests
# ---------------------------------------------------------------------------
def test_project_simple_17_game_pace():
    """100 PPR pts in 10 games -> (100/10)*17 = 170.0 projected."""
    df = pd.DataFrame({"total_ppr": [100.0], "games": [10]})
    result = project_17_game_ppr(df)
    assert result.iloc[0] == 170.0


def test_project_full_season_no_scale():
    """170 PPR pts in 17 games -> (170/17)*17 = 170.0 (no scaling)."""
    df = pd.DataFrame({"total_ppr": [170.0], "games": [17]})
    result = project_17_game_ppr(df)
    assert result.iloc[0] == 170.0


def test_project_floor_at_min_games():
    """Player with 3 games: divisor floored at 6.
    60 PPR pts -> (60/6)*17 = 170.0, NOT (60/3)*17 = 340.0."""
    df = pd.DataFrame({"total_ppr": [60.0], "games": [3]})
    result = project_17_game_ppr(df)
    assert result.iloc[0] == 170.0


def test_project_floor_exactly_at_threshold():
    """Player with exactly min_games (6): no floor applied.
    60 PPR pts -> (60/6)*17 = 170.0."""
    df = pd.DataFrame({"total_ppr": [60.0], "games": [6]})
    result = project_17_game_ppr(df)
    assert result.iloc[0] == 170.0


def test_project_custom_min_games():
    """Custom min_games=4: 40 PPR pts in 2 games -> (40/4)*17 = 170.0."""
    df = pd.DataFrame({"total_ppr": [40.0], "games": [2]})
    result = project_17_game_ppr(df, min_games=4)
    assert result.iloc[0] == 170.0


def test_project_multi_row():
    """Projection works across multiple rows."""
    df = pd.DataFrame({
        "total_ppr": [170.0, 100.0, 60.0],
        "games":     [17,     10,    3],
    })
    result = project_17_game_ppr(df)
    assert result.iloc[0] == 170.0   # (170/17)*17
    assert result.iloc[1] == 170.0   # (100/10)*17
    assert result.iloc[2] == 170.0   # (60/6)*17  — floor kicks in