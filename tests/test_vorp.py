"""Tests for src/vorp.py — VORP engine and ADP arbitrage signals."""

import pandas as pd

from src.vorp import classify_signal, calculate_vorb


def test_major_value():
    """adp_delta >= 10 -> Major Value."""
    assert classify_signal(10) == "\U0001f525 Major Value"
    assert classify_signal(15) == "\U0001f525 Major Value"


def test_slight_value():
    """5 <= adp_delta < 10 -> Slight Value."""
    assert classify_signal(5) == "\u2705 Slight Value"
    assert classify_signal(9) == "\u2705 Slight Value"


def test_fair_value():
    """-5 < adp_delta < 5 -> Fair Value."""
    assert classify_signal(0) == "\u2696\ufe0f Fair Value"
    assert classify_signal(-4) == "\u2696\ufe0f Fair Value"
    assert classify_signal(4) == "\u2696\ufe0f Fair Value"


def test_overpriced():
    """-10 < adp_delta <= -5 -> Overpriced."""
    assert classify_signal(-5) == "\u26a0\ufe0f Overpriced"
    assert classify_signal(-9) == "\u26a0\ufe0f Overpriced"


def test_heavy_reach():
    """adp_delta <= -10 -> Heavy Reach."""
    assert classify_signal(-10) == "\U0001f6ab Heavy Reach"
    assert classify_signal(-20) == "\U0001f6ab Heavy Reach"



# ---------------------------------------------------------------------------
# calculate_vorp tests
# ---------------------------------------------------------------------------
def _player_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal player DataFrame from row dicts."""
    return pd.DataFrame(rows)


def test_vorp_above_baseline():
    """Player above baseline gets positive VORP."""
    df = _player_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 300.0},
    ])
    baselines = {"WR": 200.0}
    result = calculate_vorb(df, baselines)
    assert result.iloc[0]["vorp"] == 100.0


def test_vorp_below_baseline():
    """Player below baseline gets VORP floored at 0."""
    df = _player_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 150.0},
    ])
    baselines = {"WR": 200.0}
    result = calculate_vorb(df, baselines)
    assert result.iloc[0]["vorp"] == 0.0


def test_vorp_exact_baseline():
    """Player exactly at baseline gets VORP of 0."""
    df = _player_df([
        {"player_name": "A", "position_proj": "QB", "proj_points": 350.0},
    ])
    baselines = {"QB": 350.0}
    result = calculate_vorb(df, baselines)
    assert result.iloc[0]["vorp"] == 0.0


def test_vorb_multiple_positions():
    """VORP is calculated correctly across multiple positions."""
    df = _player_df([
        {"player_name": "QB1", "position_proj": "QB", "proj_points": 400.0},
        {"player_name": "RB1", "position_proj": "RB", "proj_points": 280.0},
        {"player_name": "WR1", "position_proj": "WR", "proj_points": 300.0},
    ])
    baselines = {"QB": 350.0, "RB": 200.0, "WR": 200.0}
    result = calculate_vorb(df, baselines)
    by_name = result.set_index("player_name")
    assert by_name.loc["WR1", "vorp"] == 100.0
    assert by_name.loc["RB1", "vorp"] == 80.0
    assert by_name.loc["QB1", "vorp"] == 50.0


def test_vorp_rank_assigns_correctly():
    """vorp_rank ranks players by VORP descending."""
    df = _player_df([
        {"player_name": "A", "position_proj": "WR", "proj_points": 300.0},
        {"player_name": "B", "position_proj": "RB", "proj_points": 280.0},
        {"player_name": "C", "position_proj": "QB", "proj_points": 400.0},
    ])
    baselines = {"QB": 400.0, "RB": 200.0, "WR": 200.0}
    result = calculate_vorb(df, baselines)
    by_name = result.set_index("player_name")
    assert by_name.loc["A", "vorp_rank"] == 1  # VORP=100
    assert by_name.loc["B", "vorp_rank"] == 2  # VORP=80
    assert by_name.loc["C", "vorp_rank"] == 3  # VORP=0


def test_vorp_pos_rank():
    """pos_rank assigns positional rank (WR1, RB1, etc.)."""
    df = _player_df([
        {"player_name": "WR1", "position_proj": "WR", "proj_points": 300.0},
        {"player_name": "WR2", "position_proj": "WR", "proj_points": 250.0},
        {"player_name": "RB1", "position_proj": "RB", "proj_points": 280.0},
    ])
    baselines = {"WR": 200.0, "RB": 200.0}
    result = calculate_vorb(df, baselines)
    by_name = result.set_index("player_name")
    assert by_name.loc["WR1", "pos_rank"] == 1
    assert by_name.loc["WR2", "pos_rank"] == 2
    assert by_name.loc["RB1", "pos_rank"] == 1