"""Tests for src/scoring.py — Full-PPR scoring logic."""

import pandas as pd

from src.scoring import calculate_ppr_points


def test_passing_touchdown_only():
    """A QB who throws 1 TD and nothing else should score 4.0 pts."""
    df = pd.DataFrame({
        "passing_yards": [0],
        "passing_tds": [1],
        "passing_interceptions": [0],
        "rushing_yards": [0],
        "rushing_tds": [0],
        "receiving_yards": [0],
        "receiving_tds": [0],
        "receptions": [0],
        "sack_fumbles_lost": [0],
        "rushing_fumbles_lost": [0],
        "receiving_fumbles_lost": [0],
    })
    pts = calculate_ppr_points(df)
    assert pts.iloc[0] == 4.0
