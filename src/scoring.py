"""Full-PPR scoring rules and projection math."""

import pandas as pd


def calculate_ppr_points(df: pd.DataFrame) -> pd.Series:
    """Calculate Full-PPR fantasy points from nflreadpy stat columns.

    Scoring rules
    -------------
    - Passing:  0.04 pts/yd, +4.0/TD, -2.0/INT
    - Rushing:  0.1  pts/yd, +6.0/TD
    - Receiving: 0.1 pts/yd, +6.0/TD, +1.0/reception (PPR)
    - Fumbles:  -2.0/lost (sack + rushing + receiving)
    """
    pts = pd.Series(0.0, index=df.index)

    pts += df["passing_yards"].fillna(0) * 0.04
    pts += df["passing_tds"].fillna(0) * 4.0
    pts += df["passing_interceptions"].fillna(0) * (-2.0)

    pts += df["rushing_yards"].fillna(0) * 0.1
    pts += df["rushing_tds"].fillna(0) * 6.0

    pts += df["receiving_yards"].fillna(0) * 0.1
    pts += df["receiving_tds"].fillna(0) * 6.0
    pts += df["receptions"].fillna(0) * 1.0

    fumbles_lost = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )
    pts += fumbles_lost * (-2.0)

    return pts
