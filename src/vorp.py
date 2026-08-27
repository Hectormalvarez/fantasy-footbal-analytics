"""VORP engine and ADP arbitrage signals."""

import pandas as pd


def classify_signal(adp_delta: float) -> str:
    """Classify an ADP delta into an emoji signal.

    Parameters
    ----------
    adp_delta : search_rank - vorp_rank. Positive = value, negative = overpriced.

    Returns
    -------
    One of: "Major Value", "Slight Value", "Fair Value", "Overpriced", "Heavy Reach".
    """
    if adp_delta >= 10:
        return "\U0001f525 Major Value"
    if 5 <= adp_delta < 10:
        return "\u2705 Slight Value"
    if -5 < adp_delta < 5:
        return "\u2696\ufe0f Fair Value"
    if -10 < adp_delta <= -5:
        return "\u26a0\ufe0f Overpriced"
    # adp_delta <= -10
    return "\U0001f6ab Heavy Reach"


def calculate_vorb(
    df: pd.DataFrame,
    baselines: dict[str, float],
) -> pd.DataFrame:
    """Calculate VORP (Value Over Replacement Player) for each player.

    Parameters
    ----------
    df : DataFrame with ``player_name``, ``position_proj``, ``proj_points``.
    baselines : dict mapping position -> baseline points (e.g. {"QB": 350.0}).

    Returns
    -------
    Copy of df with added columns: ``vorp``, ``vorp_rank``, ``pos_rank``, ``pos_label``.
    """
    result = df.copy()
    result["baseline"] = result["position_proj"].map(baselines).fillna(0)
    result["vorp"] = (result["proj_points"] - result["baseline"]).clip(lower=0)

    # Overall VORP rank (1 = best)
    result["vorp_rank"] = result["vorp"].rank(ascending=False, method="min").astype(int)

    # Positional rank within each position
    result["pos_rank"] = (
        result.groupby("position_proj")["vorp"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # pos_label: e.g. "WR1", "RB12"
    result["pos_label"] = result["position_proj"] + result["pos_rank"].astype(str)

    return result.sort_values("vorp_rank").reset_index(drop=True)


def build_draft_board(
    df: pd.DataFrame,
    baselines: dict[str, float],
) -> pd.DataFrame:
    """Build a full draft board with VORP, ADP delta, and arbitrage signals.

    Parameters
    ----------
    df : DataFrame with ``player_name``, ``position_proj``, ``proj_points``, ``search_rank``.
    baselines : dict mapping position -> baseline points.

    Returns
    -------
    DataFrame sorted by vorp_rank with adp_delta and signal columns added.
    """
    board = calculate_vorb(df, baselines)
    board["adp_delta"] = board["search_rank"].astype(float) - board["vorp_rank"].astype(float)
    board["signal"] = board["adp_delta"].apply(classify_signal)
    return board.sort_values("vorp_rank").reset_index(drop=True)
