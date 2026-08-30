"""Waiver wire evaluator and FAAB allocation engine."""

import pandas as pd


def calculate_marginal_roster_value(
    roster_player_ids: list[str],
    waiver_pool_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute net PPG improvement against the manager's positional bench baseline.

    Parameters
    ----------
    roster_player_ids : list of Sleeper player_id strings on the manager's roster.
    waiver_pool_df : DataFrame of free-agent targets with columns
        ``player_id``, ``player_name``, ``position``, ``proj_points``.
    projections_df : Full projection table with the same four columns used
        to establish each position's baseline.

    Returns
    -------
    DataFrame of waiver targets that improve over the baseline, with columns
    ``player_id``, ``player_name``, ``position``, ``proj_points``,
    ``baseline``, ``marginal_value``.  Empty if nothing qualifies.
    """
    if not roster_player_ids or waiver_pool_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "position",
                "proj_points",
                "baseline",
                "marginal_value",
            ]
        )

    roster = projections_df[projections_df["player_id"].isin(roster_player_ids)]
    baselines = roster.groupby("position")["proj_points"].min().to_dict()

    result = waiver_pool_df.copy()
    result["baseline"] = result["position"].map(baselines).fillna(0.0)
    result["marginal_value"] = result["proj_points"] - result["baseline"]

    result = result[result["marginal_value"] > 0].copy()
    result = result.sort_values("marginal_value", ascending=False).reset_index(drop=True)
    return result[["player_id", "player_name", "position", "proj_points", "baseline", "marginal_value"]]
