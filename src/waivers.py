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


def rank_drop_candidates(
    roster_player_ids: list[str],
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify the lowest projected assets on the active roster.

    Parameters
    ----------
    roster_player_ids : list of Sleeper player_id strings on the manager's roster.
    projections_df : projection table with at least
        ``player_id``, ``player_name``, ``position``, ``proj_points``.

    Returns
    -------
    DataFrame sorted ascending by ``proj_points`` with an added ``drop_rank``
    column (1 = most droppable).
    """
    if not roster_player_ids or projections_df.empty:
        return pd.DataFrame(
            columns=["player_id", "player_name", "position", "proj_points", "drop_rank"]
        )

    roster = projections_df[projections_df["player_id"].isin(roster_player_ids)].copy()
    roster = roster.sort_values("proj_points", ascending=True).reset_index(drop=True)
    roster["drop_rank"] = range(1, len(roster) + 1)
    return roster[["player_id", "player_name", "position", "proj_points", "drop_rank"]]


def recommend_faab_bids(
    upgrades_df: pd.DataFrame,
    remaining_faab: int,
    min_bid: int = 0,
) -> pd.DataFrame:
    """Calculate tiered FAAB bid recommendations scaled to marginal value.

    Parameters
    ----------
    upgrades_df : DataFrame output of :func:`calculate_marginal_roster_value`
        with at least ``player_id``, ``player_name``, ``position``,
        ``marginal_value`` columns.
    remaining_faab : Manager's remaining FAAB budget.
    min_bid : Minimum allowed bid (league floor, often $0).

    Returns
    -------
    DataFrame with columns ``player_id``, ``player_name``, ``position``,
    ``marginal_value``, ``conservative_bid``, ``market_bid``,
    ``aggressive_bid``.
    """
    out_cols = [
        "player_id",
        "player_name",
        "position",
        "marginal_value",
        "conservative_bid",
        "market_bid",
        "aggressive_bid",
    ]

    if upgrades_df.empty:
        return pd.DataFrame(columns=out_cols)

    result = upgrades_df[["player_id", "player_name", "position", "marginal_value"]].copy()
    total_mv = result["marginal_value"].sum()

    if total_mv <= 0 or remaining_faab <= 0:
        result["conservative_bid"] = 0
        result["market_bid"] = 0
        result["aggressive_bid"] = 0
    else:
        shares = result["marginal_value"] / total_mv
        result["conservative_bid"] = (
            (shares * remaining_faab * 0.3).clip(lower=min_bid, upper=remaining_faab).round().astype(int)
        )
        result["market_bid"] = (
            (shares * remaining_faab * 0.5).clip(lower=min_bid, upper=remaining_faab).round().astype(int)
        )
        result["aggressive_bid"] = (
            (shares * remaining_faab * 0.7).clip(lower=min_bid, upper=remaining_faab).round().astype(int)
        )

    # Enforce min_bid when budget exists but total_mv was zero
    if remaining_faab > 0:
        for col in ("conservative_bid", "market_bid", "aggressive_bid"):
            result[col] = result[col].clip(lower=min_bid)

    return result[out_cols]


def build_waiver_recommendations(
    roster_id: int,
    rosters_df: pd.DataFrame,
    draft_board_df: pd.DataFrame,
    remaining_faab: int,
) -> pd.DataFrame:
    """End-to-end waiver recommendation for a single roster.

    Parameters
    ----------
    roster_id : Target roster to evaluate.
    rosters_df : Exploded roster DataFrame with columns
        ``roster_id``, ``owner_id``, ``player_id``.
    draft_board_df : Draft-board DataFrame with columns ``player_id``,
        ``player_name``, ``position_proj``, ``proj_points`` (and any other
        VORP columns that may be present).
    remaining_faab : Manager's remaining FAAB budget.

    Returns
    -------
    Merged DataFrame of upgrade candidates with FAAB bids, or an empty
    DataFrame if no qualifying targets exist.
    """
    roster_players = rosters_df[rosters_df["roster_id"] == roster_id]["player_id"].tolist()
    if not roster_players:
        return pd.DataFrame()

    all_rostered = set(rosters_df["player_id"].unique())
    waiver_pool = draft_board_df[~draft_board_df["player_id"].isin(all_rostered)].copy()

    # Normalise the position column for downstream functions
    proj = draft_board_df.rename(columns={"position_proj": "position"})
    waiver_pool = waiver_pool.rename(columns={"position_proj": "position"})

    if waiver_pool.empty:
        return pd.DataFrame()

    upgrades = calculate_marginal_roster_value(roster_players, waiver_pool, proj)
    if upgrades.empty:
        return pd.DataFrame()

    bids = recommend_faab_bids(upgrades, remaining_faab)
    return bids
