"""Defense-vs-Position (DvP) matchup multipliers and projection adjuster."""

from __future__ import annotations

import pandas as pd

from src.scoring import calculate_ppr_points


def calculate_defensive_rankings(
    weekly_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute full-PPR fantasy points allowed per game by each defense vs skill position.

    Parameters
    ----------
    weekly_stats_df : Game-level stats with columns ``opponent_team``,
        ``position``, and either ``total_ppr`` or the raw stat columns
        accepted by :func:`src.scoring.calculate_ppr_points`.
        An optional ``week`` column improves per-game accuracy.

    Returns
    -------
    DataFrame with columns ``team``, ``position``, ``fp_allowed_per_game``,
        ``defense_rank`` (1 = toughest, 32 = easiest within each position).
    """
    if weekly_stats_df.empty:
        return pd.DataFrame(
            columns=["team", "position", "fp_allowed_per_game", "defense_rank"]
        )

    df = weekly_stats_df.copy()

    # Ensure total_ppr column exists
    if "total_ppr" not in df.columns:
        required = [
            "passing_yards", "passing_tds", "passing_interceptions",
            "rushing_yards", "rushing_tds",
            "receiving_yards", "receiving_tds", "receptions",
            "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"weekly_stats_df missing required columns: {missing}")
        df["total_ppr"] = calculate_ppr_points(df)

    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["team", "position", "fp_allowed_per_game", "defense_rank"]
        )

    # Count games per opponent team
    if "week" in df.columns:
        games = df.groupby("opponent_team")["week"].nunique()
    else:
        n_teams = df["opponent_team"].nunique()
        if n_teams == 0:
            return pd.DataFrame(
                columns=["team", "position", "fp_allowed_per_game", "defense_rank"]
            )
        games = pd.Series(17, index=df["opponent_team"].unique(), name="games")

    games.name = "games"

    # Aggregate FP allowed per position per team
    agg = (
        df.groupby(["opponent_team", "position"])["total_ppr"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "team"})
    )
    agg = agg.merge(games, left_on="team", right_index=True, how="left")
    agg["fp_allowed_per_game"] = (agg["total_ppr"] / agg["games"]).round(2)

    # Rank within each position: 1 = fewest FP allowed (toughest)
    agg["defense_rank"] = (
        agg.groupby("position")["fp_allowed_per_game"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return (
        agg[["team", "position", "fp_allowed_per_game", "defense_rank"]]
        .sort_values(["position", "defense_rank"])
        .reset_index(drop=True)
    )


def compute_matchup_multiplier(
    defense_rank: int,
    max_boost: float = 0.15,
    max_penalty: float = 0.15,
) -> float:
    """Calculate a normalized scaling multiplier from opponent DvP ranking.

    Parameters
    ----------
    defense_rank : Opponent's defensive rank (1 = toughest, 32 = easiest).
    max_boost : Maximum positive adjustment (for easiest matchup, rank 32).
    max_penalty : Maximum negative adjustment (for toughest matchup, rank 1).

    Returns
    -------
    Multiplier between ``1.0 - max_penalty`` and ``1.0 + max_boost``.
    Rank 16-17 maps to approximately 1.00.
    """
    rank = max(1, min(32, defense_rank))
    normalized = (rank - 1) / 31.0
    return 1.0 - max_penalty + normalized * (max_boost + max_penalty)


def adjust_projections_for_matchup(
    projections_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    dvp_ranks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply opponent matchup multipliers to player baseline projections.

    Parameters
    ----------
    projections_df : Player projections with columns ``player_id``,
        ``position``, ``team``, ``proj_points``.
    schedule_df : Weekly schedule with columns ``team``, ``opponent_team``.
    dvp_ranks_df : DvP rankings from :func:`calculate_defensive_rankings`
        with columns ``team``, ``position``, ``defense_rank``.

    Returns
    -------
    DataFrame with all original columns plus ``opponent``, ``defense_rank``,
        ``multiplier``, ``adjusted_proj_points``.  Players without a
        schedule entry (bye) receive multiplier 1.0.
    """
    if projections_df.empty:
        return projections_df.assign(
            opponent=pd.Series(dtype=str),
            defense_rank=pd.Series(dtype=int),
            multiplier=pd.Series(dtype=float),
            adjusted_proj_points=pd.Series(dtype=float),
        )

    result = projections_df.copy()

    # Join with schedule to get each player's opponent
    sched = (
        schedule_df[["team", "opponent_team"]].copy()
        if not schedule_df.empty and "team" in schedule_df.columns
        else pd.DataFrame(columns=["team", "opponent_team"])
    )
    # Strip whitespace from team names (schedule data may have leading spaces)
    sched["team"] = sched["team"].str.strip()
    sched["opponent_team"] = sched["opponent_team"].str.strip()
    result["team"] = result["team"].str.strip()
    result = result.merge(sched, on="team", how="left")
    result["opponent"] = result["opponent_team"].fillna("BYE")
    result.drop(columns=["opponent_team"], errors="ignore", inplace=True)

    # Join with DvP ranks on (opponent, position)
    dvp = dvp_ranks_df[["team", "position", "defense_rank"]].copy()
    dvp.rename(columns={"team": "def_team"}, inplace=True)

    result = result.merge(
        dvp,
        left_on=["opponent", "position"],
        right_on=["def_team", "position"],
        how="left",
    )
    result.drop(columns=["def_team"], errors="ignore", inplace=True)

    # Compute multiplier (BYE / missing -> neutral 1.0)
    bye_mask = result["opponent"] == "BYE"
    result["defense_rank"] = result["defense_rank"].fillna(16).astype(int)
    result.loc[bye_mask, "multiplier"] = 1.0
    result.loc[bye_mask, "adjusted_proj_points"] = result.loc[bye_mask, "proj_points"]
    non_bye = ~bye_mask
    if non_bye.any():
        result.loc[non_bye, "multiplier"] = result.loc[non_bye, "defense_rank"].apply(
            compute_matchup_multiplier
        )
        result.loc[non_bye, "adjusted_proj_points"] = (
            result.loc[non_bye, "proj_points"] * result.loc[non_bye, "multiplier"]
        ).round(2)

    return result
