"""Multi-team trade evaluator & starting lineup impact engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.matchups import optimize_starting_lineup


def simulate_roster_swap(
    roster_player_ids: list[str],
    outgoing_ids: list[str],
    incoming_ids: list[str],
) -> list[str]:
    """Validate and apply player swaps to produce a post-trade roster.

    Parameters
    ----------
    roster_player_ids : Current player_id strings on the roster.
    outgoing_ids : Player_id strings being sent away.
    incoming_ids : Player_id strings being acquired.

    Returns
    -------
    New roster as a list of player_id strings.

    Raises
    ------
    ValueError : If an outgoing player is not on the roster or an incoming
        player is already on the roster.
    """
    roster = set(roster_player_ids)

    for pid in outgoing_ids:
        if pid not in roster:
            raise ValueError(f"Player {pid!r} not on roster")
    for pid in incoming_ids:
        if pid in roster:
            raise ValueError(f"Player {pid!r} already on roster")

    roster -= set(outgoing_ids)
    roster |= set(incoming_ids)
    return sorted(roster)


def calculate_starting_lineup_delta(
    pre_roster_ids: list[str],
    post_roster_ids: list[str],
    projections_df: pd.DataFrame,
    roster_slots: list[str] | None = None,
) -> float:
    """Calculate the net weekly starting PPG change from a trade.

    Uses ``optimize_starting_lineup`` to determine the optimal starters for
    both the pre-trade and post-trade rosters, then returns the difference
    in total projected points.

    Parameters
    ----------
    pre_roster_ids : Player_id strings on the roster before the trade.
    post_roster_ids : Player_id strings on the roster after the trade.
    projections_df : Projection table with at least
        ``player_id``, ``position``, ``proj_points``.
    roster_slots : Optional custom slot list passed to the optimizer.

    Returns
    -------
    Float representing post-trade starting PPG minus pre-trade starting PPG.
    """
    pre = optimize_starting_lineup(pre_roster_ids, projections_df, roster_slots)
    post = optimize_starting_lineup(post_roster_ids, projections_df, roster_slots)
    return round(post["total_proj_points"] - pre["total_proj_points"], 2)


# Verdict classification thresholds
_PPG_STRONG_ACCEPT = 5.0
_VORP_STRONG_ACCEPT = 20.0
_PPG_SLIGHT_UPGRADE = 1.0
_VORP_SLIGHT_UPGRADE = 5.0
_PPG_LATERAL = 0.5
_VORP_LATERAL = 2.5


def classify_trade_verdict(
    net_starting_delta: float, net_vorp_delta: float
) -> str:
    """Categorize trade impact into an actionable recommendation.

    Parameters
    ----------
    net_starting_delta : Net change in weekly starting projected PPG.
    net_vorp_delta : Net change in rest-of-season VORP.

    Returns
    -------
    One of: ``"Strong Accept"``, ``"Slight Upgrade"``, ``"Fair Trade"``,
    ``"Lateral Move"``, ``"Decline / Value Loss"``.
    """
    # Both clearly negative → decline
    if net_starting_delta < 0 and net_vorp_delta < 0:
        return "Decline / Value Loss"

    # Either side negative (mixed but net-negative impact)
    if net_starting_delta < 0 or net_vorp_delta < 0:
        return "Decline / Value Loss"

    # Both positive — classify by magnitude
    if net_starting_delta >= _PPG_STRONG_ACCEPT and net_vorp_delta >= _VORP_STRONG_ACCEPT:
        return "Strong Accept"

    if net_starting_delta >= _PPG_SLIGHT_UPGRADE and net_vorp_delta >= _VORP_SLIGHT_UPGRADE:
        return "Slight Upgrade"

    if net_starting_delta < _PPG_LATERAL and net_vorp_delta < _VORP_LATERAL:
        return "Lateral Move"

    return "Fair Trade"
