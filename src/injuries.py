"""Injury status tracking and IR stash optimizer."""

from __future__ import annotations

from typing import Any

import pandas as pd


# Status values that trigger severity warnings in starting lineups
_HIGH_SEVERITY = {"Out", "IR"}
_MEDIUM_SEVERITY = {"Doubtful"}

# Status values that qualify a player for IR/reserve slot placement
_IR_ELIGIBLE_STATUSES = {"Out", "IR", "Doubtful", "Sus"}

# Display tag mapping
_TAG_MAP = {
    "Questionable": "[Q]",
    "Doubtful": "[D]",
    "Out": "[OUT]",
    "IR": "[IR]",
    "Sus": "[SUS]",
}


def extract_roster_injury_status(
    rostered_player_ids: list[str],
    sleeper_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    """Parse injury status for all rostered players from the Sleeper catalog.

    Parameters
    ----------
    rostered_player_ids : List of Sleeper player_id strings on the roster.
    sleeper_catalog_df : Sleeper catalog DataFrame with columns ``player_id``,
        ``player_name``, ``injury_status``, ``injury_body_part``,
        ``injury_notes``.

    Returns
    -------
    DataFrame with columns ``player_id``, ``player_name``,
        ``injury_status``, ``injury_body_part``, ``injury_notes``,
        ``injury_tag``.  Empty DataFrame if no players match.
    """
    out_cols = [
        "player_id", "player_name", "injury_status",
        "injury_body_part", "injury_notes", "injury_tag",
    ]
    if not rostered_player_ids or sleeper_catalog_df.empty:
        return pd.DataFrame(columns=out_cols)

    cat = sleeper_catalog_df[
        sleeper_catalog_df["player_id"].isin(rostered_player_ids)
    ].copy()

    if cat.empty:
        return pd.DataFrame(columns=out_cols)

    cols = [c for c in [
        "player_id", "player_name", "injury_status",
        "injury_body_part", "injury_notes",
    ] if c in cat.columns]
    result = cat[cols].copy()
    result["injury_tag"] = result["injury_status"].map(_TAG_MAP).fillna("")
    return result.reset_index(drop=True)


def validate_starting_lineup_health(
    starting_lineup: dict[str, Any],
    sleeper_catalog_df: pd.DataFrame,
) -> list[dict[str, str]]:
    """Scan active starters and generate severity warnings for injured players.

    Parameters
    ----------
    starting_lineup : Dict mapping slot names to player_id strings
        (e.g., ``{"QB": "p1", "RB1": "p2", ...}``).
    sleeper_catalog_df : Sleeper catalog DataFrame with columns ``player_id``,
        ``player_name``, ``injury_status``, ``injury_body_part``.

    Returns
    -------
    List of warning dicts with keys ``player_id``, ``player_name``,
        ``slot``, ``injury_status``, ``injury_body_part``, ``severity``
        (``"high"`` for Out/IR, ``"medium"`` for Doubtful).
        Empty list if all starters are healthy.
    """
    if not starting_lineup or sleeper_catalog_df.empty:
        return []

    cat = sleeper_catalog_df.set_index("player_id")

    warnings: list[dict[str, str]] = []
    for slot, pid in starting_lineup.items():
        if pid not in cat.index:
            continue
        row = cat.loc[pid]
        status = row.get("injury_status")
        if status in _HIGH_SEVERITY:
            warnings.append({
                "player_id": pid,
                "player_name": row.get("player_name", "Unknown"),
                "slot": slot,
                "injury_status": status,
                "injury_body_part": row.get("injury_body_part", ""),
                "severity": "high",
            })
        elif status in _MEDIUM_SEVERITY:
            warnings.append({
                "player_id": pid,
                "player_name": row.get("player_name", "Unknown"),
                "slot": slot,
                "injury_status": status,
                "injury_body_part": row.get("injury_body_part", ""),
                "severity": "medium",
            })

    return warnings


def identify_ir_eligible_stashes(
    roster_dict: dict[str, Any],
    sleeper_catalog_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Identify injured bench players eligible for IR/reserve slots.

    Parameters
    ----------
    roster_dict : Dict with keys ``starters`` (list of player_id) and
        ``bench`` (list of player_id).  Any other keys are ignored.
    sleeper_catalog_df : Sleeper catalog DataFrame with columns ``player_id``,
        ``player_name``, ``position``, ``injury_status``,
        ``injury_body_part``.

    Returns
    -------
    List of dicts for each IR-eligible bench player with keys
        ``player_id``, ``player_name``, ``position``, ``injury_status``,
        ``injury_body_part``.  Empty list if none qualify.
    """
    if not roster_dict or sleeper_catalog_df.empty:
        return []

    starter_set = set(roster_dict.get("starters") or [])
    bench_ids = [
        pid for pid in (roster_dict.get("bench") or [])
        if pid not in starter_set
    ]

    if not bench_ids:
        return []

    cat = sleeper_catalog_df.set_index("player_id")

    stashes: list[dict[str, Any]] = []
    for pid in bench_ids:
        if pid not in cat.index:
            continue
        row = cat.loc[pid]
        status = row.get("injury_status")
        if status in _IR_ELIGIBLE_STATUSES:
            stashes.append({
                "player_id": pid,
                "player_name": row.get("player_name", "Unknown"),
                "position": row.get("position", "UNKNOWN"),
                "injury_status": status,
                "injury_body_part": row.get("injury_body_part", ""),
            })

    return stashes