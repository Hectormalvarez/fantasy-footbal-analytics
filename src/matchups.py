"""Weekly sit/start lineup optimizer and matchup engine."""

from typing import Any

import pandas as pd

# Default roster slots for a standard 1QB league
DEFAULT_SLOTS = [
    "QB", "RB1", "RB2", "WR1", "WR2", "WR3",
    "TE", "FLEX", "BN", "BN", "BN", "BN", "BN", "BN", "BN",
]

# Positions eligible for the FLEX slot (RB/WR/TE)
FLEX_ELIGIBLE = {"RB", "WR", "TE"}


def extract_weekly_matchup_roster(
    roster_id: int,
    week_matchups_raw: list[dict],
    player_pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract a roster's weekly matchup details from raw Sleeper data.

    Parameters
    ----------
    roster_id : Target roster to extract.
    week_matchups_raw : Raw matchup list from
        ``fetch_league_matchups(league_id, week)``.
    player_pool_df : DataFrame with ``player_id``, ``player_name``,
        ``position`` columns.

    Returns
    -------
    DataFrame with columns ``player_id``, ``player_name``, ``position``,
    ``is_starter``, ``opponent_roster_id``, ``actual_points``,
    ``proj_points``.  Empty DataFrame if the roster is not found.
    """
    out_cols = [
        "player_id",
        "player_name",
        "position",
        "is_starter",
        "opponent_roster_id",
        "actual_points",
        "proj_points",
    ]

    # Find the target matchup
    target_matchup = None
    for m in week_matchups_raw:
        if m.get("roster_id") == roster_id:
            target_matchup = m
            break

    if target_matchup is None:
        return pd.DataFrame(columns=out_cols)

    matchup_id = target_matchup.get("matchup_id")
    starter_set = set(target_matchup.get("starters") or [])
    all_players = target_matchup.get("players") or []
    players_points = target_matchup.get("players_points") or {}

    # Find opponent roster_id
    opponent_id = None
    for m in week_matchups_raw:
        if m.get("matchup_id") == matchup_id and m.get("roster_id") != roster_id:
            opponent_id = m.get("roster_id")
            break

    # Build player pool lookup
    pool_lookup = {}
    if not player_pool_df.empty:
        for _, row in player_pool_df.iterrows():
            pool_lookup[row["player_id"]] = {
                "player_name": row.get("player_name", "Unknown"),
                "position": row.get("position", "UNKNOWN"),
            }

    rows = []
    for pid in all_players:
        info = pool_lookup.get(pid, {"player_name": "Unknown", "position": "UNKNOWN"})
        rows.append({
            "player_id": pid,
            "player_name": info["player_name"],
            "position": info["position"],
            "is_starter": pid in starter_set,
            "opponent_roster_id": opponent_id,
            "actual_points": players_points.get(pid, 0.0),
            "proj_points": 0.0,
        })

    return pd.DataFrame(rows)[out_cols]


def optimize_starting_lineup(
    rostered_player_ids: list[str],
    projections_df: pd.DataFrame,
    roster_slots: list[str] | None = None,
) -> dict[str, Any]:
    """Assign roster players to legal starting slots to maximize projected points.

    Parameters
    ----------
    rostered_player_ids : List of Sleeper player_id strings on the roster.
    projections_df : Projection table with at least
        ``player_id``, ``position``, ``proj_points``.
    roster_slots : Optional custom slot list.  Defaults to a standard 1QB
        roster: QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, BN×7.

    Returns
    -------
    Dict with keys:
    - ``lineup``: dict mapping slot label → player_id
    - ``total_proj_points``: float sum of starting lineup projections
    - ``bench``: list of player_ids sorted descending by projection
    """
    if roster_slots is None:
        roster_slots = list(DEFAULT_SLOTS)

    if not rostered_player_ids:
        return {"lineup": {}, "total_proj_points": 0, "bench": []}

    # Build projection lookup
    proj_lookup: dict[str, float] = {}
    pos_lookup: dict[str, str] = {}
    if not projections_df.empty:
        for _, row in projections_df.iterrows():
            proj_lookup[row["player_id"]] = float(row.get("proj_points", 0))
            pos_lookup[row["player_id"]] = row.get("position", "UNKNOWN")

    # Prepare player list sorted by projection descending
    players = []
    for pid in rostered_player_ids:
        players.append({
            "player_id": pid,
            "proj": proj_lookup.get(pid, 0.0),
            "position": pos_lookup.get(pid, "UNKNOWN"),
        })
    players.sort(key=lambda x: x["proj"], reverse=True)

    # Classify slots
    position_slots: list[str] = []  # QB, RB1, RB2, WR1, WR2, WR3, TE
    flex_slots: list[str] = []
    bench_slots: list[str] = []
    for slot in roster_slots:
        base = slot.rstrip("0123456789")
        if slot == "FLEX":
            flex_slots.append(slot)
        elif base in {"QB", "RB", "WR", "TE"}:
            position_slots.append(slot)
        else:
            bench_slots.append(slot)

    # Assign position-specific slots greedily
    used: set[str] = set()
    lineup: dict[str, str] = {}

    for slot in position_slots:
        base_pos = slot.rstrip("0123456789")
        for p in players:
            if p["player_id"] not in used and p["position"] == base_pos:
                lineup[slot] = p["player_id"]
                used.add(p["player_id"])
                break

    # Assign FLEX slots
    for slot in flex_slots:
        for p in players:
            if p["player_id"] not in used and p["position"] in FLEX_ELIGIBLE:
                lineup[slot] = p["player_id"]
                used.add(p["player_id"])
                break

    # Bench remaining
    bench = [p["player_id"] for p in players if p["player_id"] not in used]

    total = sum(proj_lookup.get(pid, 0.0) for pid in lineup.values())

    return {"lineup": lineup, "total_proj_points": total, "bench": bench}


# Tolerance for "toss-up" classification (points)
_TOSSUP_TOLERANCE = 3.0


def compare_sit_start(
    player_a_id: str,
    player_b_id: str,
    projections_df: pd.DataFrame,
) -> dict[str, Any]:
    """Head-to-head sit/start comparison between two players.

    Parameters
    ----------
    player_a_id : Sleeper player_id for the first option.
    player_b_id : Sleeper player_id for the second option.
    projections_df : Projection table with at least
        ``player_id``, ``player_name``, ``position``, ``proj_points``.

    Returns
    -------
    Dict with keys:
    - ``player_a``: sub-dict with ``id``, ``name``, ``position``,
      ``proj_points``, ``floor``, ``ceiling``
    - ``player_b``: same structure
    - ``delta``: float (player_a proj − player_b proj)
    - ``recommendation``: "Start <Name>", "Start <Name>", or "Toss-up"
    """
    def _player_info(pid: str) -> dict:
        if projections_df.empty:
            return {"id": pid, "name": "Unknown", "position": "UNKNOWN",
                    "proj_points": 0.0, "floor": 0.0, "ceiling": 0.0}
        row = projections_df[projections_df["player_id"] == pid]
        if row.empty:
            return {"id": pid, "name": "Unknown", "position": "UNKNOWN",
                    "proj_points": 0.0, "floor": 0.0, "ceiling": 0.0}
        r = row.iloc[0]
        proj = float(r.get("proj_points", 0))
        return {
            "id": pid,
            "name": r.get("player_name", "Unknown"),
            "position": r.get("position", "UNKNOWN"),
            "proj_points": proj,
            "floor": round(proj * 0.7, 1),
            "ceiling": round(proj * 1.3, 1),
        }

    pa = _player_info(player_a_id)
    pb = _player_info(player_b_id)
    delta = round(pa["proj_points"] - pb["proj_points"], 1)

    if abs(delta) < _TOSSUP_TOLERANCE:
        recommendation = "Toss-up"
    elif delta > 0:
        recommendation = f"Start {pa['name']}"
    else:
        recommendation = f"Start {pb['name']}"

    return {
        "player_a": pa,
        "player_b": pb,
        "delta": delta,
        "recommendation": recommendation,
    }