"""Serpentine draft matrix and contingency sheet generator."""

import pandas as pd


def get_snake_picks(
    draft_slot: int,
    num_teams: int = 14,
    rounds: int = 15,
) -> list[tuple[int, int]]:
    """Return (round, pick_number) tuples for a serpentine draft.

    Parameters
    ----------
    draft_slot : Your draft position (1-indexed).
    num_teams  : Number of teams in the league (default 14).
    rounds     : Number of draft rounds to simulate (default 15).

    Returns
    -------
    list of (round_num, overall_pick) tuples.
    """
    picks = []
    for r in range(1, rounds + 1):
        if r % 2 == 1:  # odd round -> forward
            pick = (r - 1) * num_teams + draft_slot
        else:            # even round -> reverse
            pick = r * num_teams - draft_slot + 1
        picks.append((r, pick))
    return picks


def generate_contingency_sheet(
    board: pd.DataFrame,
    draft_slot: int,
    num_teams: int = 14,
    rounds: int = 15,
    reach_buffer: int = 4,
    fall_buffer: int = 8,
) -> pd.DataFrame:
    """Build a round-by-round contingency draft sheet.

    Parameters
    ----------
    board : DataFrame with player_name, position_proj, proj_points, vorp,
            vorp_rank, search_rank, and signal columns.
    draft_slot : Your draft position (1-indexed).
    num_teams  : League size (default 14).
    rounds     : Rounds to plan (default 15).
    reach_buffer : How many picks before your pick to consider (default 4).
    fall_buffer  : How many picks after your pick to consider (default 8).

    Returns
    -------
    DataFrame with round, pick, tier, player info, and signal columns.
    """
    picks = get_snake_picks(draft_slot, num_teams, rounds)
    rows = []
    drafted_ids: set = set()

    for rnd, pk in picks:
        lo = max(1, pk - reach_buffer)
        hi = pk + fall_buffer

        pool = board[
            (~board.index.isin(drafted_ids))
            & (board["search_rank"].astype(float) >= lo)
            & (board["search_rank"].astype(float) <= hi)
        ].copy()

        pool = pool.sort_values("vorp", ascending=False)

        is_value = pool["signal"].str.contains("Major Value", na=False)
        primary   = pool[~is_value].head(3)
        secondary = pool[~is_value].iloc[3:6]
        value     = pool[is_value].head(3)

        def _add_rows(sub: pd.DataFrame, tier: str):
            for idx, p in sub.iterrows():
                rows.append({
                    "round":       rnd,
                    "pick":        pk,
                    "tier":        tier,
                    "vorp_rank":   int(p["vorp_rank"]),
                    "pos_label":   p["pos_label"],
                    "player":      p["player_name"],
                    "team":        p.get("team", ""),
                    "position":    p["position_proj"],
                    "proj_pts":    round(p["proj_points"], 1),
                    "vorp":        round(p["vorp"], 1),
                    "search_rank": int(p["search_rank"]),
                    "adp_delta":   round(p["adp_delta"], 1),
                    "signal":      p["signal"],
                })

        _add_rows(primary,   "Primary")
        _add_rows(secondary, "Secondary")
        _add_rows(value,     "Value")

        # Mark primary targets as drafted for subsequent rounds
        for idx in primary.index:
            drafted_ids.add(idx)

    return pd.DataFrame(rows)