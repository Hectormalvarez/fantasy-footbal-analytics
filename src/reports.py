"""Weekly digest and multi-format report generator."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from src.matchups import extract_weekly_matchup_roster, optimize_starting_lineup
from src.simulation import simulate_team_matchup, calculate_league_power_rankings
from src.waivers import build_waiver_recommendations, rank_drop_candidates
from src.injuries import extract_roster_injury_status, validate_starting_lineup_health
from src.dvp import calculate_defensive_rankings, adjust_projections_for_matchup
from src.league import fetch_league_matchups


def generate_weekly_digest(
    league_id: str,
    week: int,
    roster_id: int,
    rosters_df: pd.DataFrame,
    draft_board_df: pd.DataFrame,
    catalog_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Consolidate all weekly analytics into a single digest dictionary.

    Parameters
    ----------
    league_id : Sleeper league ID.
    week : NFL week number.
    roster_id : Target roster ID.
    rosters_df : Exploded roster DataFrame with ``roster_id``,
        ``owner_id``, ``player_id``.
    draft_board_df : Draft board with ``player_id``, ``player_name``,
        ``position_proj``, ``proj_points``.
    catalog_df : Optional Sleeper catalog for injury lookups.

    Returns
    -------
    Dict with keys:
    - ``league_id``, ``week``, ``roster_id``
    - ``optimal_lineup``: dict from optimize_starting_lineup
    - ``total_proj_points``: float
    - ``matchup``: dict with ``win_prob_a``, ``median_a``,
      ``floor_a``, ``ceiling_a``
    - ``opponent_roster_id``: int or None
    - ``waivers``: DataFrame of top 5 waiver targets (or empty)
    - ``drop_candidates``: DataFrame of drop candidates (or empty)
    - ``injury_warnings``: list of warning dicts
    - ``power_rankings``: DataFrame of league-wide power rankings
    - ``roster_power_rank``: int or None
    """
    if catalog_df is None:
        catalog_df = pd.DataFrame()

    # Ensure player_id is available on the board
    board = draft_board_df.copy()
    if "player_id" not in board.columns and "norm_name" in board.columns:
        from src.sleeper import fetch_sleeper_players, parse_sleeper_catalog
        cat = parse_sleeper_catalog(fetch_sleeper_players())
        id_lookup = dict(zip(cat["norm_name"], cat["player_id"]))
        board["player_id"] = board["norm_name"].map(id_lookup)

    # Fetch weekly matchups
    matchups_raw = fetch_league_matchups(league_id, week)

    player_pool = board[["player_id", "player_name", "position_proj"]].copy()
    player_pool.rename(columns={"position_proj": "position"}, inplace=True)

    matchup_df = extract_weekly_matchup_roster(roster_id, matchups_raw, player_pool)

    # --- Optimal Lineup ---
    proj = board[["player_id", "player_name", "position_proj", "proj_points"]].copy()
    proj.rename(columns={"position_proj": "position"}, inplace=True)
    # Convert season-long (17-game) projections to weekly
    proj["proj_points"] = proj["proj_points"] / 17.0
    roster_ids = matchup_df["player_id"].tolist() if not matchup_df.empty else []
    optimal = optimize_starting_lineup(roster_ids, proj)

    # --- Opponent & Win Probability ---
    opponent_id = None
    if not matchup_df.empty and "opponent_roster_id" in matchup_df.columns:
        opp_vals = matchup_df["opponent_roster_id"].dropna()
        if not opp_vals.empty:
            opponent_id = int(opp_vals.iloc[0])

    sim_result: dict[str, Any] = {
        "win_prob_a": 0.5, "median_a": 0.0,
        "floor_a": 0.0, "ceiling_a": 0.0,
    }
    if opponent_id is not None:
        opp_matchup_df = extract_weekly_matchup_roster(
            opponent_id, matchups_raw, player_pool,
        )
        opp_roster_ids = opp_matchup_df[opp_matchup_df["is_starter"]]["player_id"].tolist()
        opp_proj = proj[proj["player_id"].isin(opp_roster_ids)][[
            "player_id", "position", "proj_points",
        ]].copy()
        team_a = proj[proj["player_id"].isin(optimal["lineup"].values())].copy()
        sim_result = simulate_team_matchup(team_a, opp_proj, iterations=3000)

    # --- Waivers ---
    waivers_df = pd.DataFrame()
    drops_df = pd.DataFrame()
    if roster_ids:
        # Use weekly projections for waiver recommendations
        board_weekly = board.copy()
        board_weekly["proj_points"] = board_weekly["proj_points"] / 17.0
        waivers_df = build_waiver_recommendations(
            roster_id, rosters_df, board_weekly, remaining_faab=100,
        )
        drops_df = rank_drop_candidates(roster_ids, proj)

    # --- Injuries ---
    injury_warnings: list[dict[str, str]] = []
    if not catalog_df.empty and optimal["lineup"]:
        injury_warnings = validate_starting_lineup_health(optimal["lineup"], catalog_df)

    # --- Power Rankings ---
    power_df = pd.DataFrame()
    my_rank: int | None = None
    if not rosters_df.empty:
        power_df = calculate_league_power_rankings(
            rosters_df, proj, iterations=500,
        )
        if not power_df.empty:
            rank_row = power_df[power_df["roster_id"] == roster_id]
            if not rank_row.empty:
                my_rank = int(rank_row.iloc[0]["power_rank"])

    # Build name lookup for markdown export
    name_lookup = dict(zip(board["player_id"], board["player_name"]))

    return {
        "league_id": league_id,
        "week": week,
        "roster_id": roster_id,
        "optimal_lineup": optimal,
        "total_proj_points": optimal["total_proj_points"],
        "matchup": sim_result,
        "opponent_roster_id": opponent_id,
        "waivers": waivers_df.head(5) if not waivers_df.empty else waivers_df,
        "drop_candidates": drops_df.head(5) if not drops_df.empty else drops_df,
        "injury_warnings": injury_warnings,
        "power_rankings": power_df,
        "roster_power_rank": my_rank,
        "player_name_lookup": name_lookup,
    }


def export_digest_markdown(
    digest_data: dict[str, Any],
    filepath: str,
) -> str:
    """Convert digest dictionary into a cleanly formatted Markdown report.

    Parameters
    ----------
    digest_data : Output of :func:`generate_weekly_digest`.
    filepath : Destination path for the .md file.  Parent directories
        are created automatically.

    Returns
    -------
    Absolute path of the written file.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    week = digest_data.get("week", "?")
    roster_id = digest_data.get("roster_id", "?")
    matchup = digest_data.get("matchup", {})
    optimal = digest_data.get("optimal_lineup", {})
    waivers = digest_data.get("waivers", pd.DataFrame())
    drops = digest_data.get("drop_candidates", pd.DataFrame())
    injuries = digest_data.get("injury_warnings", [])
    power = digest_data.get("power_rankings", pd.DataFrame())
    my_rank = digest_data.get("roster_power_rank")

    lines: list[str] = []
    lines.append(f"# Weekly Digest -- Week {week}, Roster {roster_id}")
    lines.append("")

    # --- Matchup Win Probability ---
    lines.append("## Matchup Projection")
    lines.append("")
    if matchup:
        wp = matchup.get("win_prob_a", 0.5) * 100
        med = matchup.get("median_a", 0.0)
        floor = matchup.get("floor_a", 0.0)
        ceil = matchup.get("ceiling_a", 0.0)
        opp_id = digest_data.get("opponent_roster_id")
        opp_str = f"vs Roster {opp_id}" if opp_id else "(no opponent)"
        lines.append(f"**Win Probability:** {wp:.1f}% {opp_str}")
        lines.append(f"**Projected Score:** {med:.1f} (Floor: {floor:.1f}, Ceiling: {ceil:.1f})")
    else:
        lines.append("No matchup data available.")
    lines.append("")

    # --- Optimal Lineup ---
    lines.append("## Optimal Lineup")
    lines.append("")
    total = digest_data.get("total_proj_points", 0.0)
    lineup = optimal.get("lineup", {})
    name_lookup = digest_data.get("player_name_lookup", {})
    if lineup:
        lines.append(f"**Total Projected:** {total:.1f} pts")
        lines.append("")
        lines.append("| Slot | Player |")
        lines.append("|------|--------|")
        for slot in sorted(lineup):
            pid = lineup[slot]
            name = name_lookup.get(pid, pid)
            lines.append(f"| {slot} | {name} |")
    else:
        lines.append("No optimal lineup available.")
    lines.append("")

    # --- Injury Alerts ---
    if injuries:
        lines.append("## Injury Alerts")
        lines.append("")
        for w in injuries:
            body = f" ({w.get('injury_body_part', '')})" if w.get("injury_body_part") else ""
            lines.append(f"- **{w['player_name']}** [{w['injury_status']}]{body} at {w['slot']}")
        lines.append("")

    # --- Waiver Targets ---
    if not waivers.empty:
        lines.append("## Top Waiver Targets")
        lines.append("")
        lines.append("| Player | Pos | Proj Pts | Marginal Value |")
        lines.append("|--------|-----|----------|----------------|")
        for _, row in waivers.iterrows():
            lines.append(
                f"| {row.get('player_name', '')} | {row.get('position', '')} | "
                f"{row.get('proj_points', 0):.1f} | {row.get('marginal_value', 0):.1f} |"
            )
        lines.append("")

    # --- Drop Candidates ---
    if not drops.empty:
        lines.append("## Drop Candidates")
        lines.append("")
        lines.append("| Rank | Player | Pos | Proj Pts |")
        lines.append("|------|--------|-----|----------|")
        for _, row in drops.iterrows():
            lines.append(
                f"| {int(row.get('drop_rank', 0))} | {row.get('player_name', '')} | "
                f"{row.get('position', '')} | {row.get('proj_points', 0):.1f} |"
            )
        lines.append("")

    # --- Power Rankings ---
    if not power.empty:
        lines.append("## League Power Rankings")
        lines.append("")
        if my_rank is not None:
            lines.append(f"**Your Power Rank:** {my_rank}")
            lines.append("")
        lines.append("| Rank | Roster | Win % | Median Pts |")
        lines.append("|------|--------|-------|------------|")
        for _, row in power.iterrows():
            rid = int(row.get("roster_id", 0))
            rnk = int(row.get("power_rank", 0))
            wpct = row.get("true_talent_win_pct", 0) * 100
            med = row.get("median_total", 0)
            marker = " **<-- YOU**" if rid == roster_id else ""
            lines.append(f"| {rnk} | Roster {rid} | {wpct:.1f}% | {med:.1f} |{marker}")
        lines.append("")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by fantasy-footbal-analytics*")

    content = "\n".join(lines)
    abs_path = os.path.abspath(filepath)
    with open(abs_path, "w") as f:
        f.write(content)
    return abs_path


def export_power_rankings_csv(
    power_rankings_df: pd.DataFrame,
    filepath: str,
) -> str:
    """Export league-wide all-play power rankings to CSV.

    Parameters
    ----------
    power_rankings_df : DataFrame from :func:`calculate_league_power_rankings`.
    filepath : Destination path for the .csv file.

    Returns
    -------
    Absolute path of the written file.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    abs_path = os.path.abspath(filepath)
    power_rankings_df.to_csv(abs_path, index=False)
    return abs_path
