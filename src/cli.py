"""CLI entry point for the fantasy football analytics pipeline.

Usage::

    python -m src.cli draft --slot 14
    python -m src.cli waivers --league-id 12345 --roster-id 1 --faab 100
    python -m src.cli weekly --league-id 12345 --roster-id 1 --week 5
    python -m src.cli trade --league-id 12345 --team-a-roster-id 1 \\
        --team-b-roster-id 2 --team-a-sends p1,p2 --team-b-sends p3,p4
"""

import argparse
import os
import sys

import pandas as pd

from src.sleeper import fetch_sleeper_players, parse_sleeper_catalog, clean_player_name
from src.draft import generate_contingency_sheet
from src.league import (
    fetch_league_rosters,
    parse_rosters_dataframe,
    fetch_league_matchups,
)
from src.waivers import build_waiver_recommendations, rank_drop_candidates
from src.matchups import extract_weekly_matchup_roster, optimize_starting_lineup
from src.trade import evaluate_trade
from src.injuries import extract_roster_injury_status, validate_starting_lineup_health
from src.dvp import calculate_defensive_rankings, adjust_projections_for_matchup
from src.simulation import simulate_team_matchup
from src.reports import generate_weekly_digest, export_digest_markdown, export_power_rankings_csv


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="fantasy-analytics",
        description="Fantasy football analytics -- draft board, waivers, weekly lineup, trade evaluator",
    )
    subs = parser.add_subparsers(dest="command", help="Available commands")

    # -- draft --
    dp = subs.add_parser("draft", help="Pre-draft contingency matrix generation")
    dp.add_argument("--slot", type=int, required=True, choices=range(1, 15), help="Assigned draft slot (1-14).")
    dp.add_argument("--refresh", action="store_true", default=False, help="Force a fresh Sleeper API download.")
    dp.add_argument("--rounds", type=int, default=15, help="Draft depth in rounds (default: 15).")
    dp.add_argument("--reach-buffer", type=int, default=4, dest="reach_buffer", help="Max picks to reach ahead of ADP (default: 4).")
    dp.add_argument("--fall-buffer", type=int, default=8, dest="fall_buffer", help="Max picks to catch falling value (default: 8).")
    dp.add_argument("--visuals", action="store_true", default=False, help="Generate visual cheat sheets (PNG + HTML).")
    dp.set_defaults(func=handle_draft)

    # -- waivers --
    wp = subs.add_parser("waivers", help="Waiver wire recommendations & FAAB bids")
    wp.add_argument("--league-id", type=str, required=True, dest="league_id", help="Sleeper league ID.")
    wp.add_argument("--roster-id", type=int, required=True, dest="roster_id", help="Target roster ID.")
    wp.add_argument("--refresh", action="store_true", default=False, help="Force a fresh Sleeper API download.")
    wp.add_argument("--faab", type=int, default=100, help="Remaining FAAB budget (default: 100).")
    wp.add_argument("--output", type=str, default=None, help="Optional CSV output path.")
    wp.set_defaults(func=handle_waivers)

    # -- weekly --
    wk = subs.add_parser("weekly", help="Weekly sit/start lineup optimizer")
    wk.add_argument("--league-id", type=str, required=True, dest="league_id", help="Sleeper league ID.")
    wk.add_argument("--roster-id", type=int, required=True, dest="roster_id", help="Target roster ID.")
    wk.add_argument("--week", type=int, required=True, help="NFL week number.")
    wk.add_argument("--refresh", action="store_true", default=False, help="Force a fresh Sleeper API download.")
    wk.add_argument("--visuals", action="store_true", default=False, help="Generate lineup visualizations (PNG).")
    wk.set_defaults(func=handle_weekly)

    # -- trade --
    tp = subs.add_parser("trade", help="Two-way trade evaluator")
    tp.add_argument("--league-id", type=str, required=True, dest="league_id", help="Sleeper league ID.")
    tp.add_argument("--team-a-roster-id", type=int, required=True, dest="team_a_roster_id", help="Team A roster ID.")
    tp.add_argument("--team-b-roster-id", type=int, required=True, dest="team_b_roster_id", help="Team B roster ID.")
    tp.add_argument("--team-a-sends", type=str, required=True, dest="team_a_sends", help="Comma-separated player IDs team A sends.")
    tp.add_argument("--team-b-sends", type=str, required=True, dest="team_b_sends", help="Comma-separated player IDs team B sends.")
    tp.add_argument("--refresh", action="store_true", default=False, help="Force a fresh Sleeper API download.")
    tp.set_defaults(func=handle_trade)

    # -- report --
    rp = subs.add_parser("report", help="Generate weekly digest report")
    rp.add_argument("--league-id", type=str, required=True, dest="league_id", help="Sleeper league ID.")
    rp.add_argument("--roster-id", type=int, required=True, dest="roster_id", help="Target roster ID.")
    rp.add_argument("--week", type=int, required=True, help="NFL week number.")
    rp.add_argument("--output-dir", type=str, default="reports", dest="output_dir", help="Output directory (default: reports).")
    rp.add_argument("--refresh", action="store_true", default=False, help="Force a fresh Sleeper API download.")
    rp.set_defaults(func=handle_report)

    return parser


# ---------------------------------------------------------------------------
# Abbreviation key builder
# ---------------------------------------------------------------------------
def _make_abbrev_key(name: str) -> str:
    """Build an abbreviation-style key from a full player name.

    "Lamar Jackson" -> "l.jackson"
    "Ja'Marr Chase" -> "j.chase"
    "Bijan Robinson" -> "b.robinson"

    This lets us match nflreadpy's ``player_name`` format ("L.Jackson")
    against the Sleeper catalog's full names.
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return clean_player_name(name)
    first_initial = parts[0][0].lower()
    last_name = parts[-1].lower()
    return f"{first_initial}.{last_name}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _load_draft_board(
    projections_path: str = "data/projections.csv",
    sleeper_cache: str = "data/sleeper_players_raw.json",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load projections + Sleeper catalog and build the VORP draft board."""
    projections = pd.read_csv(projections_path)

    raw_sleeper = fetch_sleeper_players(cache_path=sleeper_cache, force_refresh=refresh)
    catalog = parse_sleeper_catalog(raw_sleeper)

    catalog_rank = catalog[["player_name", "position", "search_rank", "team", "player_id"]].copy()
    catalog_rank["norm_name"] = catalog_rank["player_name"].apply(clean_player_name)
    projections["norm_name"] = projections["player_name"].apply(clean_player_name)

    merged = projections.merge(
        catalog_rank[["norm_name", "position", "search_rank", "team", "player_id"]],
        on=["norm_name", "position"], how="left",
        suffixes=("", "_sleeper"),
    )

    # --- Secondary merge for abbreviated names ---
    # nflreadpy stores names like "L.Jackson" while Sleeper has "Lamar Jackson".
    # Build a fallback lookup: "l.jackson" -> catalog row, then match remaining NaN rows.
    unmatched = merged["player_id"].isna()
    if unmatched.any():
        # Build abbreviation key from catalog: "l.jackson" -> catalog entry
        # If multiple catalog players share the same abbreviation+position,
        # keep the one with the best (lowest) search_rank.
        cat_abbrev = catalog_rank.copy()
        cat_abbrev["_abbrev"] = cat_abbrev["player_name"].apply(
            lambda n: _make_abbrev_key(n)
        )
        cat_abbrev = cat_abbrev.sort_values("search_rank").drop_duplicates(
            subset=["_abbrev", "position"], keep="first",
        )
        abbrev_lookup = cat_abbrev.set_index(["_abbrev", "position"])[
            ["search_rank", "team", "player_id"]
        ].to_dict("index")

        for idx in merged.index[unmatched]:
            row = merged.loc[idx]
            key = (row["norm_name"], row["position"])
            if key in abbrev_lookup:
                info = abbrev_lookup[key]
                merged.loc[idx, "player_id"] = info["player_id"]
                merged.loc[idx, "team"] = info.get("team", "")
                if pd.isna(merged.loc[idx, "search_rank"]) or merged.loc[idx, "search_rank"] == 9999:
                    merged.loc[idx, "search_rank"] = info.get("search_rank", 9999)

    merged["search_rank"] = merged["search_rank"].fillna(9999)
    # Use team_sleeper where available, but keep original team as fallback
    if "team_sleeper" in merged.columns:
        merged["team"] = merged["team_sleeper"].fillna(merged["team"])
        merged.drop(columns=["team_sleeper"], inplace=True, errors="ignore")
    merged["team"] = merged["team"].fillna("")

    merged.rename(columns={"position": "position_proj"}, inplace=True)

    starter_cutoffs = {"QB": 14, "RB": 28, "WR": 28, "TE": 14}
    baselines = {}
    for pos, cutoff in starter_cutoffs.items():
        pos_df = merged[merged["position_proj"] == pos].sort_values(
            "proj_points", ascending=False
        )
        baseline_idx = min(cutoff - 1, len(pos_df) - 1)
        baselines[pos] = (
            pos_df.iloc[baseline_idx]["proj_points"] if baseline_idx >= 0 else 0.0
        )

    merged["baseline"] = merged["position_proj"].map(baselines).fillna(0.0)
    merged["vorp"] = merged["proj_points"] - merged["baseline"]

    merged["vorp_rank"] = merged["vorp"].rank(ascending=False, method="min").astype(int)
    # Deduplicate: if both "J.Chase" and "Ja'Marr Chase" resolved to the same
    # player_id, keep the row with the higher projection (the imputed version).
    if merged["player_id"].notna().any():
        merged = merged.sort_values("proj_points", ascending=False).drop_duplicates(
            subset=["player_id"], keep="first",
        )

    merged = merged.sort_values("vorp", ascending=False).reset_index(drop=True)

    merged["pos_label"] = (
        merged["position_proj"]
        + merged.groupby("position_proj").cumcount().add(1).astype(str)
    )
    merged["adp_delta"] = merged["search_rank"] - merged["vorp_rank"]

    def _signal(row):
        if row["adp_delta"] >= 10:
            return "Major Value"
        elif row["adp_delta"] >= 5:
            return "Slight Value"
        elif row["adp_delta"] > -5:
            return "Fair Value"
        elif row["adp_delta"] > -10:
            return "Slight Reach"
        return "Major Reach"

    merged["signal"] = merged.apply(_signal, axis=1)
    return merged


def _load_sleeper_catalog(refresh: bool = False) -> pd.DataFrame:
    """Load the Sleeper NFL player catalog as a DataFrame."""
    raw = fetch_sleeper_players(
        cache_path="data/sleeper_players_raw.json", force_refresh=refresh,
    )
    return parse_sleeper_catalog(raw)


def _load_weekly_stats(refresh: bool = False) -> pd.DataFrame:
    """Load weekly NFL player stats for DvP calculations."""
    cache_path = os.path.join("data", "weekly_stats.csv")
    if not refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    import nflreadpy as nfl

    raw = nfl.load_player_stats(2024, summary_level="week")
    records = raw.rows(named=True)
    df = pd.DataFrame(records)
    if "recent_team" in df.columns:
        df.rename(columns={"recent_team": "team"}, inplace=True)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _format_contingency_sheet(
    sheet: pd.DataFrame, slot: int, num_teams: int = 14
) -> str:
    """Format a contingency sheet DataFrame as a human-readable text block."""
    lines = []
    header = (
        f"{'=' * 72}\n"
        f"  DRAFT CONTINGENCY SHEET  --  Slot {slot} of {num_teams}\n"
        f"{'=' * 72}\n"
    )
    lines.append(header)

    if sheet.empty:
        lines.append("  (no targets found in ADP window)\n")
        lines.append(f"{'=' * 72}\n")
        return "\n".join(lines)

    for rnd in sheet["round"].unique():
        round_df = sheet[sheet["round"] == rnd]
        pick = int(round_df["pick"].iloc[0])
        lines.append(f"--- Round {rnd:>2}  |  Pick {pick:>3}  ---")

        for tier in ["Primary", "Secondary", "Value"]:
            tier_df = round_df[round_df["tier"] == tier]
            if tier_df.empty:
                continue
            lines.append(f"  [{tier}]")
            for _, row in tier_df.iterrows():
                adp = int(row["search_rank"])
                delta = int(row["adp_delta"])
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"    {row['pos_label']:<6s} {row['player']:<24s} "
                    f"VORP {row['vorp']:>5.0f}  Proj {row['proj_pts']:>5.0f}  "
                    f"ADP {adp:>3d} ({sign}{delta:>3d})  {row['signal']}"
                )
        lines.append("")

    lines.append(f"{'=' * 72}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler: draft
# ---------------------------------------------------------------------------
def handle_draft(args) -> None:
    """Run the pre-draft contingency matrix pipeline."""
    data_dir = "data"
    projections_path = os.path.join(data_dir, "projections.csv")
    sleeper_cache = os.path.join(data_dir, "sleeper_players_raw.json")

    print("Loading projections and building VORP draft board ...")
    board = _load_draft_board(
        projections_path=projections_path,
        sleeper_cache=sleeper_cache,
        refresh=args.refresh,
    )

    print(f"\nBoard: {len(board)} players")
    print("\nTop 10 by VORP:")
    top = board.head(10)[["pos_label", "player_name", "proj_points", "vorp", "signal"]]
    print(top.to_string(index=False))

    sheet = generate_contingency_sheet(
        board, draft_slot=args.slot, num_teams=14,
        rounds=args.rounds, reach_buffer=args.reach_buffer,
        fall_buffer=args.fall_buffer,
    )
    formatted = _format_contingency_sheet(sheet, args.slot, num_teams=14)
    print(formatted)

    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"contingency_sheet_slot_{args.slot}.csv")
    txt_path = os.path.join(data_dir, f"contingency_sheet_slot_{args.slot}.txt")
    sheet.to_csv(csv_path, index=False)
    print(f"Exported: {csv_path}")
    with open(txt_path, "w") as fh:
        fh.write(formatted)
    print(f"Exported: {txt_path}")

    if args.visuals:
        from src.visuals import (
            render_round_matrix,
            render_arbitrage_scatter,
            render_positional_cliffs,
            export_styled_html_board,
        )
        print("\nGenerating visual cheat sheets ...")
        render_round_matrix(sheet, args.slot)
        print(f"  -> data/cheat_sheet_slot_{args.slot}.png")
        render_arbitrage_scatter(board)
        print("  -> data/market_arbitrage.png")
        render_positional_cliffs(board)
        print("  -> data/positional_cliffs.png")
        export_styled_html_board(board)
        print("  -> data/draft_board_styled.html")
    print("Done.")


# ---------------------------------------------------------------------------
# Handler: waivers
# ---------------------------------------------------------------------------
def handle_waivers(args) -> None:
    """Fetch rosters and run waiver recommendations."""
    print("Loading projections ...")
    board = _load_draft_board(refresh=args.refresh)
    catalog = _load_sleeper_catalog(refresh=args.refresh)

    print("Syncing league rosters ...")
    rosters = fetch_league_rosters(args.league_id)
    rosters_df = parse_rosters_dataframe(rosters)

    print(f"Building waiver recommendations for roster {args.roster_id} ...")
    # Convert season-long (17-game) projections to weekly for display
    board_weekly = board.copy()
    board_weekly["proj_points"] = board_weekly["proj_points"] / 17.0
    result = build_waiver_recommendations(args.roster_id, rosters_df, board_weekly, args.faab)

    if result.empty:
        print("No qualifying waiver targets found.")
        return

    print("\n--- Waiver Upgrades ---")
    print(result.to_string(index=False))

    # Drop candidates with injury tags
    roster_players = rosters_df[rosters_df["roster_id"] == args.roster_id]["player_id"].tolist()
    injury_df = extract_roster_injury_status(roster_players, catalog)
    injury_lookup = (
        dict(zip(injury_df["player_id"], injury_df["injury_tag"]))
        if not injury_df.empty else {}
    )

    proj = board_weekly[["player_id", "player_name", "position_proj", "proj_points"]].copy()
    proj.rename(columns={"position_proj": "position"}, inplace=True)
    drops = rank_drop_candidates(roster_players, proj)
    if not drops.empty:
        drops = drops.copy()
        drops["injury"] = drops["player_id"].map(lambda pid: injury_lookup.get(pid, ""))
        print("\n--- Drop Candidates ---")
        print(drops.to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"\nExported: {args.output}")


# ---------------------------------------------------------------------------
# Handler: weekly
# ---------------------------------------------------------------------------
def handle_weekly(args) -> None:
    """Optimize starting lineup for a given week."""
    print("Loading projections ...")
    board = _load_draft_board(refresh=args.refresh)
    catalog = _load_sleeper_catalog(refresh=args.refresh)

    # Ensure player_id is available on the board
    if "player_id" not in board.columns and "norm_name" in board.columns:
        id_lookup = dict(zip(catalog["norm_name"], catalog["player_id"]))
        board["player_id"] = board["norm_name"].map(id_lookup)

    print(f"Fetching Week {args.week} matchups ...")
    matchups = fetch_league_matchups(args.league_id, args.week)

    player_pool = board[["player_id", "player_name", "position_proj"]].copy()
    player_pool.rename(columns={"position_proj": "position"}, inplace=True)

    matchup_df = extract_weekly_matchup_roster(args.roster_id, matchups, player_pool)
    if matchup_df.empty:
        print(f"Roster {args.roster_id} not found in Week {args.week} matchups.")
        return

    roster_ids = matchup_df["player_id"].tolist()

    # Load DvP data
    print("Computing DvP rankings ...")
    dvp_ranks = pd.DataFrame(columns=["team", "position", "defense_rank"])
    schedule = pd.DataFrame(columns=["team", "opponent_team"])
    try:
        weekly_stats = _load_weekly_stats(refresh=args.refresh)
        dvp_ranks = calculate_defensive_rankings(weekly_stats)
        week_stats = weekly_stats[weekly_stats["week"] == args.week]
        schedule = week_stats[["team", "opponent_team"]].drop_duplicates()
    except Exception as exc:
        print(f"  (DvP data unavailable: {exc})")

    proj = board[["player_id", "player_name", "position_proj", "proj_points", "team"]].copy()
    proj.rename(columns={"position_proj": "position"}, inplace=True)
    # Convert season-long (17-game) projections to weekly
    proj["proj_points"] = proj["proj_points"] / 17.0

    # Apply DvP adjustments
    adj_proj = adjust_projections_for_matchup(proj, schedule, dvp_ranks)

    # Build lookups
    name_lookup = dict(zip(board["player_id"], board["player_name"]))
    # Fallback: add names from matchup_df for players not on the board
    if not matchup_df.empty:
        matchup_names = dict(zip(matchup_df["player_id"], matchup_df["player_name"]))
        for pid, pname in matchup_names.items():
            if pid not in name_lookup:
                name_lookup[pid] = pname
    pos_lookup = dict(zip(board["player_id"], board["position_proj"]))
    dvp_lookup = {}
    for _, row in adj_proj.iterrows():
        dvp_lookup[row["player_id"]] = {
            "opponent": row.get("opponent", "BYE"),
            "defense_rank": int(row.get("defense_rank", 16)),
            "multiplier": float(row.get("multiplier", 1.0)),
        }

    # Injury lookup
    injury_df = extract_roster_injury_status(roster_ids, catalog)
    injury_lookup = (
        dict(zip(injury_df["player_id"], injury_df["injury_tag"]))
        if not injury_df.empty else {}
    )

    # Use DvP-adjusted projections for optimization
    opt_proj = adj_proj[["player_id", "player_name", "position", "adjusted_proj_points"]].copy()
    opt_proj.rename(columns={"adjusted_proj_points": "proj_points"}, inplace=True)
    result = optimize_starting_lineup(roster_ids, opt_proj)

    current_starters = set(matchup_df[matchup_df["is_starter"]]["player_id"])
    optimized_starters = set(result["lineup"].values())
    to_start = optimized_starters - current_starters
    to_sit = current_starters - optimized_starters

    print(f"\n{'=' * 62}")
    print(f"  OPTIMAL LINEUP -- Week {args.week}")
    print(f"{'=' * 62}")
    for slot in sorted(result["lineup"]):
        pid = result["lineup"][slot]
        name = name_lookup.get(pid, pid)
        pos = pos_lookup.get(pid, "?")
        tag = injury_lookup.get(pid, "")
        dvp = dvp_lookup.get(pid, {})
        opp = dvp.get("opponent", "")
        rank = dvp.get("defense_rank", 16)
        mult = dvp.get("multiplier", 1.0)

        if opp and opp != "BYE":
            pct = (mult - 1.0) * 100
            if pct > 1:
                dvp_str = f" vs {opp} (#{rank} vs {pos}) [+{pct:.0f}%]"
            elif pct < -1:
                dvp_str = f" vs {opp} (#{rank} vs {pos}) [{pct:.0f}%]"
            else:
                dvp_str = f" vs {opp} (#{rank} vs {pos})"
        else:
            dvp_str = ""

        marker = "  <- START" if pid in to_start else ""
        print(f"  {slot:<6s} {pos:<4s} {name}{tag}{dvp_str}{marker}")

    if to_sit:
        print("\n  Sit:")
        for pid in to_sit:
            print(f"    {name_lookup.get(pid, pid)}")

    print(f"\n  Projected Points: {result['total_proj_points']:.1f}")

    # Monte Carlo win probability
    team_a_starters = opt_proj[opt_proj["player_id"].isin(result["lineup"].values())].copy()
    opp_roster_id = matchup_df["opponent_roster_id"].iloc[0] if not matchup_df.empty else None
    if opp_roster_id is not None:
        opp_matchup_df = extract_weekly_matchup_roster(opp_roster_id, matchups, player_pool)
        opp_roster_ids = opp_matchup_df[opp_matchup_df["is_starter"]]["player_id"].tolist()
        opp_proj = adj_proj[adj_proj["player_id"].isin(opp_roster_ids)][["player_id", "position", "adjusted_proj_points"]].copy()
        opp_proj.rename(columns={"adjusted_proj_points": "proj_points"}, inplace=True)
    else:
        opp_proj = pd.DataFrame(columns=["player_id", "position", "proj_points"])

    sim = simulate_team_matchup(team_a_starters, opp_proj, iterations=5000)
    print(f"  Win Prob: {sim['win_prob_a'] * 100:.1f}% | Proj: {sim['median_a']:.1f} "
          f"(10th: {sim['floor_a']:.1f}, 90th: {sim['ceiling_a']:.1f})")
    print(f"{'=' * 62}")

    # Starter health validation
    starter_rows = matchup_df[matchup_df["is_starter"]]
    current_lineup = dict(zip(starter_rows["player_name"], starter_rows["player_id"]))
    warnings = validate_starting_lineup_health(current_lineup, catalog)
    if warnings:
        print("\n  INJURY ALERTS:")
        for w in warnings:
            body = f" ({w['injury_body_part']})" if w["injury_body_part"] else ""
            print(f"   * {w['player_name']} [{w['injury_status']}]{body} at {w['slot']}")


# ---------------------------------------------------------------------------
# Handler: trade
# ---------------------------------------------------------------------------
def handle_trade(args) -> None:
    """Evaluate a two-way trade proposal."""
    print("Loading projections ...")
    board = _load_draft_board(refresh=args.refresh)
    catalog = _load_sleeper_catalog(refresh=args.refresh)

    print("Syncing league rosters ...")
    rosters = fetch_league_rosters(args.league_id)
    rosters_df = parse_rosters_dataframe(rosters)

    team_a_ids = rosters_df[rosters_df["roster_id"] == args.team_a_roster_id]["player_id"].tolist()
    team_b_ids = rosters_df[rosters_df["roster_id"] == args.team_b_roster_id]["player_id"].tolist()

    team_a_sends = [x.strip() for x in args.team_a_sends.split(",")]
    team_b_sends = [x.strip() for x in args.team_b_sends.split(",")]

    proj = board[["player_id", "player_name", "position_proj", "proj_points"]].copy()
    proj.rename(columns={"position_proj": "position"}, inplace=True)
    # Convert season-long (17-game) projections to weekly
    proj["proj_points"] = proj["proj_points"] / 17.0

    result = evaluate_trade(
        team_a_ids, team_b_ids, team_a_sends, team_b_sends, proj, board,
    )

    # Build injury lookup for both teams
    all_ids = list(set(team_a_ids + team_b_ids))
    injury_df = extract_roster_injury_status(all_ids, catalog)
    injury_lookup = (
        dict(zip(injury_df["player_id"], injury_df["injury_tag"]))
        if not injury_df.empty else {}
    )
    injury_detail = (
        dict(zip(injury_df["player_id"], injury_df["injury_body_part"]))
        if not injury_df.empty else {}
    )

    # Name lookup for display
    name_lookup = dict(zip(board["player_id"], board["player_name"]))

    print(f"\n{'=' * 60}")
    print("  TRADE EVALUATION")
    print(f"{'=' * 60}")
    for side_key, tag in [("team_a", "TEAM A"), ("team_b", "TEAM B")]:
        d = result[side_key]
        v = result[f"verdict_{side_key[-1]}"]
        print(f"\n  {tag}:")
        print(f"    Starting PPG Delta:  {d['starting_delta']:+.2f}")
        print(f"    ROS VORP Delta:      {d['vorp_delta']:+.2f}")
        print(f"    Verdict: {v}")

    # Injury summary for players involved in the trade
    trade_pids = team_a_sends + team_b_sends
    injured_in_trade = [
        pid for pid in trade_pids if injury_lookup.get(pid)
    ]
    if injured_in_trade:
        print("\n  Injury Status (traded players):")
        for pid in team_a_sends:
            tag = injury_lookup.get(pid, "")
            if tag:
                body = f" ({injury_detail.get(pid, '')})" if injury_detail.get(pid) else ""
                print(f"    Team A sends {name_lookup.get(pid, pid)} {tag}{body}")
        for pid in team_b_sends:
            tag = injury_lookup.get(pid, "")
            if tag:
                body = f" ({injury_detail.get(pid, '')})" if injury_detail.get(pid) else ""
                print(f"    Team B sends {name_lookup.get(pid, pid)} {tag}{body}")

    print(f"\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Handler: report
# ---------------------------------------------------------------------------
def handle_report(args) -> None:
    """Generate weekly digest and export to Markdown + CSV."""
    print("Loading projections ...")
    board = _load_draft_board(refresh=args.refresh)
    catalog = _load_sleeper_catalog(refresh=args.refresh)

    print("Syncing league rosters ...")
    rosters = fetch_league_rosters(args.league_id)
    rosters_df = parse_rosters_dataframe(rosters)

    print(f"Generating Week {args.week} digest for Roster {args.roster_id} ...")
    digest = generate_weekly_digest(
        league_id=args.league_id,
        week=args.week,
        roster_id=args.roster_id,
        rosters_df=rosters_df,
        draft_board_df=board,
        catalog_df=catalog,
    )

    # Export Markdown report
    md_path = os.path.join(
        args.output_dir,
        f"week_{args.week}_roster_{args.roster_id}.md",
    )
    written = export_digest_markdown(digest, md_path)
    print(f"  Report saved: {written}")

    # Export power rankings CSV
    power_df = digest.get("power_rankings", pd.DataFrame())
    if not power_df.empty:
        csv_path = os.path.join(
            args.output_dir,
            f"week_{args.week}_power_rankings.csv",
        )
        csv_written = export_power_rankings_csv(power_df, csv_path)
        print(f"  Power rankings saved: {csv_written}")

    print(f"\nDigest complete. {len(os.listdir(args.output_dir))} files in {args.output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
