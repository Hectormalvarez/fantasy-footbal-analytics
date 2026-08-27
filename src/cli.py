"""CLI entry point for the fantasy football draft analytics pipeline.

Usage::

    python -m src.cli --slot 14
    python -m src.cli --slot 1 --refresh --rounds 10
"""

import argparse
import os
import sys

import pandas as pd

from src.sleeper import fetch_sleeper_players, parse_sleeper_catalog, clean_player_name
from src.vorp import build_draft_board
from src.draft import generate_contingency_sheet


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="fantasy-draft",
        description="Fantasy football draft analytics engine — VORP board + contingency sheet",
    )
    parser.add_argument(
        "--slot", type=int, required=True,
        help="Assigned draft slot (1–14).",
    )
    parser.add_argument(
        "--refresh", action="store_true", default=False,
        help="Force a fresh Sleeper API download.",
    )
    parser.add_argument(
        "--rounds", type=int, default=15,
        help="Draft depth in rounds (default: 15).",
    )
    parser.add_argument(
        "--reach-buffer", type=int, default=4,
        dest="reach_buffer",
        help="Max picks to reach ahead of ADP (default: 4).",
    )
    parser.add_argument(
        "--fall-buffer", type=int, default=8,
        dest="fall_buffer",
        help="Max picks to catch falling value (default: 8).",
    )
    return parser


def _load_draft_board(
    projections_path: str = "data/projections.csv",
    sleeper_cache: str = "data/sleeper_players_raw.json",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load projections + Sleeper catalog and build the VORP draft board."""
    projections = pd.read_csv(projections_path)

    raw_sleeper = fetch_sleeper_players(cache_path=sleeper_cache, force_refresh=refresh)
    catalog = parse_sleeper_catalog(raw_sleeper)

    catalog_rank = catalog[["player_name", "position", "search_rank", "team"]].copy()
    catalog_rank["norm_name"] = catalog_rank["player_name"].apply(clean_player_name)
    projections["norm_name"] = projections["player_name"].apply(clean_player_name)

    merged = projections.merge(
        catalog_rank[["norm_name", "position", "search_rank", "team"]],
        on=["norm_name", "position"], how="left",
        suffixes=("", "_sleeper"),
    )

    merged["search_rank"] = merged["search_rank"].fillna(9999)
    merged["team"] = merged.get("team_sleeper", merged.get("team", "")).fillna("")
    if "team_sleeper" in merged.columns:
        merged.drop(columns=["team_sleeper"], inplace=True, errors="ignore")

    merged.rename(columns={"position": "position_proj"}, inplace=True)

    baselines = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = merged[merged["position_proj"] == pos].sort_values(
            "proj_points", ascending=False
        )
        baseline_idx = min(11, len(pos_df) - 1)
        baselines[pos] = pos_df.iloc[baseline_idx]["proj_points"] if len(pos_df) > 0 else 0.0

    return build_draft_board(merged, baselines)



def _format_contingency_sheet(sheet: pd.DataFrame, slot: int, num_teams: int = 14) -> str:
    """Format a contingency sheet DataFrame as a human-readable text block."""
    lines = []
    header = (
        f"{'=' * 72}\n"
        f"  DRAFT CONTINGENCY SHEET  —  Slot {slot} of {num_teams}\n"
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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

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

    with open(txt_path, "w") as f:
        f.write(formatted)
    print(f"Exported: {txt_path}")


if __name__ == "__main__":
    main()