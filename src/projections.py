"""Historical stats ingestion, rookie imputation, and full projection export."""

import os

import pandas as pd
import nflreadpy as nfl

from src.scoring import calculate_ppr_points, project_17_game_ppr
from src.sleeper import fetch_sleeper_players, parse_sleeper_catalog

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def extract_player_name(row: dict) -> str:
    """Extract the best available full player name from a stat row.

    Priority: player_display_name > display_name > player_name.
    """
    for key in ("player_display_name", "display_name", "player_name"):
        val = row.get(key)
        if val:
            return val
    return ""


# ---------------------------------------------------------------------------
# Imputation curve  (position + search_rank -> projected PPR points)
# ---------------------------------------------------------------------------
_IMPUTE_CURVE: dict[str, list[tuple[int, float]]] = {
    "QB": [(5, 360), (10, 320), (15, 280), (20, 260), (30, 230)],
    "RB": [(5, 290), (10, 265), (20, 235), (30, 210), (40, 185), (60, 160)],
    "WR": [(5, 310), (10, 285), (20, 250), (30, 220), (40, 195), (60, 170)],
    "TE": [(5, 230), (10, 195), (15, 170), (20, 150), (30, 130)],
}
_IMPUTE_DEFAULTS = {"QB": 200.0, "RB": 140.0, "WR": 145.0, "TE": 110.0}


def _impute_projection(position: str, search_rank: float) -> float:
    """Return an imputed 17-game Full-PPR projection for a player."""
    thresholds = _IMPUTE_CURVE.get(position, [])
    for max_rank, pts in thresholds:
        if search_rank <= max_rank:
            return float(pts)
    return _IMPUTE_DEFAULTS.get(position, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_historical_stats(season: int = 2024) -> pd.DataFrame:
    """Pull season-level stats from nflreadpy and convert to Pandas.

    Returns a DataFrame filtered to skill positions with columns needed for
    PPR scoring (passing_yards, rushing_yards, receptions, etc.).
    """
    raw = nfl.load_player_stats(season, summary_level="reg")

    # Polars -> Pandas via rows (avoids pyarrow)
    records = raw.rows(named=True)
    df = pd.DataFrame(records)

    # Filter to skill positions
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()

    # Extract canonical full name
    df["player_name"] = df.apply(
        lambda r: extract_player_name(r.to_dict() if hasattr(r, "to_dict") else dict(r)),
        axis=1,
    )

    # Rename team column for downstream consistency
    if "recent_team" in df.columns:
        df.rename(columns={"recent_team": "team"}, inplace=True)

    return df


def impute_missing_rookies(
    projections_df: pd.DataFrame,
    sleeper_catalog: pd.DataFrame,
    rank_threshold: int = 250,
) -> pd.DataFrame:
    """Impute projections for players in the Sleeper catalog but not in the veteran set.

    Parameters
    ----------
    projections_df : DataFrame with player_id, player_name, position, team, proj_points.
    sleeper_catalog : Parsed Sleeper catalog with player_id, player_name, position, etc.
    rank_threshold : Only impute players with search_rank <= this value.

    Returns
    -------
    DataFrame with canonical columns: player_name, position, team, proj_points.
    """
    if projections_df.empty or "player_id" not in projections_df.columns:
        existing_ids = set()
    else:
        existing_ids = set(projections_df["player_id"].astype(str).unique())

    # Also use player_name for dedup if available
    if not projections_df.empty and "player_name" in projections_df.columns:
        existing_names = set(projections_df["player_name"].astype(str).unique())
    else:
        existing_names = set()

    # Filter catalog to high-profile active players not already in veteran set
    candidates = sleeper_catalog.copy()
    candidates["player_id_str"] = candidates["player_id"].astype(str)
    # Normalize names for matching
    from src.sleeper import clean_player_name
    candidates["norm_name"] = candidates["player_name"].apply(clean_player_name)
    existing_norm = {clean_player_name(n) for n in existing_names}

    candidates = candidates[
        (~candidates["player_id_str"].isin(existing_ids))
        & (~candidates["norm_name"].isin(existing_norm))
        & (candidates["search_rank"].astype(float) <= rank_threshold)
        & (candidates["status"] == "Active")
    ].copy()

    if candidates.empty:
        return pd.DataFrame(columns=["player_name", "position", "team", "proj_points"])

    candidates["proj_points"] = candidates.apply(
        lambda r: _impute_projection(r["position"], float(r["search_rank"])),
        axis=1,
    )

    return candidates[["player_name", "position", "team", "proj_points"]].reset_index(drop=True)


def generate_full_projections(
    output_path: str = "data/projections.csv",
    sleeper_cache: str = "data/sleeper_players_raw.json",
    season: int = 2024,
) -> pd.DataFrame:
    """End-to-end: ingest historical stats, impute rookies, merge, and export.

    Returns the final canonical DataFrame with columns:
    ['player_name', 'position', 'team', 'proj_points']
    """
    # 1. Veteran projections from nflreadpy
    stats = load_historical_stats(season)
    stats = stats[stats["games"] >= 3].copy()
    stats["total_ppr"] = calculate_ppr_points(stats)
    stats["proj_points"] = project_17_game_ppr(stats)

    veterans = stats[["player_name", "position", "team", "proj_points",
                       "player_id"]].copy()

    # 2. Impute missing rookies / profiled players from Sleeper
    sleeper_raw = fetch_sleeper_players(cache_path=sleeper_cache)
    catalog = parse_sleeper_catalog(sleeper_raw)

    imputed = impute_missing_rookies(veterans, catalog)

    # 3. Combine, deduplicate (keep veteran stats), export
    combined = pd.concat([veterans, imputed], ignore_index=True)
    combined = combined.sort_values("proj_points", ascending=False)
    combined = combined.drop_duplicates(subset=["player_name", "position"], keep="first")
    combined = combined.sort_values("proj_points", ascending=False).reset_index(drop=True)

    output = combined[["player_name", "position", "team", "proj_points"]].copy()
    output["proj_points"] = output["proj_points"].round(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output.to_csv(output_path, index=False)

    return output
