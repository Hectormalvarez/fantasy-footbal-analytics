"""Live Sleeper API data pull and schema verification for league 1386423085304938496."""

import sys
from typing import Any

import pandas as pd

from src.league import (
    fetch_league,
    fetch_league_users,
    fetch_league_rosters,
    fetch_league_matchups,
    parse_users_dataframe,
    parse_rosters_dataframe,
    get_all_rostered_player_ids,
)
from src.sleeper import (
    fetch_sleeper_players,
    parse_sleeper_catalog,
)

LEAGUE_ID = "1386423085304938496"
WEEK = 1


# ---------------------------------------------------------------------------
# Section 1 -- League Metadata
# ---------------------------------------------------------------------------
def verify_league_metadata(league_id: str) -> dict[str, Any]:
    """Fetch and validate league metadata."""
    league = fetch_league(league_id)
    assert "name" in league, "Missing league name"
    assert "season" in league, "Missing season"
    assert "total_rosters" in league, "Missing total_rosters"
    assert "status" in league, "Missing status"
    return league


# ---------------------------------------------------------------------------
# Section 2 -- Manager Mapping
# ---------------------------------------------------------------------------
def verify_manager_mapping(league_id: str) -> pd.DataFrame:
    """Fetch users and build manager mapping."""
    users = fetch_league_users(league_id)
    assert len(users) > 0, "No users returned"
    df = parse_users_dataframe(users)
    assert "user_id" in df.columns, "Missing user_id column"
    assert "display_name" in df.columns, "Missing display_name column"
    return df


# ---------------------------------------------------------------------------
# Section 3 -- Rosters & Ownership
# ---------------------------------------------------------------------------
def verify_rosters(
    league_id: str,
) -> tuple[list[dict], pd.DataFrame, set[str]]:
    """Fetch rosters, build exploded DataFrame, collect all rostered IDs."""
    rosters = fetch_league_rosters(league_id)
    assert len(rosters) > 0, "No rosters returned"
    df = parse_rosters_dataframe(rosters)
    assert "roster_id" in df.columns, "Missing roster_id"
    assert "player_id" in df.columns, "Missing player_id"
    all_ids = get_all_rostered_player_ids(rosters)
    assert len(all_ids) > 0, "No rostered player IDs found"
    return rosters, df, all_ids


# ---------------------------------------------------------------------------
# Section 4 -- Weekly Matchups
# ---------------------------------------------------------------------------
def verify_weekly_matchups(
    league_id: str, week: int
) -> list[dict]:
    """Fetch matchups for a given week and validate structure."""
    matchups = fetch_league_matchups(league_id, week)
    assert len(matchups) > 0, f"No matchups for week {week}"
    for m in matchups:
        assert "roster_id" in m, "Matchup missing roster_id"
        assert "matchup_id" in m, "Matchup missing matchup_id"
        assert "starters" in m, f"Roster {m.get('roster_id')} missing starters"
        assert isinstance(m["starters"], list), f"Roster {m.get('roster_id')} starters not a list"
    return matchups


# ---------------------------------------------------------------------------
# Section 5 -- Catalog Resolution
# ---------------------------------------------------------------------------
def verify_catalog_resolution(
    catalog_df: pd.DataFrame,
    rosters: list[dict],
    users_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Resolve starter names for all rosters via the player catalog."""
    cat_lookup = {}
    for _, row in catalog_df.iterrows():
        cat_lookup[str(row["player_id"])] = {
            "player_name": row.get("player_name", "Unknown"),
            "position": row.get("position", "?") ,
        }

    owner_lookup = dict(zip(users_df["user_id"], users_df["display_name"]))

    roster_details = []
    for roster in rosters:
        rid = roster["roster_id"]
        owner_id = roster.get("owner_id", "")
        manager = owner_lookup.get(owner_id, owner_id)
        team_name = roster.get("settings", {}).get("name") or manager
        players = roster.get("players") or []
        starters = roster.get("starters") or []
        starter_names = []
        for pid in starters:
            if pid in cat_lookup:
                info = cat_lookup[pid]
                starter_names.append(f"{info['player_name']} ({info['position']})")
            else:
                starter_names.append(f"Unknown ({pid})")
        roster_details.append({
            "roster_id": rid,
            "manager": manager,
            "team_name": team_name,
            "total_players": len(players),
            "starters_count": len([s for s in starters if s]),
            "starter_names": starter_names,
        })

    roster_details.sort(key=lambda x: x["roster_id"])
    return roster_details


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------
def print_league_overview(league: dict[str, Any]) -> None:
    """Print league overview summary."""
    print()
    print("=" * 70)
    print("  SLEEPER LEAGUE VERIFICATION")
    print("=" * 70)
    print()
    print(f"  League Name:    {league['name']}")
    print(f"  Season:         {league['season']}")
    print(f"  Status:         {league['status']}")
    print(f"  Total Rosters:  {league['total_rosters']}")
    print(f"  Scoring:        {league.get('scoring_settings', {}).get('rec', 'standard')}")
    print()


def print_roster_table(roster_details: list[dict[str, Any]]) -> None:
    """Print manager/roster table."""
    print("-" * 70)
    print(f"  {'Roster ID':<11s} {'Manager':<22s} {'Team Name':<22s} {'Players':<9s} {'Starters':<8s}")
    print("-" * 70)
    for rd in roster_details:
        print(f"  {rd['roster_id']:<11d} {rd['manager']:<22s} {rd['team_name']:<22s} {rd['total_players']:<9d} {rd['starters_count']:<8d}")
    print("-" * 70)
    print()


def print_matchup_summary(matchups: list[dict]) -> None:
    """Print matchup summary for week."""
    print("-" * 70)
    print("  WEEK 1 MATCHUPS")
    print("-" * 70)
    pairs: dict[int, list[dict]] = {}
    for m in matchups:
        mid = m["matchup_id"]
        pairs.setdefault(mid, []).append(m)
    for mid, pair in sorted(pairs.items()):
        if len(pair) == 2:
            r1, r2 = pair[0], pair[1]
            print(f"  Match {mid}: Roster {r1['roster_id']:<3d} vs Roster {r2['roster_id']:<3d}")
        else:
            print(f"  Match {mid}: Roster {pair[0]['roster_id']} (bye)")
    print("-" * 70)
    print()


def print_starter_names(roster_details: list[dict[str, Any]]) -> None:
    """Print resolved starter names for each roster."""
    print("-" * 70)
    print("  STARTER NAMES (resolved via catalog)")
    print("-" * 70)
    for rd in roster_details:
        print(f"  Roster {rd['roster_id']} ({rd['manager']}):")
        for i, name in enumerate(rd["starter_names"], 1):
            print(f"    {i:>2d}. {name}")
        print()
    print("-" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Connecting to Sleeper API...")
    print()

    # 1. League metadata
    league = verify_league_metadata(LEAGUE_ID)
    print_league_overview(league)

    # 2. Manager mapping
    users_df = verify_manager_mapping(LEAGUE_ID)
    print(f"  Manager mapping: {len(users_df)} users loaded")

    # 3. Rosters
    rosters, rosters_df, all_rostered_ids = verify_rosters(LEAGUE_ID)

    # 4. Catalog
    raw_catalog = fetch_sleeper_players()
    catalog_df = parse_sleeper_catalog(raw_catalog)
    all_catalog_ids = set(catalog_df["player_id"].astype(str))
    waiver_pool = all_catalog_ids - all_rostered_ids

    # 5. Matchups
    matchups = verify_weekly_matchups(LEAGUE_ID, WEEK)
    print_matchup_summary(matchups)

    # 6. Catalog resolution
    roster_details = verify_catalog_resolution(catalog_df, rosters, users_df)
    print_roster_table(roster_details)
    print_starter_names(roster_details)

    # Summary
    print("=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  League:              {league['name']} (Season {league['season']})")
    print(f"  Rosters:             {league['total_rosters']}")
    print(f"  Total Rostered IDs:  {len(all_rostered_ids)}")
    print(f"  Catalog Players:     {len(all_catalog_ids)}")
    print(f"  Waiver Pool:         {len(waiver_pool)} players")
    print(f"  Week {WEEK} Matchups:     {len(matchups)} rosters")
    print("=" * 70)
    print()
    print("All verifications passed.")


if __name__ == "__main__":
    main()
