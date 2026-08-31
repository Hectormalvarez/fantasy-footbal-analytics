"""Optional live integration tests for Sleeper API endpoints."""

import pytest

from src.league import (
    fetch_league,
    fetch_league_users,
    fetch_league_rosters,
    fetch_league_matchups,
    parse_users_dataframe,
    parse_rosters_dataframe,
    get_all_rostered_player_ids,
    extract_waiver_wire_pool,
)
from src.sleeper import fetch_sleeper_players, parse_sleeper_catalog

LEAGUE_ID = "1386423085304938496"
WEEK = 1


# ---------------------------------------------------------------------------
# Skip all network tests if offline
# ---------------------------------------------------------------------------
def _require_network():
    """Skip if Sleeper API is unreachable."""
    try:
        import requests
        requests.get("https://api.sleeper.app/v1/league/12345", timeout=5)
    except Exception:
        pytest.skip("Sleeper API unreachable (offline)")


@pytest.fixture(scope="module", autouse=True)
def _network_check():
    """Auto-skip all tests in this module if offline."""
    _require_network()


@pytest.mark.network
class TestLeagueMetadata:
    def test_fetch_league_returns_nonempty_dict(self):
        league = fetch_league(LEAGUE_ID)
        assert isinstance(league, dict)
        assert len(league) > 0

    def test_league_has_required_keys(self):
        league = fetch_league(LEAGUE_ID)
        for key in ("name", "season", "total_rosters", "status"):
            assert key in league, f"Missing key: {key}"

    def test_league_total_rosters_is_14(self):
        league = fetch_league(LEAGUE_ID)
        assert league["total_rosters"] == 14


@pytest.mark.network
class TestManagerMapping:
    def test_fetch_users_returns_list(self):
        users = fetch_league_users(LEAGUE_ID)
        assert isinstance(users, list)
        assert len(users) == 14

    def test_parse_users_dataframe_schema(self):
        users = fetch_league_users(LEAGUE_ID)
        df = parse_users_dataframe(users)
        assert "user_id" in df.columns
        assert "display_name" in df.columns
        assert "league_id" in df.columns

    def test_all_users_have_display_name(self):
        users = fetch_league_users(LEAGUE_ID)
        df = parse_users_dataframe(users)
        assert df["display_name"].notna().all()


@pytest.mark.network
class TestRosters:
    def test_fetch_rosters_returns_14(self):
        rosters = fetch_league_rosters(LEAGUE_ID)
        assert len(rosters) == 14

    def test_parse_rosters_dataframe_explodes(self):
        rosters = fetch_league_rosters(LEAGUE_ID)
        df = parse_rosters_dataframe(rosters)
        assert len(df) > 0
        assert "roster_id" in df.columns
        assert "player_id" in df.columns

    def test_all_rosters_have_players(self):
        rosters = fetch_league_rosters(LEAGUE_ID)
        for r in rosters:
            assert isinstance(r.get("players"), list)
            assert len(r["players"]) > 0, f"Roster {r['roster_id']} has no players"

    def test_rostered_player_ids_deduplicated(self):
        rosters = fetch_league_rosters(LEAGUE_ID)
        ids = get_all_rostered_player_ids(rosters)
        assert isinstance(ids, set)
        assert len(ids) > 100

    def test_roster_ids_unique(self):
        rosters = fetch_league_rosters(LEAGUE_ID)
        roster_ids = [r["roster_id"] for r in rosters]
        assert len(set(roster_ids)) == 14


@pytest.mark.network
class TestMatchups:
    def test_fetch_week1_matchups(self):
        matchups = fetch_league_matchups(LEAGUE_ID, WEEK)
        assert isinstance(matchups, list)
        assert len(matchups) == 14

    def test_matchup_structure(self):
        matchups = fetch_league_matchups(LEAGUE_ID, WEEK)
        for m in matchups:
            assert "roster_id" in m
            assert "matchup_id" in m
            assert "starters" in m
            assert isinstance(m["starters"], list)

    def test_matchup_ids_pair_correctly(self):
        matchups = fetch_league_matchups(LEAGUE_ID, WEEK)
        mid_counts = {}
        for m in matchups:
            mid = m["matchup_id"]
            mid_counts[mid] = mid_counts.get(mid, 0) + 1
        for mid, count in mid_counts.items():
            assert count == 2, f"Matchup {mid} has {count} rosters (expected 2)"


@pytest.mark.network
class TestCatalogResolution:
    def test_catalog_returns_dataframe(self):
        raw = fetch_sleeper_players()
        df = parse_sleeper_catalog(raw)
        assert len(df) > 0

    def test_catalog_has_required_columns(self):
        raw = fetch_sleeper_players()
        df = parse_sleeper_catalog(raw)
        for col in ("player_id", "player_name", "position"):
            assert col in df.columns, f"Missing column: {col}"

    def test_rostered_ids_exist_in_catalog(self):
        rosters = fetch_league_rosters(LEAGUE_ID)
        rostered = get_all_rostered_player_ids(rosters)
        raw = fetch_sleeper_players()
        # Check only skill-position players (QB/RB/WR/TE) against catalog.
        # K and D/ST are excluded: kickers have player_name=None in raw
        # data and D/ST use team abbreviations as player_ids.
        skill_positions = {"QB", "RB", "WR", "TE"}
        skill_rostered = {
            pid for pid in rostered
            if raw.get(pid, {}).get("position") in skill_positions
        }
        catalog_df = parse_sleeper_catalog(raw)
        catalog_ids = set(catalog_df["player_id"].astype(str))
        missing = skill_rostered - catalog_ids
        assert len(missing) == 0, f"Skill-position IDs not in catalog: {missing}"

    def test_waiver_pool_nonempty(self):
        rosters = fetch_league_rosters(LEAGUE_ID)
        rostered = get_all_rostered_player_ids(rosters)
        raw = fetch_sleeper_players()
        catalog_df = parse_sleeper_catalog(raw)
        all_ids = set(catalog_df["player_id"].astype(str))
        waiver_pool = extract_waiver_wire_pool(all_ids, rostered)
        assert len(waiver_pool) > 0, "Waiver pool is empty"
