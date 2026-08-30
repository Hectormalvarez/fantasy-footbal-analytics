"""Sleeper League API client and data transformers."""

import pandas as pd
import requests

SLEEPER_BASE_URL = "https://api.sleeper.app/v1"


def fetch_league(league_id: str) -> dict:
    """Fetch league metadata from the Sleeper API."""
    resp = requests.get(f"{SLEEPER_BASE_URL}/league/{league_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_league_users(league_id: str) -> list[dict]:
    """Fetch all users in a league."""
    resp = requests.get(f"{SLEEPER_BASE_URL}/league/{league_id}/users", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_league_rosters(league_id: str) -> list[dict]:
    """Fetch all rosters in a league."""
    resp = requests.get(f"{SLEEPER_BASE_URL}/league/{league_id}/rosters", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_league_matchups(league_id: str, week: int) -> list[dict]:
    """Fetch matchups for a given week."""
    resp = requests.get(
        f"{SLEEPER_BASE_URL}/league/{league_id}/matchups/{week}", timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def fetch_league_transactions(league_id: str, round: int) -> list[dict]:
    """Fetch transactions for a given round."""
    resp = requests.get(
        f"{SLEEPER_BASE_URL}/league/{league_id}/transactions/{round}", timeout=30
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Data transformers
# ---------------------------------------------------------------------------
_USER_COLUMNS = ["user_id", "display_name", "league_id"]


def parse_users_dataframe(users: list[dict]) -> pd.DataFrame:
    """Parse raw Sleeper user dicts into a DataFrame."""
    if not users:
        return pd.DataFrame(columns=_USER_COLUMNS)
    return pd.DataFrame(users)[_USER_COLUMNS]


_ROSTER_COLUMNS = ["roster_id", "owner_id", "player_id"]


def parse_rosters_dataframe(rosters: list[dict]) -> pd.DataFrame:
    """Parse raw Sleeper roster dicts into an exploded DataFrame.

    Each roster's ``players`` list is exploded so that one row equals
    one ``(roster_id, owner_id, player_id)`` triple.
    """
    rows: list[dict] = []
    for roster in rosters:
        for player_id in roster.get("players") or []:
            rows.append({
                "roster_id": roster["roster_id"],
                "owner_id": roster["owner_id"],
                "player_id": player_id,
            })
    if not rows:
        return pd.DataFrame(columns=_ROSTER_COLUMNS)
    return pd.DataFrame(rows)


def get_all_rostered_player_ids(rosters: list[dict]) -> set[str]:
    """Return the union of all player IDs across every roster."""
    ids: set[str] = set()
    for roster in rosters:
        ids.update(roster.get("players") or [])
    return ids
