"""Tests for src/league.py — Sleeper league API client and data transformers."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.league import (
    fetch_league,
    fetch_league_users,
    fetch_league_rosters,
    fetch_league_matchups,
    fetch_league_transactions,
    parse_users_dataframe,
    parse_rosters_dataframe,
    get_all_rostered_player_ids,
    SLEEPER_BASE_URL,
)


# ---------------------------------------------------------------------------
# fetch_league
# ---------------------------------------------------------------------------
FAKE_LEAGUE = {
    "league_id": "12345",
    "name": "Test League",
    "status": "in_season",
    "total_rosters": 12,
}


def test_fetch_league_returns_dict():
    """fetch_league returns a dict from the API."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_LEAGUE
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp) as m:
        result = fetch_league("12345")

    assert isinstance(result, dict)
    assert result["league_id"] == "12345"
    mock_resp.raise_for_status.assert_called_once()


def test_fetch_league_url():
    """fetch_league constructs the correct URL."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_LEAGUE
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp) as m:
        fetch_league("12345")

    m.assert_called_once()
    call_args = m.call_args
    assert call_args[0][0] == f"{SLEEPER_BASE_URL}/league/12345"


# ---------------------------------------------------------------------------
# fetch_league_users
# ---------------------------------------------------------------------------
FAKE_USERS = [
    {"user_id": "u1", "display_name": "Alice", "league_id": "12345"},
    {"user_id": "u2", "display_name": "Bob", "league_id": "12345"},
]


def test_fetch_league_users_returns_list():
    """fetch_league_users returns a list of user dicts."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_USERS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp):
        result = fetch_league_users("12345")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["display_name"] == "Alice"
    mock_resp.raise_for_status.assert_called_once()


def test_fetch_league_users_url():
    """fetch_league_users constructs the correct URL with /users suffix."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_USERS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp) as m:
        fetch_league_users("12345")

    assert m.call_args[0][0] == f"{SLEEPER_BASE_URL}/league/12345/users"


# ---------------------------------------------------------------------------
# fetch_league_rosters
# ---------------------------------------------------------------------------
FAKE_ROSTERS = [
    {"roster_id": 1, "owner_id": "u1", "players": ["p1", "p2"]},
    {"roster_id": 2, "owner_id": "u2", "players": ["p3"]},
]


def test_fetch_league_rosters_returns_list():
    """fetch_league_rosters returns a list of roster dicts."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_ROSTERS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp):
        result = fetch_league_rosters("12345")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["roster_id"] == 1
    mock_resp.raise_for_status.assert_called_once()


def test_fetch_league_rosters_url():
    """fetch_league_rosters constructs the correct URL with /rosters suffix."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_ROSTERS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp) as m:
        fetch_league_rosters("12345")

    assert m.call_args[0][0] == f"{SLEEPER_BASE_URL}/league/12345/rosters"


# ---------------------------------------------------------------------------
# fetch_league_matchups
# ---------------------------------------------------------------------------
FAKE_MATCHUPS = [
    {"roster_id": 1, "matchup_id": 1, "points": 120.5},
    {"roster_id": 2, "matchup_id": 1, "points": 95.3},
]


def test_fetch_league_matchups_returns_list():
    """fetch_league_matchups returns a list of matchup dicts."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_MATCHUPS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp):
        result = fetch_league_matchups("12345", week=1)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["points"] == 120.5


def test_fetch_league_matchups_url_includes_week():
    """fetch_league_matchups includes the week number in the URL."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_MATCHUPS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp) as m:
        fetch_league_matchups("12345", week=3)

    assert m.call_args[0][0] == f"{SLEEPER_BASE_URL}/league/12345/matchups/3"


# ---------------------------------------------------------------------------
# fetch_league_transactions
# ---------------------------------------------------------------------------
FAKE_TRANSACTIONS = [
    {"transaction_id": "t1", "type": "trade", "status": "complete"},
    {"transaction_id": "t2", "type": "waiver", "status": "complete"},
]


def test_fetch_league_transactions_returns_list():
    """fetch_league_transactions returns a list of transaction dicts."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_TRANSACTIONS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp):
        result = fetch_league_transactions("12345", round=1)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["type"] == "trade"


def test_fetch_league_transactions_url_includes_round():
    """fetch_league_transactions includes the round number in the URL."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_TRANSACTIONS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.league.requests.get", return_value=mock_resp) as m:
        fetch_league_transactions("12345", round=5)

    assert m.call_args[0][0] == f"{SLEEPER_BASE_URL}/league/12345/transactions/5"


# ---------------------------------------------------------------------------
# parse_users_dataframe
# ---------------------------------------------------------------------------
RAW_USERS = [
    {"user_id": "u1", "display_name": "Alice", "league_id": "12345", "avatar": "abc"},
    {"user_id": "u2", "display_name": "Bob", "league_id": "12345", "avatar": "def"},
    {"user_id": "u3", "display_name": "Charlie", "league_id": "12345", "avatar": "ghi"},
]


def test_parse_users_dataframe_columns():
    """parse_users_dataframe returns a DataFrame with expected columns."""
    df = parse_users_dataframe(RAW_USERS)
    assert isinstance(df, pd.DataFrame)
    assert "user_id" in df.columns
    assert "display_name" in df.columns
    assert "league_id" in df.columns


def test_parse_users_dataframe_row_count():
    """parse_users_dataframe preserves row count from input."""
    df = parse_users_dataframe(RAW_USERS)
    assert len(df) == 3


def test_parse_users_dataframe_values():
    """parse_users_dataframe preserves values from input."""
    df = parse_users_dataframe(RAW_USERS)
    assert df.iloc[0]["display_name"] == "Alice"
    assert df.iloc[2]["user_id"] == "u3"


def test_parse_users_dataframe_empty():
    """parse_users_dataframe handles empty input."""
    df = parse_users_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# parse_rosters_dataframe
# ---------------------------------------------------------------------------
RAW_ROSTERS = [
    {"roster_id": 1, "owner_id": "u1", "players": ["p1", "p2"]},
    {"roster_id": 2, "owner_id": "u2", "players": ["p3"]},
    {"roster_id": 3, "owner_id": "u3", "players": []},
]


def test_parse_rosters_dataframe_explodes_players():
    """parse_rosters_dataframe creates one row per (roster, player)."""
    df = parse_rosters_dataframe(RAW_ROSTERS)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 2 + 1 + 0


def test_parse_rosters_dataframe_columns():
    """parse_rosters_dataframe returns expected columns."""
    df = parse_rosters_dataframe(RAW_ROSTERS)
    assert "roster_id" in df.columns
    assert "owner_id" in df.columns
    assert "player_id" in df.columns


def test_parse_rosters_dataframe_values():
    """parse_rosters_dataframe preserves correct roster-player pairing."""
    df = parse_rosters_dataframe(RAW_ROSTERS)
    row = df[df["player_id"] == "p2"].iloc[0]
    assert row["roster_id"] == 1
    assert row["owner_id"] == "u1"


def test_parse_rosters_dataframe_empty():
    """parse_rosters_dataframe handles empty input."""
    df = parse_rosters_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# get_all_rostered_player_ids
# ---------------------------------------------------------------------------
def test_get_all_rostered_player_ids_returns_set():
    """get_all_rostered_player_ids returns a set of player_id strings."""
    result = get_all_rostered_player_ids(RAW_ROSTERS)
    assert isinstance(result, set)
    assert result == {"p1", "p2", "p3"}


def test_get_all_rostered_player_ids_empty():
    """get_all_rostered_player_ids returns empty set for empty input."""
    result = get_all_rostered_player_ids([])
    assert result == set()


def test_get_all_rostered_player_ids_deduplicates():
    """get_all_rostered_player_ids deduplicates across rosters."""
    duped = [
        {"roster_id": 1, "owner_id": "u1", "players": ["p1"]},
        {"roster_id": 2, "owner_id": "u2", "players": ["p1", "p2"]},
    ]
    result = get_all_rostered_player_ids(duped)
    assert result == {"p1", "p2"}
