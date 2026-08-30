"""Tests for src/league.py — Sleeper league API client and data transformers."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.league import fetch_league, fetch_league_users, SLEEPER_BASE_URL


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
