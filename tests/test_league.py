"""Tests for src/league.py — Sleeper league API client and data transformers."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.league import fetch_league, SLEEPER_BASE_URL


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
