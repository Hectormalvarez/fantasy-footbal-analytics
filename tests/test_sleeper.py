"""Tests for src/sleeper.py — Sleeper API, catalog parsing, name normalization."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from src.sleeper import clean_player_name, fetch_sleeper_players, parse_sleeper_catalog


def test_clean_lowercase():
    """Names are lowercased and stripped."""
    assert clean_player_name(" Josh Allen ") == "josh allen"


def test_clean_suffix_jr():
    """Jr. suffix is stripped."""
    assert clean_player_name("Patrick Mahomes Jr.") == "patrick mahomes"


def test_clean_suffix_sr():
    """Sr. suffix is stripped."""
    assert clean_player_name("Arnold Jackson Sr.") == "arnold jackson"


def test_clean_suffix_iii():
    """III suffix is stripped."""
    assert clean_player_name("Odell Beckham III") == "odell beckham"


def test_clean_suffix_iv():
    """IV suffix is stripped."""
    assert clean_player_name("Emmitt Smith IV") == "emmitt smith"


def test_clean_apostrophe():
    """Apostrophes are removed."""
    assert clean_player_name("Ja'Marr Chase") == "jamarr chase"


def test_clean_hyphenated():
    """Hyphens are preserved."""
    assert clean_player_name("Amon-Ra St. Brown") == "amon-ra st. brown"


def test_clean_multi_space():
    """Multiple spaces collapse to one."""
    assert clean_player_name("Kyler   Murray") == "kyler murray"



# ---------------------------------------------------------------------------
# fetch_sleeper_players tests
# ---------------------------------------------------------------------------
FAKE_PLAYERS = {"123": {"full_name": "Test Player", "position": "QB"}}


def test_fetch_returns_dict():
    """fetch_sleeper_players returns a dict."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = FAKE_PLAYERS
    mock_resp.raise_for_status = MagicMock()

    with patch("src.sleeper.requests.get", return_value=mock_resp):
        result = fetch_sleeper_players(cache_path="/tmp/fake_cache.json")
    assert isinstance(result, dict)
    assert "123" in result


def test_fetch_caches_to_disk():
    """On a fresh fetch, the result is written to cache_path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "test_cache.json")

        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_PLAYERS
        mock_resp.raise_for_status = MagicMock()

        with patch("src.sleeper.requests.get", return_value=mock_resp):
            fetch_sleeper_players(cache_path=cache)

        assert os.path.exists(cache)


def test_fetch_loads_from_cache():
    """If cache exists and force_refresh=False, API is NOT called."""
    import json as _json
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "test_cache.json")
        with open(cache, "w") as f:
            _json.dump(FAKE_PLAYERS, f)

        with patch("src.sleeper.requests.get") as m:
            result = fetch_sleeper_players(cache_path=cache, force_refresh=False)

        m.assert_not_called()
        assert result == FAKE_PLAYERS


def test_fetch_force_refresh_ignores_cache():
    """force_refresh=True always hits the API."""
    import json as _json
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "test_cache.json")
        with open(cache, "w") as f:
            _json.dump({"old": True}, f)

        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_PLAYERS
        mock_resp.raise_for_status = MagicMock()

        with patch("src.sleeper.requests.get", return_value=mock_resp) as m:
            result = fetch_sleeper_players(cache_path=cache, force_refresh=True)

        m.assert_called_once()
        assert "123" in result


# ---------------------------------------------------------------------------
# parse_sleeper_catalog tests
# ---------------------------------------------------------------------------
RAW_CATALOG = {
    "1": {
        "full_name": "Patrick Mahomes",
        "position": "QB",
        "team": "KC",
        "search_rank": 2,
        "years_exp": 7,
        "status": "Active",
        "player_id": "1",
    },
    "2": {
        "full_name": "Tyreek Hill",
        "position": "WR",
        "team": "MIA",
        "search_rank": 5,
        "years_exp": 8,
        "status": "Active",
        "player_id": "2",
    },
    "3": {
        "full_name": "Retired Guy",
        "position": "QB",
        "team": None,
        "search_rank": 99999,
        "years_exp": 20,
        "status": "Retired",
        "player_id": "3",
    },
}


def test_parse_returns_dataframe():
    """parse_sleeper_catalog returns a DataFrame."""
    df = parse_sleeper_catalog(RAW_CATALOG)
    assert isinstance(df, pd.DataFrame)


def test_parse_filters_by_position():
    """Only skill positions are returned."""
    df = parse_sleeper_catalog(RAW_CATALOG, positions=["QB", "WR"])
    assert len(df) == 2
    assert set(df["position"]) == {"QB", "WR"}


def test_parse_renames_full_name():
    """full_name is renamed to player_name for downstream consistency."""
    df = parse_sleeper_catalog(RAW_CATALOG, positions=["QB"])
    assert "player_name" in df.columns
    assert df.iloc[0]["player_name"] == "Patrick Mahomes"


def test_parse_adds_norm_columns():
    """norm_name and norm_position columns are added."""
    df = parse_sleeper_catalog(RAW_CATALOG, positions=["QB"])
    assert "norm_name" in df.columns
    assert df.iloc[0]["norm_name"] == "patrick mahomes"
    assert df.iloc[0]["norm_position"] == "QB"


def test_parse_filters_status():
    """Only Active players are returned by default."""
    df = parse_sleeper_catalog(RAW_CATALOG)
    assert "Retired" not in df["status"].values