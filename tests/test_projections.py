"""Tests for src/projections.py — historical stats ingestion and rookie imputation."""

import pandas as pd

from src.projections import extract_player_name, impute_missing_rookies


# ---------------------------------------------------------------------------
# extract_player_name tests
# ---------------------------------------------------------------------------
def test_extract_from_player_display_name():
    """player_display_name is preferred when present."""
    assert extract_player_name({"player_display_name": "Patrick Mahomes"}) == "Patrick Mahomes"


def test_extract_from_display_name():
    """Falls back to display_name if player_display_name is missing."""
    assert extract_player_name({"display_name": "Josh Allen"}) == "Josh Allen"


def test_extract_from_player_name():
    """Falls back to player_name if neither display column exists."""
    assert extract_player_name({"player_name": "L.Jackson"}) == "L.Jackson"


def test_extract_falsy_display_name():
    """Falsy player_display_name (None, empty string) triggers fallback."""
    assert extract_player_name({"player_display_name": None, "player_name": "J.Goff"}) == "J.Goff"
    assert extract_player_name({"player_display_name": "", "player_name": "J.Goff"}) == "J.Goff"


def test_extract_missing_all():
    """If no name columns exist, returns empty string."""


# ---------------------------------------------------------------------------
# impute_missing_rookies tests — synthetic data, no API calls
# ---------------------------------------------------------------------------
def _sleeper_catalog_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal Sleeper catalog DataFrame."""
    return pd.DataFrame(rows)


def _veterans_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal veterans projections DataFrame."""
    return pd.DataFrame(rows)


def test_impute_adds_rookies_with_good_rank():
    """A rookie with search_rank <= 250 gets an imputed projection."""
    veterans = _veterans_df([
        {"player_name": "Existing Vet", "position": "QB", "team": "KC", "proj_points": 300.0, "player_id": "1"},
    ])
    catalog = _sleeper_catalog_df([
        {"player_name": "New Rookie", "position": "QB", "team": "BUF",
         "search_rank": 10, "years_exp": 0, "player_id": "2", "status": "Active"},
    ])
    result = impute_missing_rookies(veterans, catalog)
    assert "New Rookie" in result["player_name"].tolist()


def test_impute_skips_low_rank_players():
    """A player with search_rank > 250 is NOT added."""
    veterans = _veterans_df([
        {"player_name": "Existing Vet", "position": "QB", "team": "KC", "proj_points": 300.0, "player_id": "1"},
    ])
    catalog = _sleeper_catalog_df([
        {"player_name": "Deep Bench", "position": "QB", "team": "BUF",
         "search_rank": 500, "years_exp": 0, "player_id": "2", "status": "Active"},
    ])
    result = impute_missing_rookies(veterans, catalog)
    assert "Deep Bench" not in result["player_name"].tolist()


def test_impute_does_not_duplicate_veterans():
    """Players already in the veterans set are NOT included in imputed output."""
    veterans = _veterans_df([
        {"player_name": "Josh Allen", "position": "QB", "team": "BUF",
         "proj_points": 360.0, "player_id": "1"},
    ])
    catalog = _sleeper_catalog_df([
        {"player_name": "Josh Allen", "position": "QB", "team": "BUF",
         "search_rank": 4, "years_exp": 8, "player_id": "1", "status": "Active"},
    ])
    result = impute_missing_rookies(veterans, catalog)
    # Josh Allen is already in veterans, so imputed output should be empty
    assert len(result) == 0


def test_impute_qb_projection_values():
    """QB imputation follows the expected curve."""
    catalog = _sleeper_catalog_df([
        {"player_name": "Elite QB", "position": "QB", "team": "KC",
         "search_rank": 3, "years_exp": 0, "player_id": "10", "status": "Active"},
        {"player_name": "Good QB",  "position": "QB", "team": "BUF",
         "search_rank": 8, "years_exp": 0, "player_id": "11", "status": "Active"},
        {"player_name": "Mid QB",   "position": "QB", "team": "NE",
         "search_rank": 14, "years_exp": 0, "player_id": "12", "status": "Active"},
        {"player_name": "Low QB",   "position": "QB", "team": "NYG",
         "search_rank": 25, "years_exp": 0, "player_id": "13", "status": "Active"},
    ])
    result = impute_missing_rookies(_veterans_df([]), catalog)
    by_name = result.set_index("player_name")
    assert by_name.loc["Elite QB", "proj_points"] == 360.0
    assert by_name.loc["Good QB", "proj_points"] == 320.0
    assert by_name.loc["Mid QB", "proj_points"] == 280.0
    assert by_name.loc["Low QB", "proj_points"] == 230.0


def test_impute_rb_projection_values():
    """RB imputation follows the expected curve."""
    catalog = _sleeper_catalog_df([
        {"player_name": "Elite RB", "position": "RB", "team": "ATL",
         "search_rank": 2, "years_exp": 0, "player_id": "20", "status": "Active"},
        {"player_name": "Good RB",  "position": "RB", "team": "DET",
         "search_rank": 8, "years_exp": 0, "player_id": "21", "status": "Active"},
        {"player_name": "Mid RB",   "position": "RB", "team": "IND",
         "search_rank": 25, "years_exp": 0, "player_id": "22", "status": "Active"},
    ])
    result = impute_missing_rookies(_veterans_df([]), catalog)
    by_name = result.set_index("player_name")
    assert by_name.loc["Elite RB", "proj_points"] == 290.0
    assert by_name.loc["Good RB", "proj_points"] == 265.0
    assert by_name.loc["Mid RB", "proj_points"] == 210.0
    assert extract_player_name({"position": "QB"}) == ""



def test_impute_wr_projection_values():
    """WR imputation follows the expected curve."""
    catalog = _sleeper_catalog_df([
        {"player_name": "Elite WR", "position": "WR", "team": "CIN",
         "search_rank": 4, "years_exp": 0, "player_id": "30", "status": "Active"},
        {"player_name": "Good WR",  "position": "WR", "team": "DAL",
         "search_rank": 15, "years_exp": 0, "player_id": "31", "status": "Active"},
    ])
    result = impute_missing_rookies(_veterans_df([]), catalog)
    by_name = result.set_index("player_name")
    assert by_name.loc["Elite WR", "proj_points"] == 310.0
    assert by_name.loc["Good WR", "proj_points"] == 250.0


def test_impute_te_projection_values():
    """TE imputation follows the expected curve."""
    catalog = _sleeper_catalog_df([
        {"player_name": "Elite TE", "position": "TE", "team": "LV",
         "search_rank": 3, "years_exp": 0, "player_id": "40", "status": "Active"},
        {"player_name": "Good TE",  "position": "TE", "team": "ARI",
         "search_rank": 12, "years_exp": 0, "player_id": "41", "status": "Active"},
    ])
    result = impute_missing_rookies(_veterans_df([]), catalog)
    by_name = result.set_index("player_name")
    assert by_name.loc["Elite TE", "proj_points"] == 230.0
    assert by_name.loc["Good TE", "proj_points"] == 170.0


def test_impute_output_has_canonical_columns():
    """Imputed output always has player_name, position, team, proj_points."""
    catalog = _sleeper_catalog_df([
        {"player_name": "New Guy", "position": "WR", "team": "SF",
         "search_rank": 20, "years_exp": 0, "player_id": "50", "status": "Active"},
    ])
    result = impute_missing_rookies(_veterans_df([]), catalog)
    assert set(result.columns) >= {"player_name", "position", "team", "proj_points"}


def test_impute_filters_to_active_only():
    """Inactive/retired players in catalog are not imputed."""
    catalog = _sleeper_catalog_df([
        {"player_name": "Retired Star", "position": "QB", "team": None,
         "search_rank": 10, "years_exp": 20, "player_id": "60", "status": "Retired"},
    ])
    result = impute_missing_rookies(_veterans_df([]), catalog)
    assert len(result) == 0