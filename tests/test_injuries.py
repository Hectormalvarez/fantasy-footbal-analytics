"""Tests for src/injuries.py -- injury status tracking and IR stash optimizer."""

import pandas as pd
import pytest

from src.injuries import (
    extract_roster_injury_status,
    validate_starting_lineup_health,
    identify_ir_eligible_stashes,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _catalog_df():
    """Return a sample Sleeper catalog DataFrame with injury fields."""
    return pd.DataFrame({
        "player_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "player_name": [
            "Jalen Hurts", "Saquon Barkley", "AJ Brown",
            "Dallas Goedert", "DeVonta Smith", "Jaylen Waddle",
        ],
        "position": ["QB", "RB", "WR", "TE", "WR", "WR"],
        "injury_status": [None, "Questionable", "Out", "Doubtful", "IR", None],
        "injury_body_part": [None, "Knee", "Hamstring", "Ankle", "Foot", None],
        "injury_notes": [None, None, "DNP Friday", None, None, None],
    })


# ---------------------------------------------------------------------------
# extract_roster_injury_status
# ---------------------------------------------------------------------------
def test_extract_returns_dataframe():
    """extract_roster_injury_status returns a DataFrame."""
    result = extract_roster_injury_status(["p1", "p2"], _catalog_df())
    assert isinstance(result, pd.DataFrame)


def test_extract_columns():
    """Result contains expected columns."""
    result = extract_roster_injury_status(["p1"], _catalog_df())
    expected = {
        "player_id", "player_name", "injury_status",
        "injury_body_part", "injury_notes", "injury_tag",
    }
    assert set(result.columns) == expected


def test_extract_healthy_player():
    """Healthy player has empty injury_tag."""
    result = extract_roster_injury_status(["p1"], _catalog_df())
    assert result.iloc[0]["injury_tag"] == ""
    assert pd.isna(result.iloc[0]["injury_status"])


def test_extract_questionable():
    """Questionable player gets [Q] tag."""
    result = extract_roster_injury_status(["p2"], _catalog_df())
    assert result.iloc[0]["injury_tag"] == "[Q]"
    assert result.iloc[0]["injury_status"] == "Questionable"
    assert result.iloc[0]["injury_body_part"] == "Knee"


def test_extract_out():
    """Out player gets [OUT] tag."""
    result = extract_roster_injury_status(["p3"], _catalog_df())
    assert result.iloc[0]["injury_tag"] == "[OUT]"


def test_extract_doubtful():
    """Doubtful player gets [D] tag."""
    result = extract_roster_injury_status(["p4"], _catalog_df())
    assert result.iloc[0]["injury_tag"] == "[D]"


def test_extract_ir():
    """IR player gets [IR] tag."""
    result = extract_roster_injury_status(["p5"], _catalog_df())
    assert result.iloc[0]["injury_tag"] == "[IR]"


def test_extract_multiple_players():
    """Multiple players are extracted correctly."""
    result = extract_roster_injury_status(["p1", "p3", "p5"], _catalog_df())
    assert len(result) == 3
    tags = dict(zip(result["player_id"], result["injury_tag"]))
    assert tags["p1"] == ""
    assert tags["p3"] == "[OUT]"
    assert tags["p5"] == "[IR]"


def test_extract_empty_roster():
    """Empty roster list returns empty DataFrame."""
    result = extract_roster_injury_status([], _catalog_df())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_extract_empty_catalog():
    """Empty catalog returns empty DataFrame."""
    result = extract_roster_injury_status(["p1"], pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_extract_no_match():
    """Player IDs not in catalog return empty DataFrame."""
    result = extract_roster_injury_status(["unknown1", "unknown2"], _catalog_df())
    assert len(result) == 0


def test_extract_injury_notes():
    """Injury notes are preserved."""
    result = extract_roster_injury_status(["p3"], _catalog_df())
    assert result.iloc[0]["injury_notes"] == "DNP Friday"


def test_extract_preserves_player_name():
    """Player name is carried through from catalog."""
    result = extract_roster_injury_status(["p2"], _catalog_df())
    assert result.iloc[0]["player_name"] == "Saquon Barkley"


# ---------------------------------------------------------------------------
# validate_starting_lineup_health
# ---------------------------------------------------------------------------
def test_validate_out_starters_high_severity():
    """Out starters produce high-severity warnings."""
    lineup = {"WR1": "p3", "QB": "p1"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert len(warnings) == 1
    assert warnings[0]["player_name"] == "AJ Brown"
    assert warnings[0]["severity"] == "high"
    assert warnings[0]["injury_status"] == "Out"
    assert warnings[0]["slot"] == "WR1"


def test_validate_ir_starters_high_severity():
    """IR starters produce high-severity warnings."""
    lineup = {"WR2": "p5", "QB": "p1"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert len(warnings) == 1
    assert warnings[0]["severity"] == "high"
    assert warnings[0]["injury_status"] == "IR"


def test_validate_doubtful_starters_medium_severity():
    """Doubtful starters produce medium-severity warnings."""
    lineup = {"TE": "p4", "QB": "p1"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert len(warnings) == 1
    assert warnings[0]["severity"] == "medium"
    assert warnings[0]["injury_status"] == "Doubtful"


def test_validate_questionable_no_warning():
    """Questionable starters do NOT produce warnings (playable)."""
    lineup = {"RB1": "p2", "QB": "p1"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert len(warnings) == 0


def test_validate_healthy_starters():
    """All-healthy lineup produces no warnings."""
    lineup = {"QB": "p1", "WR1": "p6"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert len(warnings) == 0


def test_validate_multiple_injured():
    """Multiple injured starters produce multiple warnings."""
    lineup = {"WR1": "p3", "WR2": "p5", "TE": "p4"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert len(warnings) == 3
    severities = {w["severity"] for w in warnings}
    assert "high" in severities
    assert "medium" in severities


def test_validate_empty_lineup():
    """Empty lineup produces no warnings."""
    warnings = validate_starting_lineup_health({}, _catalog_df())
    assert warnings == []


def test_validate_empty_catalog():
    """Empty catalog produces no warnings."""
    warnings = validate_starting_lineup_health({"QB": "p1"}, pd.DataFrame())
    assert warnings == []


def test_validate_unknown_player():
    """Unknown player_id in lineup is silently skipped."""
    lineup = {"QB": "unknown", "WR1": "p3"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert len(warnings) == 1
    assert warnings[0]["player_name"] == "AJ Brown"


def test_validate_injury_body_part_in_warning():
    """Warning includes injury body part."""
    lineup = {"WR1": "p3"}
    warnings = validate_starting_lineup_health(lineup, _catalog_df())
    assert warnings[0]["injury_body_part"] == "Hamstring"


# ---------------------------------------------------------------------------
# identify_ir_eligible_stashes
# ---------------------------------------------------------------------------
def test_identify_ir_returns_list():
    """identify_ir_eligible_stashes returns a list."""
    roster = {"starters": ["p1", "p2"], "bench": ["p3", "p4", "p5"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert isinstance(result, list)


def test_identify_ir_out_player():
    """Out bench player is IR-eligible."""
    roster = {"starters": ["p1"], "bench": ["p3"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert len(result) == 1
    assert result[0]["player_name"] == "AJ Brown"
    assert result[0]["injury_status"] == "Out"


def test_identify_ir_ir_player():
    """IR bench player is IR-eligible."""
    roster = {"starters": ["p1"], "bench": ["p5"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert len(result) == 1
    assert result[0]["injury_status"] == "IR"


def test_identify_ir_doubtful_player():
    """Doubtful bench player is IR-eligible."""
    roster = {"starters": ["p1"], "bench": ["p4"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert len(result) == 1
    assert result[0]["injury_status"] == "Doubtful"


def test_identify_ir_sus_player():
    """Suspended bench player is IR-eligible."""
    catalog = _catalog_df()
    catalog.loc[len(catalog)] = ["p7", "Suspended Player", "RB", "Sus", None, None]
    roster = {"starters": ["p1"], "bench": ["p7"]}
    result = identify_ir_eligible_stashes(roster, catalog)
    assert len(result) == 1
    assert result[0]["injury_status"] == "Sus"


def test_identify_ir_excludes_starters():
    """Injured starters are NOT included (they are not on bench)."""
    roster = {"starters": ["p3", "p5"], "bench": ["p1"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert len(result) == 0


def test_identify_ir_excludes_healthy_bench():
    """Healthy bench players are NOT included."""
    roster = {"starters": ["p1"], "bench": ["p6"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert len(result) == 0


def test_identify_ir_empty_roster():
    """Empty roster dict returns empty list."""
    result = identify_ir_eligible_stashes({}, _catalog_df())
    assert result == []


def test_identify_ir_empty_catalog():
    """Empty catalog returns empty list."""
    roster = {"starters": ["p1"], "bench": ["p3"]}
    result = identify_ir_eligible_stashes(roster, pd.DataFrame())
    assert result == []


def test_identify_ir_no_bench():
    """Roster with no bench returns empty list."""
    roster = {"starters": ["p1", "p2"], "bench": []}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert result == []


def test_identify_ir_multiple_eligible():
    """Multiple IR-eligible bench players are all returned."""
    roster = {"starters": ["p1"], "bench": ["p3", "p4", "p5"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert len(result) == 3
    names = {r["player_name"] for r in result}
    assert "AJ Brown" in names
    assert "Dallas Goedert" in names
    assert "DeVonta Smith" in names


def test_identify_ir_result_fields():
    """IR-eligible result dicts have all required fields."""
    roster = {"starters": ["p1"], "bench": ["p3"]}
    result = identify_ir_eligible_stashes(roster, _catalog_df())
    assert len(result) == 1
    entry = result[0]
    assert "player_id" in entry
    assert "player_name" in entry
    assert "position" in entry
    assert "injury_status" in entry
    assert "injury_body_part" in entry
