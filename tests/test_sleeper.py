"""Tests for src/sleeper.py — Sleeper API, catalog parsing, name normalization."""

from src.sleeper import clean_player_name


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
