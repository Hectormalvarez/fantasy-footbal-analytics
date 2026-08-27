"""Tests for src/cli.py — CLI entry point and argument parsing."""

import sys
from unittest.mock import patch

from src.cli import build_parser, _format_contingency_sheet


def test_parser_requires_slot():
    """--slot is required; omitting it causes SystemExit."""
    parser = build_parser()
    try:
        parser.parse_args([])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def test_parser_slot_int():
    """--slot is parsed as an integer."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "7"])
    assert args.slot == 7


def test_parser_default_rounds():
    """--rounds defaults to 15."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "1"])
    assert args.rounds == 15


def test_parser_custom_rounds():
    """--rounds can be set to a custom value."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "1", "--rounds", "10"])
    assert args.rounds == 10


def test_parser_refresh_flag():
    """--refresh is a boolean flag, default False."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "1"])
    assert args.refresh is False


def test_parser_refresh_set():
    """--refresh becomes True when present."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "1", "--refresh"])
    assert args.refresh is True


def test_parser_reach_buffer_default():
    """--reach-buffer defaults to 4."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "1"])
    assert args.reach_buffer == 4


def test_parser_fall_buffer_default():
    """--fall-buffer defaults to 8."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "1"])
    assert args.fall_buffer == 8


def test_parser_all_args():
    """All arguments can be specified together."""
    parser = build_parser()
    args = parser.parse_args([
        "--slot", "14",
        "--rounds", "12",
        "--refresh",
        "--reach-buffer", "3",
        "--fall-buffer", "10",
    ])
    assert args.slot == 14
    assert args.rounds == 12
    assert args.refresh is True
    assert args.reach_buffer == 3
    assert args.fall_buffer == 10


def test_parser_slot_boundary_low():
    """Slot 1 is valid."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "1"])
    assert args.slot == 1


def test_parser_slot_boundary_high():
    """Slot 14 is valid."""
    parser = build_parser()
    args = parser.parse_args(["--slot", "14"])
    assert args.slot == 14



# ---------------------------------------------------------------------------
# _format_contingency_sheet tests
# ---------------------------------------------------------------------------
import pandas as pd


def test_format_contains_header():
    """Formatted output includes the slot header."""
    sheet = pd.DataFrame([{
        "round": 1, "pick": 14, "tier": "Primary",
        "vorp_rank": 1, "pos_label": "WR1", "player": "Ja'Marr Chase",
        "team": "CIN", "position": "WR", "proj_pts": 310.0,
        "vorp": 165.0, "search_rank": 4, "adp_delta": 3.0,
        "signal": "Fair Value",
    }])
    text = _format_contingency_sheet(sheet, slot=14, num_teams=14)
    assert "Slot 14 of 14" in text
    assert "Round  1" in text


def test_format_empty_sheet():
    """An empty sheet still produces a valid header."""
    sheet = pd.DataFrame()
    text = _format_contingency_sheet(sheet, slot=1, num_teams=14)
    assert "Slot 1 of 14" in text
    assert "no targets found" in text


def test_format_tiers_shown():
    """Primary, Secondary, and Value tiers appear when populated."""
    sheet = pd.DataFrame([
        {"round": 1, "pick": 14, "tier": "Primary", "vorp_rank": 1,
         "pos_label": "WR1", "player": "A", "team": "", "position": "WR",
         "proj_pts": 300, "vorp": 100, "search_rank": 14, "adp_delta": 5,
         "signal": "Slight Value"},
        {"round": 1, "pick": 14, "tier": "Secondary", "vorp_rank": 2,
         "pos_label": "RB2", "player": "B", "team": "", "position": "RB",
         "proj_pts": 250, "vorp": 60, "search_rank": 12, "adp_delta": 3,
         "signal": "Fair Value"},
        {"round": 1, "pick": 14, "tier": "Value", "vorp_rank": 3,
         "pos_label": "QB3", "player": "C", "team": "", "position": "QB",
         "proj_pts": 200, "vorp": 0, "search_rank": 30, "adp_delta": 20,
         "signal": "Major Value"},
    ])
    text = _format_contingency_sheet(sheet, slot=14, num_teams=14)
    assert "[Primary]" in text
    assert "[Secondary]" in text
    assert "[Value]" in text