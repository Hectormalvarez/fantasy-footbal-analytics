"""Tests for src/cli.py — CLI entry point and argument parsing."""

import sys
from unittest.mock import patch

from src.cli import build_parser


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
