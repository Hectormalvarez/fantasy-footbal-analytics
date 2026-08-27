"""Tests for src/vorp.py — VORP engine and ADP arbitrage signals."""

from src.vorp import classify_signal


def test_major_value():
    """adp_delta >= 10 -> Major Value."""
    assert classify_signal(10) == "\U0001f525 Major Value"
    assert classify_signal(15) == "\U0001f525 Major Value"


def test_slight_value():
    """5 <= adp_delta < 10 -> Slight Value."""
    assert classify_signal(5) == "\u2705 Slight Value"
    assert classify_signal(9) == "\u2705 Slight Value"


def test_fair_value():
    """-5 < adp_delta < 5 -> Fair Value."""
    assert classify_signal(0) == "\u2696\ufe0f Fair Value"
    assert classify_signal(-4) == "\u2696\ufe0f Fair Value"
    assert classify_signal(4) == "\u2696\ufe0f Fair Value"


def test_overpriced():
    """-10 < adp_delta <= -5 -> Overpriced."""
    assert classify_signal(-5) == "\u26a0\ufe0f Overpriced"
    assert classify_signal(-9) == "\u26a0\ufe0f Overpriced"


def test_heavy_reach():
    """adp_delta <= -10 -> Heavy Reach."""
    assert classify_signal(-10) == "\U0001f6ab Heavy Reach"
    assert classify_signal(-20) == "\U0001f6ab Heavy Reach"
