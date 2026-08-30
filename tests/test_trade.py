"""Tests for src/trade.py — multi-team trade evaluator & lineup impact engine."""

import pandas as pd
import pytest

from src.trade import simulate_roster_swap


# ---------------------------------------------------------------------------
# simulate_roster_swap
# ---------------------------------------------------------------------------
def test_simulate_swap_basic():
    """Basic 1-for-1 swap removes outgoing and adds incoming."""
    roster = ["p1", "p2", "p3", "p4"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p5"])
    assert "p5" in result
    assert "p2" not in result
    assert len(result) == len(roster)


def test_simulate_swap_preserves_other_players():
    """Players not involved in the trade remain on the roster."""
    roster = ["p1", "p2", "p3", "p4"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p5"])
    assert set(result) == {"p1", "p3", "p4", "p5"}


def test_simulate_swap_2_for_1():
    """2-for-1 consolidation: two outgoing, one incoming."""
    roster = ["p1", "p2", "p3", "p4"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2", "p3"], incoming_ids=["p5"])
    assert set(result) == {"p1", "p4", "p5"}
    assert len(result) == 3


def test_simulate_swap_1_for_2():
    """1-for-2 expansion: one outgoing, two incoming."""
    roster = ["p1", "p2", "p3"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p5", "p6"])
    assert set(result) == {"p1", "p3", "p5", "p6"}
    assert len(result) == 4


def test_simulate_swap_outgoing_not_on_roster():
    """Trading a player not on the roster raises ValueError."""
    roster = ["p1", "p2"]
    with pytest.raises(ValueError, match="not on roster"):
        simulate_roster_swap(roster, outgoing_ids=["p99"], incoming_ids=["p3"])


def test_simulate_swap_incoming_already_on_roster():
    """Acquiring a player already on the roster raises ValueError."""
    roster = ["p1", "p2", "p3"]
    with pytest.raises(ValueError, match="already on roster"):
        simulate_roster_swap(roster, outgoing_ids=["p1"], incoming_ids=["p2"])


def test_simulate_swap_noop():
    """Empty outgoing list returns the original roster unchanged."""
    roster = ["p1", "p2", "p3"]
    result = simulate_roster_swap(roster, outgoing_ids=[], incoming_ids=[])
    assert set(result) == {"p1", "p2", "p3"}


def test_simulate_swap_returns_list():
    """Result is a list of strings."""
    roster = ["p1", "p2"]
    result = simulate_roster_swap(roster, outgoing_ids=["p2"], incoming_ids=["p3"])
    assert isinstance(result, list)
    assert all(isinstance(pid, str) for pid in result)


# ---------------------------------------------------------------------------
# calculate_starting_lineup_delta
# ---------------------------------------------------------------------------
from src.trade import calculate_starting_lineup_delta


def _proj_df():
    """Return a projections DataFrame used for lineup optimization tests."""
    return pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "player_name": [
                "Jalen Hurts", "Saquon Barkley", "AJ Brown",
                "Dallas Goedert", "DeVonta Smith", "Jaylen Waddle",
            ],
            "position": ["QB", "RB", "WR", "TE", "WR", "WR"],
            "proj_points": [25.0, 18.0, 16.0, 10.0, 14.0, 12.0],
        }
    )


def test_lineup_delta_starter_upgrade():
    """Replacing a bench WR with a better WR boosts starting lineup."""
    pre = ["p1", "p2", "p3", "p4", "p6"]  # starter WR: p3(16)
    post = ["p1", "p2", "p5", "p4", "p6"]  # starter WR: p5(14) → downgrade
    delta = calculate_starting_lineup_delta(pre, post, _proj_df())
    assert isinstance(delta, float)
    assert delta < 0  # starting lineup got worse


def test_lineup_delta_bench_only_change():
    """Bench-only swap does not affect starting lineup delta."""
    pre = ["p1", "p2", "p3", "p4", "p5", "p6"]
    # Replace p6(WR, 12) with a new WR "p7"(13) — neither starts with 2-WR slots
    post = ["p1", "p2", "p3", "p4", "p5", "p7"]
    proj = _proj_df()
    proj = pd.concat(
        [proj, pd.DataFrame({"player_id": ["p7"], "player_name": ["X"],
                              "position": ["WR"], "proj_points": [13.0]})],
        ignore_index=True,
    )
    slots = ["QB", "RB1", "WR1", "WR2", "TE"]
    delta = calculate_starting_lineup_delta(pre, post, proj, roster_slots=slots)
    assert delta == pytest.approx(0.0)


def test_lineup_delta_positive_upgrade():
    """Upgrading a starter yields a positive delta."""
    pre = ["p1", "p6", "p3", "p4"]     # FLEX: p6(12)
    post = ["p1", "p5", "p3", "p4"]    # FLEX: p5(14)
    delta = calculate_starting_lineup_delta(pre, post, _proj_df())
    assert delta > 0


def test_lineup_delta_empty_rosters():
    """Empty rosters produce zero delta."""
    delta = calculate_starting_lineup_delta([], [], _proj_df())
    assert delta == pytest.approx(0.0)


def test_lineup_delta_custom_slots():
    """Custom roster_slots are passed through to optimizer."""
    pre = ["p1", "p2", "p3", "p4"]
    post = ["p1", "p2", "p3", "p4"]  # same roster → delta 0
    slots = ["QB", "RB1", "WR1", "TE"]
    delta = calculate_starting_lineup_delta(pre, post, _proj_df(), roster_slots=slots)
    assert delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# classify_trade_verdict
# ---------------------------------------------------------------------------
from src.trade import classify_trade_verdict


def test_verdict_strong_accept():
    """Large positive deltas in both PPG and VORP → Strong Accept."""
    assert classify_trade_verdict(6.0, 25.0) == "Strong Accept"


def test_verdict_slight_upgrade():
    """Moderate positive → Slight Upgrade."""
    assert classify_trade_verdict(2.0, 10.0) == "Slight Upgrade"


def test_verdict_fair_trade():
    """Both deltas near zero → Fair Trade."""
    assert classify_trade_verdict(0.5, 3.0) == "Fair Trade"


def test_verdict_lateral_move():
    """Tiny deltas → Lateral Move."""
    assert classify_trade_verdict(0.1, 1.0) == "Lateral Move"


def test_verdict_decline():
    """Negative net delta → Decline / Value Loss."""
    assert classify_trade_verdict(-3.0, -15.0) == "Decline / Value Loss"


def test_verdict_boundary_strong_accept():
    """At exact boundary (5.0 PPG, 20 VORP) → Strong Accept."""
    assert classify_trade_verdict(5.0, 20.0) == "Strong Accept"


def test_verdict_boundary_slight_upgrade():
    """Just below Strong Accept threshold → Slight Upgrade."""
    assert classify_trade_verdict(4.9, 19.9) == "Slight Upgrade"


# ---------------------------------------------------------------------------
# evaluate_trade
# ---------------------------------------------------------------------------
from src.trade import evaluate_trade


def _eval_proj_df():
    """Projections DataFrame for evaluate_trade tests."""
    return pd.DataFrame(
        {
            "player_id": [
                "p1", "p2", "p3", "p4", "p5", "p6",
                "p7", "p8", "p9", "p10",
            ],
            "player_name": [
                "Jalen Hurts", "Saquon Barkley", "AJ Brown",
                "Dallas Goedert", "DeVonta Smith", "Jaylen Waddle",
                "Tyreek Hill", "Travis Kelce", "CeeDee Lamb",
                "Derrick Henry",
            ],
            "position": ["QB", "RB", "WR", "TE", "WR", "WR",
                         "WR", "TE", "WR", "RB"],
            "proj_points": [25.0, 18.0, 16.0, 10.0, 14.0, 12.0,
                            17.0, 11.0, 15.0, 13.0],
        }
    )


def _eval_draft_board():
    """Draft board DataFrame with VORP column."""
    return pd.DataFrame(
        {
            "player_id": [
                "p1", "p2", "p3", "p4", "p5", "p6",
                "p7", "p8", "p9", "p10",
            ],
            "player_name": [
                "Jalen Hurts", "Saquon Barkley", "AJ Brown",
                "Dallas Goedert", "DeVonta Smith", "Jaylen Waddle",
                "Tyreek Hill", "Travis Kelce", "CeeDee Lamb",
                "Derrick Henry",
            ],
            "position": ["QB", "RB", "WR", "TE", "WR", "WR",
                         "WR", "TE", "WR", "RB"],
            "vorp": [15.0, 12.0, 10.0, 4.0, 8.0, 6.0,
                     11.0, 5.0, 9.0, 7.0],
        }
    )


def test_evaluate_trade_returns_dict():
    """evaluate_trade returns a dict with expected keys."""
    proj = _eval_proj_df()
    db = _eval_draft_board()
    result = evaluate_trade(
        team_a_roster=["p1", "p2", "p3", "p4"],
        team_b_roster=["p7", "p8", "p9", "p10"],
        team_a_sends=["p3"],
        team_b_sends=["p9"],
        projections_df=proj,
        draft_board_df=db,
    )
    assert isinstance(result, dict)
    assert "team_a" in result
    assert "team_b" in result
    assert "verdict_a" in result
    assert "verdict_b" in result


def test_evaluate_trade_team_fields():
    """Each team sub-dict has starting_delta and vorp_delta."""
    proj = _eval_proj_df()
    db = _eval_draft_board()
    result = evaluate_trade(
        team_a_roster=["p1", "p2", "p3", "p4"],
        team_b_roster=["p7", "p8", "p9", "p10"],
        team_a_sends=["p3"],
        team_b_sends=["p9"],
        projections_df=proj,
        draft_board_df=db,
    )
    for side in ("team_a", "team_b"):
        assert "starting_delta" in result[side]
        assert "vorp_delta" in result[side]
        assert isinstance(result[side]["starting_delta"], float)
        assert isinstance(result[side]["vorp_delta"], float)


def test_evaluate_trade_symmetric_deltas():
    """Team A gains what Team B loses and vice versa."""
    proj = _eval_proj_df()
    db = _eval_draft_board()
    result = evaluate_trade(
        team_a_roster=["p1", "p2", "p3", "p4"],
        team_b_roster=["p7", "p8", "p9", "p10"],
        team_a_sends=["p3"],
        team_b_sends=["p9"],
        projections_df=proj,
        draft_board_df=db,
    )
    assert result["team_a"]["starting_delta"] == pytest.approx(
        -result["team_b"]["starting_delta"]
    )
    assert result["team_a"]["vorp_delta"] == pytest.approx(
        -result["team_b"]["vorp_delta"]
    )


def test_evaluate_trade_2_for_1():
    """2-for-1 consolidation trade evaluates without errors."""
    proj = _eval_proj_df()
    db = _eval_draft_board()
    result = evaluate_trade(
        team_a_roster=["p1", "p2", "p3", "p5", "p4"],
        team_b_roster=["p7", "p8", "p9", "p10"],
        team_a_sends=["p3", "p5"],
        team_b_sends=["p9"],
        projections_df=proj,
        draft_board_df=db,
    )
    assert isinstance(result["verdict_a"], str)
    assert isinstance(result["verdict_b"], str)


def test_evaluate_trade_invalid_outgoing():
    """Trading a player not on the roster raises ValueError."""
    proj = _eval_proj_df()
    db = _eval_draft_board()
    with pytest.raises(ValueError, match="not on roster"):
        evaluate_trade(
            team_a_roster=["p1", "p2"],
            team_b_roster=["p7", "p8"],
            team_a_sends=["p99"],
            team_b_sends=["p8"],
            projections_df=proj,
            draft_board_df=db,
        )


def test_evaluate_trade_returns_verdicts():
    """Verdicts are valid classification strings."""
    proj = _eval_proj_df()
    db = _eval_draft_board()
    result = evaluate_trade(
        team_a_roster=["p1", "p2", "p3", "p4"],
        team_b_roster=["p7", "p8", "p9", "p10"],
        team_a_sends=["p3"],
        team_b_sends=["p9"],
        projections_df=proj,
        draft_board_df=db,
    )
    valid = {
        "Strong Accept", "Slight Upgrade", "Fair Trade",
        "Lateral Move", "Decline / Value Loss",
    }
    assert result["verdict_a"] in valid
    assert result["verdict_b"] in valid


def test_evaluate_trade_asymmetric_impact():
    """One side upgrades, other downgrades — deltas are opposite signs."""
    proj = _eval_proj_df()
    db = _eval_draft_board()
    result = evaluate_trade(
        team_a_roster=["p1", "p2", "p3", "p4", "p6"],
        team_b_roster=["p7", "p8", "p9", "p10"],
        team_a_sends=["p6"],
        team_b_sends=["p10"],
        projections_df=proj,
        draft_board_df=db,
    )
    assert isinstance(result["verdict_a"], str)
    assert isinstance(result["verdict_b"], str)
    # Opposite signs for starting delta
    assert result["team_a"]["starting_delta"] >= 0
    assert result["team_b"]["starting_delta"] <= 0

