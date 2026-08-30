"""Multi-team trade evaluator & starting lineup impact engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.matchups import optimize_starting_lineup


def simulate_roster_swap(
    roster_player_ids: list[str],
    outgoing_ids: list[str],
    incoming_ids: list[str],
) -> list[str]:
    """Validate and apply player swaps to produce a post-trade roster.

    Parameters
    ----------
    roster_player_ids : Current player_id strings on the roster.
    outgoing_ids : Player_id strings being sent away.
    incoming_ids : Player_id strings being acquired.

    Returns
    -------
    New roster as a list of player_id strings.

    Raises
    ------
    ValueError : If an outgoing player is not on the roster or an incoming
        player is already on the roster.
    """
    roster = set(roster_player_ids)

    for pid in outgoing_ids:
        if pid not in roster:
            raise ValueError(f"Player {pid!r} not on roster")
    for pid in incoming_ids:
        if pid in roster:
            raise ValueError(f"Player {pid!r} already on roster")

    roster -= set(outgoing_ids)
    roster |= set(incoming_ids)
    return sorted(roster)
