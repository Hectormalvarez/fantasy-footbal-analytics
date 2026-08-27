"""Serpentine draft matrix and contingency sheet generator."""


def get_snake_picks(
    draft_slot: int,
    num_teams: int = 14,
    rounds: int = 15,
) -> list[tuple[int, int]]:
    """Return (round, pick_number) tuples for a serpentine draft.

    Parameters
    ----------
    draft_slot : Your draft position (1-indexed).
    num_teams  : Number of teams in the league (default 14).
    rounds     : Number of draft rounds to simulate (default 15).

    Returns
    -------
    list of (round_num, overall_pick) tuples.
    """
    picks = []
    for r in range(1, rounds + 1):
        if r % 2 == 1:  # odd round -> forward
            pick = (r - 1) * num_teams + draft_slot
        else:            # even round -> reverse
            pick = r * num_teams - draft_slot + 1
        picks.append((r, pick))
    return picks
