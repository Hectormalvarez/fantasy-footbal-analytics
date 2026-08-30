"""Monte Carlo simulation engine for matchup win probabilities and power rankings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def simulate_player_weekly_distribution(
    projected_points: float,
    floor_ratio: float = 0.70,
    ceiling_ratio: float = 1.30,
    iterations: int = 10000,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate a variance-bounded array of simulated point outcomes.

    Parameters
    ----------
    projected_points : Baseline weekly projection (mean).
    floor_ratio : Lower bound as a fraction of projected_points (e.g. 0.70).
    ceiling_ratio : Upper bound as a fraction of projected_points (e.g. 1.30).
    iterations : Number of Monte Carlo samples to generate.
    random_seed : Optional seed for reproducibility.

    Returns
    -------
    1-D numpy array of length ``iterations`` with simulated weekly points,
    clipped to ``[projected_points * floor_ratio, projected_points * ceiling_ratio]``.
    """
    rng = np.random.default_rng(random_seed)

    if projected_points <= 0:
        return np.zeros(iterations)

    # Derive std-dev from the ceiling ratio:  P(X <= ceiling) ~ 0.90
    # For a normal distribution, the 90th percentile is ~1.282 sigma above mean.
    ceiling_val = projected_points * ceiling_ratio
    std = max((ceiling_val - projected_points) / 1.282, 0.1)

    samples = rng.normal(loc=projected_points, scale=std, size=iterations)

    floor_val = projected_points * floor_ratio
    np.clip(samples, floor_val, ceiling_val, out=samples)
    return samples


def simulate_team_matchup(
    team_a_starters_df: pd.DataFrame,
    team_b_starters_df: pd.DataFrame,
    iterations: int = 10000,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Run parallel Monte Carlo simulations of two starting lineups.

    Parameters
    ----------
    team_a_starters_df : DataFrame with ``player_id``, ``position``,
        ``proj_points`` for Team A starters.
    team_b_starters_df : Same structure for Team B starters.
    iterations : Number of simulation iterations.
    random_seed : Optional seed for reproducibility.

    Returns
    -------
    Dict with keys:
    - ``win_prob_a``: Team A win probability (0.0-1.0)
    - ``win_prob_b``: Team B win probability (0.0-1.0)
    - ``median_a``: Median simulated score for Team A
    - ``median_b``: Median simulated score for Team B
    - ``floor_a``: 10th percentile score for Team A
    - ``floor_b``: 10th percentile score for Team B
    - ``ceiling_a``: 90th percentile score for Team A
    - ``ceiling_b``: 90th percentile score for Team B
    - ``avg_margin``: Average margin of victory (positive = Team A wins)
    - ``iterations``: Number of iterations used
    """
    rng = np.random.default_rng(random_seed)

    def _empty_result() -> dict[str, Any]:
        return {
            "win_prob_a": 0.5, "win_prob_b": 0.5,
            "median_a": 0.0, "median_b": 0.0,
            "floor_a": 0.0, "floor_b": 0.0,
            "ceiling_a": 0.0, "ceiling_b": 0.0,
            "avg_margin": 0.0, "iterations": iterations,
        }

    if team_a_starters_df.empty or team_b_starters_df.empty:
        return _empty_result()

    team_a_total = np.zeros(iterations)
    for _, row in team_a_starters_df.iterrows():
        proj = float(row.get("proj_points", 0))
        team_a_total += simulate_player_weekly_distribution(
            proj, iterations=iterations,
            random_seed=int(rng.integers(0, 2**31)),
        )

    team_b_total = np.zeros(iterations)
    for _, row in team_b_starters_df.iterrows():
        proj = float(row.get("proj_points", 0))
        team_b_total += simulate_player_weekly_distribution(
            proj, iterations=iterations,
            random_seed=int(rng.integers(0, 2**31)),
        )

    a_wins = int(np.sum(team_a_total > team_b_total))
    b_wins = int(np.sum(team_b_total > team_a_total))
    ties = iterations - a_wins - b_wins

    return {
        "win_prob_a": round((a_wins + ties * 0.5) / iterations, 4),
        "win_prob_b": round((b_wins + ties * 0.5) / iterations, 4),
        "median_a": round(float(np.median(team_a_total)), 1),
        "median_b": round(float(np.median(team_b_total)), 1),
        "floor_a": round(float(np.percentile(team_a_total, 10)), 1),
        "floor_b": round(float(np.percentile(team_b_total, 10)), 1),
        "ceiling_a": round(float(np.percentile(team_a_total, 90)), 1),
        "ceiling_b": round(float(np.percentile(team_b_total, 90)), 1),
        "avg_margin": round(float(np.mean(team_a_total - team_b_total)), 2),
        "iterations": iterations,
    }


def calculate_league_power_rankings(
    rosters_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    iterations: int = 2000,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Perform all-play-all round-robin simulation for league power rankings.

    Parameters
    ----------
    rosters_df : DataFrame with ``roster_id`` and ``player_id`` columns.
        Each unique ``roster_id`` is one team.
    projections_df : Projection table with ``player_id``, ``position``,
        ``proj_points``.
    iterations : Simulation iterations per head-to-head matchup.
    random_seed : Optional seed for reproducibility.

    Returns
    -------
    DataFrame with columns ``roster_id``, ``true_talent_win_pct``,
        ``power_rank`` (1 = best), ``median_total``, ``floor_total``,
        ``ceiling_total``.
    """
    rng = np.random.default_rng(random_seed)

    if rosters_df.empty or projections_df.empty:
        return pd.DataFrame(
            columns=["roster_id", "true_talent_win_pct", "power_rank",
                     "median_total", "floor_total", "ceiling_total"]
        )

    roster_ids = sorted(rosters_df["roster_id"].unique())
    n = len(roster_ids)
    if n < 2:
        return pd.DataFrame(
            columns=["roster_id", "true_talent_win_pct", "power_rank",
                     "median_total", "floor_total", "ceiling_total"]
        )

    # Build per-roster starter DataFrames (top 9 by projection per roster)
    proj_lookup = {}
    for _, row in projections_df.iterrows():
        proj_lookup[row["player_id"]] = {
            "player_id": row["player_id"],
            "position": row.get("position", "UNKNOWN"),
            "proj_points": float(row.get("proj_points", 0)),
        }

    roster_starters: dict[int, pd.DataFrame] = {}
    for rid in roster_ids:
        pids = rosters_df[rosters_df["roster_id"] == rid]["player_id"].tolist()
        players = [proj_lookup[pid] for pid in pids if pid in proj_lookup]
        players.sort(key=lambda x: x["proj_points"], reverse=True)
        roster_starters[rid] = pd.DataFrame(players[:9])

    # Simulate all head-to-head matchups
    win_counts: dict[int, int] = {rid: 0 for rid in roster_ids}
    total_possible_wins = 0
    total_scores: dict[int, list[float]] = {rid: [] for rid in roster_ids}

    for i in range(n):
        for j in range(i + 1, n):
            rid_a = roster_ids[i]
            rid_b = roster_ids[j]
            team_a = roster_starters[rid_a]
            team_b = roster_starters[rid_b]

            seed_a = int(rng.integers(0, 2**31))
            result = simulate_team_matchup(
                team_a, team_b,
                iterations=iterations,
                random_seed=seed_a,
            )

            total_possible_wins += result["iterations"]

            # Accumulate wins from head-to-head probabilities
            win_counts[rid_a] += round(result["win_prob_a"] * result["iterations"])
            win_counts[rid_b] += round(result["win_prob_b"] * result["iterations"])

            # Accumulate simulated totals for median/floor/ceiling
            team_a_sims = simulate_team_matchup(
                team_a, team_a, iterations=iterations,
                random_seed=int(rng.integers(0, 2**31)),
            )
            total_scores[rid_a].append(team_a_sims["median_a"])

            team_b_sims = simulate_team_matchup(
                team_b, team_b, iterations=iterations,
                random_seed=int(rng.integers(0, 2**31)),
            )
            total_scores[rid_b].append(team_b_sims["median_a"])

    # Build results DataFrame
    rows = []
    for rid in roster_ids:
        win_pct = win_counts[rid] / max(total_possible_wins, 1)
        scores = total_scores[rid]
        median_total = round(float(np.median(scores)), 1) if scores else 0.0
        floor_total = round(float(np.percentile(scores, 10)), 1) if scores else 0.0
        ceiling_total = round(float(np.percentile(scores, 90)), 1) if scores else 0.0
        rows.append({
            "roster_id": rid,
            "true_talent_win_pct": round(win_pct, 4),
            "median_total": median_total,
            "floor_total": floor_total,
            "ceiling_total": ceiling_total,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("true_talent_win_pct", ascending=False).reset_index(drop=True)
    df["power_rank"] = df.index + 1
    return df[["roster_id", "true_talent_win_pct", "power_rank",
               "median_total", "floor_total", "ceiling_total"]]
