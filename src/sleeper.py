"""Sleeper API client, catalog parsing, and name normalization."""

import json
import os
import re
import unicodedata

import pandas as pd
import requests

SLEEPER_API_URL = "https://api.sleeper.app/v1/players/nfl"

# Suffixes to strip (matched as whole trailing tokens)
_SUFFIX_PATTERN = re.compile(
    r"\b(jr|sr|ii|iii|iv|v|esq|phd)\.?\s*$", re.IGNORECASE
)

# Apostrophes (smart + straight)
_APOSTROPHE_RE = re.compile(r"['\u2019\u2018]")


def _strip_accents(text: str) -> str:
    """Strip Unicode accents via NFKD decomposition (e.g. é -> e)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if unicodedata.category(c) != "Mn"
    )


def clean_player_name(name: str) -> str:
    """Normalize a player name for fuzzy matching.

    Steps: strip accents -> lowercase -> strip suffixes -> remove apostrophes
    -> collapse whitespace.
    """
    s = _strip_accents(name.strip())
    s = s.lower()
    s = _SUFFIX_PATTERN.sub("", s)
    s = _APOSTROPHE_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_sleeper_players(
    cache_path: str = "data/sleeper_players_raw.json",
    force_refresh: bool = False,
) -> dict:
    """Download (or load from cache) the full Sleeper NFL player catalog."""
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    resp = requests.get(SLEEPER_API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data


def parse_sleeper_catalog(
    raw_data: dict,
    positions: list[str] | None = None,
) -> pd.DataFrame:
    """Parse raw Sleeper JSON into a clean DataFrame.

    Parameters
    ----------
    raw_data : dict keyed by player_id.
    positions : Filter to these positions. None keeps all.

    Returns
    -------
    pd.DataFrame with player_name, norm_name, norm_position, and raw Sleeper fields.
    """
    if positions is None:
        positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

    df = pd.DataFrame.from_dict(raw_data, orient="index")
    df = df.reset_index(drop=True)

    # Filter to skill positions + active status
    df = df[df["position"].isin(positions)].copy()
    # Keep Active status OR players with no status (DEF entries lack status)
    df = df[(df["status"] == "Active") | (df["status"].isna())].copy()

    # Rename full_name -> player_name for consistency with nflreadpy
    if "full_name" in df.columns and "player_name" not in df.columns:
        df.rename(columns={"full_name": "player_name"}, inplace=True)

    # Fill missing player_name for DEF/K entries that lack full_name
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].fillna(df.get("first_name", pd.Series(dtype=str)))
        df["player_name"] = df["player_name"].fillna(df["player_id"].astype(str))

    # Add normalization columns
    df["norm_name"] = df["player_name"].apply(clean_player_name)
    df["norm_position"] = df["position"]

    return df

