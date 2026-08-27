"""Sleeper API client, catalog parsing, and name normalization."""

import json
import os
import re

import requests

SLEEPER_API_URL = "https://api.sleeper.app/v1/players/nfl"

# Suffixes to strip (matched as whole trailing tokens)
_SUFFIX_PATTERN = re.compile(
    r"\b(jr|sr|ii|iii|iv|v|esq|phd)\.?\s*$", re.IGNORECASE
)

# Apostrophes (smart + straight)
_APOSTROPHE_RE = re.compile(r"['\u2019\u2018]")


def clean_player_name(name: str) -> str:
    """Normalize a player name for fuzzy matching.

    Steps: lowercase -> strip suffixes -> remove apostrophes -> collapse whitespace.
    """
    s = name.strip().lower()
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
