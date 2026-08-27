"""Sleeper API client, catalog parsing, and name normalization."""

import re

# Suffixes to strip (matched as whole trailing tokens)
_SUFFIX_PATTERN = re.compile(
    r"\b(jr|sr|ii|iii|iv|v|esq|phd)\.?\s*$", re.IGNORECASE
)

# Apostrophes (smart + straight)
_APOSTROPHE_RE = re.compile(r"['\u2019\u2018]")


def clean_player_name(name: str) -> str:
    """Normalize a player name for fuzzy matching.

    Steps: lowercase → strip suffixes → remove apostrophes → collapse whitespace.
    """
    s = name.strip().lower()
    s = _SUFFIX_PATTERN.sub("", s)
    s = _APOSTROPHE_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
