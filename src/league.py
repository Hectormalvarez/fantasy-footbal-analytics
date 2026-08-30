"""Sleeper League API client and data transformers."""

import requests
import pandas as pd

SLEEPER_BASE_URL = "https://api.sleeper.app/v1"


def fetch_league(league_id: str) -> dict:
    """Fetch league metadata from the Sleeper API."""
    resp = requests.get(f"{SLEEPER_BASE_URL}/league/{league_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()
