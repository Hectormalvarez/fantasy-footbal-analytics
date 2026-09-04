"""Client helpers for the Sleeper fantasy-football API."""

from __future__ import annotations

import requests

BASE_URL = "https://api.sleeper.app/v1"
TIMEOUT = 10.0


def fetch_rosters(league_id: str, timeout: float = TIMEOUT) -> requests.Response:
    """Return the raw HTTP response for a league's rosters endpoint.

    Network and timeout errors propagate as ``requests.RequestException``.
    The caller is responsible for checking the HTTP status code.
    """
    url = f"{BASE_URL}/league/{league_id}/rosters"
    return requests.get(url, timeout=timeout)
