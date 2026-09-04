"""CLI entry point for fantasy-football-analytics."""

import sys

import requests
import typer

import sleeper

app = typer.Typer()


@app.command()
def rosters(league_id: str) -> None:
    """Fetch and print the rosters for a Sleeper league."""
    try:
        response = sleeper.fetch_rosters(league_id)
    except requests.RequestException as exc:
        print(
            f"error: failed to fetch rosters for league '{league_id}': {exc}",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    if not response.ok:
        print(
            f"error: request failed with HTTP {response.status_code} "
            f"for league '{league_id}'",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    sys.stdout.write(response.text)
