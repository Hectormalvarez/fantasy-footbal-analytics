"""CLI entry point for fantasy-football-analytics."""

import sys

import typer

import sleeper

app = typer.Typer()


@app.command()
def rosters(league_id: str) -> None:
    """Fetch and print the rosters for a Sleeper league."""
    response = sleeper.fetch_rosters(league_id)
    sys.stdout.write(response.text)
