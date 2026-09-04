"""CLI entry point for fantasy-football-analytics."""

import typer

app = typer.Typer()


@app.command()
def rosters(league_id: str) -> None:
    """Fetch and print the rosters for a Sleeper league."""
