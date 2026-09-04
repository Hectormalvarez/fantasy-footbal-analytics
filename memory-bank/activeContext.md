# Active Context

## Current focus
- Shipped `sleeper-cli rosters <league_id>` command — Typer CLI + Sleeper API client.

## Recent changes
- Created `pyproject.toml` (hatchling, typer, requests deps).
- Created `sleeper.py` with `fetch_rosters()`.
- Created `cli.py` with Typer app, `rosters` subcommand, error handling (stderr + exit 1).
- Added 3 mocked tests in `tests/test_cli.py` (success, timeout, non-200).
- Fixed hatchling flat-layout config and Typer subcommand routing.

## Next steps
- Add remaining CLI subcommands (trades, players, league info, etc.).
- Wire the notebook to import from `sleeper.py` for analysis.
- Consider adding `[project.optional-dependencies]` for dev tools.
