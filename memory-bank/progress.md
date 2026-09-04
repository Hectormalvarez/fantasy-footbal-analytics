# Progress

## What works
- `sleeper-cli rosters <league_id>` — fetches raw JSON from Sleeper API and writes to stdout.
- Error handling: network/timeout errors and non-200 responses write to stderr, exit code 1.
- 3 unit tests covering success, timeout, and non-200 paths.

## What's left to build
- Additional CLI subcommands (trades, players, league info).
- Jupyter notebook for analysis.
- Dev tooling in pyproject (optional-dependencies for pytest/ruff/mypy).

## Known issues
- Typer's `CliRunner` does not support `mix_stderr`; tests assert on combined output + exit code.
