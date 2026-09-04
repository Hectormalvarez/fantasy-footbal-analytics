# Project: Fantasy Football Analytics

## Purpose
Data scraper for fantasy football: fetch data via a Python CLI, store it as
CSV/JSON, and analyze it in a Jupyter notebook.

## Stack
- Python 3.x, run inside a `.venv/` virtual environment.
- CLI: Typer. HTTP: requests. Parsing: BeautifulSoup + lxml.
- Data: pandas + numpy. Notebook: Jupyter.
- Tests: pytest · Lint: ruff · Types: mypy.

## Conventions
- Flat layout: library modules live at the repo root; tests under `tests/`.
- Keep scraping logic in importable modules. The CLI and the notebook both
  import the same functions — never duplicate logic between them.
- The CLI is a thin wrapper over the library functions.
- Type hints on public functions; concise docstrings.
- Run Python and pip through `.venv/bin/` when `.venv/` exists.
- Use `pathlib` for paths.
- Never commit raw/scraped data or artifacts (see .gitignore).

## Scraping etiquette & reliability
- Respect `robots.txt` and the target site's terms of service.
- Set a descriptive User-Agent identifying the project.
- Rate-limit politely: delay between requests (e.g., 1–2s) with jitter.
- Retry transient failures (429/5xx) with exponential backoff.
- Cache raw responses locally so re-runs don't re-hit the network.
- Make fetches idempotent and resumable; prefer upserts over full re-scrapes.
- Never hard-code secrets/API tokens; load them from env (see security rule).
- Parse defensively: expect schema drift; handle missing/renamed fields.
- Log every request (URL, status, duration) at debug level.

## Verification commands
- Test:  `.venv/bin/pytest -q`
- Lint:  `.venv/bin/ruff check .`
- Types: `.venv/bin/mypy .`

## Structure (flat)
- `<module>.py` — scraper/library modules at repo root
- `cli.py`     — CLI entry point
- `tests/`     — pytest tests
- `notebooks/` — Jupyter analysis
- `data/`      — scraped CSV/JSON (git-ignored)
