# System Patterns

## Architecture
- Flat layout: library modules at repo root, tests in `tests/`.
- CLI (`cli.py`) is a thin wrapper; the notebook imports the same functions.

## Key technical decisions
- CLI: Typer. HTTP: requests. Parsing: BeautifulSoup + lxml.
- Scraped data stored as CSV/JSON under git-ignored `data/`.

## Component relationships
- (To be filled in as modules are built.)
