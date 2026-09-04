# Tech Context

## Technologies
- Python 3.x (venv at `.venv/`)
- Typer, requests, BeautifulSoup + lxml, pandas, numpy, Jupyter
- pytest, ruff, mypy

## Development setup
- `.venv/bin/` for all Python/pip commands.
- Verify: `.venv/bin/pytest -q`, `.venv/bin/ruff check .`, `.venv/bin/mypy .`

## Constraints
- Respect robots.txt and target sites' terms of service.
- Never commit scraped data or secrets.
