---
paths:
  - "data/**"
  - "notebooks/**"
  - "**/*.csv"
  - "**/*.json"
---
# Data Handling Rules

- Never commit raw or scraped data; keep it under git-ignored `data/` paths.
- Store scraped output as CSV or JSON with stable, documented schemas.
- Pipelines must be reproducible: pin random seeds, sort output,
  avoid non-deterministic ordering.
- Validate schemas/columns after loading; handle missing values explicitly
  (never silently drop).
- Read/write with explicit encodings and dtypes.
- Document data provenance (source + fetch date) for every dataset.
