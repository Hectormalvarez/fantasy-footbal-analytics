# Testing Standards

- Use pytest. Name files `test_*.py`, functions `test_<behavior>`.
- Descriptive names: `test_<function>_<scenario>_<expected>`.
- Keep unit tests fast; mark slow/integration tests with
  `@pytest.mark.integration`.
- One logical assertion per test where practical; use
  `@pytest.mark.parametrize` for repeated cases.
- Mock network/HTTP calls; never hit live sites in tests.
- Mock external dependencies (APIs, DBs); do not mock internal helpers.
- Always run with auto-terminating flags (`pytest -q`); never watch/interactive mode.
- Add or update tests alongside any behavior change.
