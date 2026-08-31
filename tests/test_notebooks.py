"""Tests for Jupyter notebook structure and syntax validation."""

import ast
import os

import nbformat
import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "notebooks", "04_weekly_matchup_dashboard.ipynb",
)

EXPECTED_SECTIONS = ["# 04", "## 1", "## 2", "## 3", "## 4", "## 5"]

REQUIRED_IMPORTS = [
    "src.league", "src.sleeper", "src.matchups",
    "src.simulation", "src.dvp", "src.injuries", "src.cli",
]


def _load_notebook():
    path = os.path.abspath(NOTEBOOK_PATH)
    assert os.path.exists(path), f"Notebook not found: {path}"
    return nbformat.read(path, as_version=4)


def _all_code(nb):
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _all_markdown(nb):
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


class TestNotebookExists:
    def test_file_exists(self):
        assert os.path.isfile(os.path.abspath(NOTEBOOK_PATH))

    def test_valid_format(self):
        nb = _load_notebook()
        assert len(nb.cells) > 0


class TestCellStructure:
    def test_has_markdown_and_code(self):
        nb = _load_notebook()
        types = {c.cell_type for c in nb.cells}
        assert "markdown" in types and "code" in types

    def test_minimum_cell_count(self):
        nb = _load_notebook()
        assert len(nb.cells) >= 10

    def test_all_section_headers_present(self):
        nb = _load_notebook()
        md = _all_markdown(nb)
        for section in EXPECTED_SECTIONS:
            assert section in md, f"Section '{section}' not found"


class TestRequiredImports:
    def test_all_imports_present(self):
        code = _all_code(_load_notebook())
        for imp in REQUIRED_IMPORTS:
            assert imp in code, f"Import '{imp}' missing"


class TestCodeSyntax:
    def test_all_cells_parse(self):
        nb = _load_notebook()
        errors = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            try:
                ast.parse(cell.source)
            except SyntaxError as e:
                errors.append(f"Cell {i + 1}: {e}")
        assert not errors, "Syntax errors:\n" + "\n".join(errors)


class TestKeyReferences:
    def test_simulation(self):
        code = _all_code(_load_notebook())
        assert "simulate_team_matchup" in code
        assert "simulate_player_weekly_distribution" in code

    def test_lineup_optimization(self):
        code = _all_code(_load_notebook())
        assert "optimize_starting_lineup" in code
        assert "extract_weekly_matchup_roster" in code

    def test_dvp_adjustment(self):
        code = _all_code(_load_notebook())
        assert "calculate_defensive_rankings" in code
        assert "adjust_projections_for_matchup" in code

    def test_sit_start_tossup(self):
        code = _all_code(_load_notebook())
        assert "compare_sit_start" in code
        assert "_TOSSUP_TOLERANCE" in code

    def test_matplotlib_charts(self):
        code = _all_code(_load_notebook())
        assert "plt.subplots" in code
        assert "ax.barh" in code
        assert "fig.savefig" in code

    def test_saves_to_reports(self):
        code = _all_code(_load_notebook())
        assert "reports/" in code


class TestNotebookMetadata:
    def test_kernel_is_python(self):
        nb = _load_notebook()
        ks = nb.get("metadata", {}).get("kernelspec", {})
        assert ks.get("language") == "python"

    def test_config_constants(self):
        code = _all_code(_load_notebook())
        for const in ["LEAGUE_ID", "ROSTER_ID", "WEEK"]:
            assert const in code


@pytest.mark.network
class TestNotebookExecution:
    def test_full_execution(self):
        """Execute the notebook end-to-end. Requires live Sleeper API + Jupyter kernel."""
        nbconvert = pytest.importorskip("nbconvert")
        from nbconvert.preprocessors import ExecutePreprocessor

        nb = _load_notebook()
        ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
        path = os.path.dirname(os.path.abspath(NOTEBOOK_PATH))
        try:
            ep.preprocess(nb, {"metadata": {"path": path}})
        except Exception as exc:
            if "NoSuchKernel" in type(exc).__name__ or "NoSuchKernel" in str(exc):
                pytest.skip("Jupyter python3 kernel not installed")
            raise
