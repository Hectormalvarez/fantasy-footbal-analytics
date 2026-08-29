"""Tests for src/visuals.py -- visual cheat-sheet generators."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.visuals import (
    render_round_matrix,
    render_arbitrage_scatter,
    render_positional_cliffs,
    export_styled_html_board,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _make_sheet_df() -> pd.DataFrame:
    """Minimal contingency-sheet DataFrame for testing."""
    rows = []
    players = [
        ("Primary", "WR1", "Ja'Marr Chase", "WR", 310.0, 165.0, 4, 3.0, "Fair Value"),
        ("Primary", "RB1", "Breece Hall", "RB", 280.0, 135.0, 10, 5.0, "Fair Value"),
        ("Secondary", "QB2", "Josh Allen", "QB", 360.0, 110.0, 3, -1.0, "Fair Value"),
        ("Secondary", "TE3", "Sam LaPorta", "TE", 220.0, 80.0, 30, 15.0, "Major Value"),
        ("Value", "RB3", "James Cook", "RB", 230.0, 85.0, 50, 30.0, "Major Value"),
    ]
    for rnd in range(1, 16):
        pick = rnd * 14 if rnd % 2 == 1 else rnd * 14 - 13
        for tier, pos_label, name, pos, pts, vorp, sr, delta, sig in players:
            rows.append({
                "round": rnd, "pick": pick, "tier": tier,
                "vorp_rank": players.index((tier, pos_label, name, pos, pts, vorp, sr, delta, sig)) + 1,
                "pos_label": pos_label, "player": name, "team": "",
                "position": pos, "proj_pts": pts, "vorp": vorp,
                "search_rank": sr, "adp_delta": delta, "signal": sig,
            })
    return pd.DataFrame(rows)


def _make_board_df() -> pd.DataFrame:
    """Minimal draft-board DataFrame for testing."""
    players = [
        ("Ja'Marr Chase", "WR", 310.0, 4.0, 3.0),
        ("Breece Hall", "RB", 280.0, 10.0, 5.0),
        ("Josh Allen", "QB", 360.0, 3.0, -1.0),
        ("Sam LaPorta", "TE", 220.0, 30.0, 15.0),
        ("James Cook", "RB", 230.0, 50.0, 30.0),
        ("CeeDee Lamb", "WR", 300.0, 5.0, 2.0),
        ("Bijan Robinson", "RB", 290.0, 6.0, 3.0),
        ("Tyreek Hill", "WR", 290.0, 8.0, -3.0),
        ("Travis Kelce", "TE", 210.0, 55.0, 20.0),
        ("Lamar Jackson", "QB", 400.0, 2.0, 1.0),
    ]
    rows = []
    for i, (name, pos, pts, sr, delta) in enumerate(players, start=1):
        vorp = pts - {"WR": 145.0, "RB": 130.0, "QB": 200.0, "TE": 100.0}[pos]
        rows.append({
            "vorp_rank": i,
            "pos_label": f"{pos}{i}",
            "player_name": name,
            "position_proj": pos,
            "proj_points": pts,
            "vorp": vorp,
            "search_rank": sr,
            "adp_delta": delta,
            "signal": "Fair Value",
            "pos_rank": i,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# render_round_matrix tests
# ---------------------------------------------------------------------------
def test_render_round_matrix_returns_figure():
    """render_round_matrix returns a matplotlib Figure."""
    sheet = _make_sheet_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_matrix.png")
        fig = render_round_matrix(sheet, draft_slot=7, output_path=path)
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_render_round_matrix_default_path():
    """When output_path is None, file is saved to data/cheat_sheet_slot_{N}.png."""
    sheet = _make_sheet_df()
    fig = render_round_matrix(sheet, draft_slot=3)
    assert os.path.exists("data/cheat_sheet_slot_3.png")
    plt.close(fig)


def test_render_round_matrix_empty_sheet():
    """An empty sheet still produces a valid figure."""
    sheet = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty.png")
        fig = render_round_matrix(sheet, draft_slot=1, output_path=path)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# render_arbitrage_scatter tests
# ---------------------------------------------------------------------------
def test_render_arbitrage_scatter_returns_figure():
    """render_arbitrage_scatter returns a matplotlib Figure."""
    board = _make_board_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_scatter.png")
        fig = render_arbitrage_scatter(board, top_n=10, output_path=path)
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_render_arbitrage_scatter_default_path():
    """Default output path is data/market_arbitrage.png."""
    board = _make_board_df()
    fig = render_arbitrage_scatter(board, top_n=10)
    assert os.path.exists("data/market_arbitrage.png")
    plt.close(fig)


def test_render_arbitrage_scatter_empty():
    """Empty board still produces a figure."""
    board = pd.DataFrame(columns=["vorp_rank", "position_proj", "search_rank",
                                   "adp_delta", "player_name"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty_scatter.png")
        fig = render_arbitrage_scatter(board, output_path=path)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# render_positional_cliffs tests
# ---------------------------------------------------------------------------
def test_render_positional_cliffs_returns_figure():
    """render_positional_cliffs returns a matplotlib Figure."""
    board = _make_board_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cliffs.png")
        fig = render_positional_cliffs(board, top_n=10, output_path=path)
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_render_positional_cliffs_default_path():
    """Default output path is data/positional_cliffs.png."""
    board = _make_board_df()
    fig = render_positional_cliffs(board, top_n=10)
    assert os.path.exists("data/positional_cliffs.png")
    plt.close(fig)


def test_render_positional_cliffs_empty():
    """Empty board still produces a figure."""
    board = pd.DataFrame(columns=["vorp_rank", "position_proj", "vorp", "pos_rank"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty_cliffs.png")
        fig = render_positional_cliffs(board, output_path=path)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# export_styled_html_board tests
# ---------------------------------------------------------------------------
def test_export_styled_html_returns_path():
    """export_styled_html_board returns the output path and creates a file."""
    board = _make_board_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "board.html")
        result = export_styled_html_board(board, output_path=path)
        assert result == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_export_styled_html_default_path():
    """Default output path is data/draft_board_styled.html."""
    board = _make_board_df()
    result = export_styled_html_board(board)
    assert os.path.exists("data/draft_board_styled.html")


def test_export_styled_html_contains_table():
    """Output HTML contains a <table> element."""
    board = _make_board_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "board.html")
        export_styled_html_board(board, output_path=path)
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        assert "<table" in html
        assert "<style>" in html
        assert "Ja'Marr Chase" in html
