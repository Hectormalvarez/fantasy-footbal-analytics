"""Visual cheat-sheet generators for the fantasy football draft board.

All renderers use ``matplotlib.use('Agg')`` for headless (non-interactive)
backends so they can run in CI or from the CLI without a display server.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# -- Colour palette -----------------------------------------------------------
_POS_COLORS = {
    "QB": "#4C78A8",
    "RB": "#54A24B",
    "WR": "#E45756",
    "TE": "#F58518",
}

_BG_PRIMARY = "#D5F5E3"   # soft green
_BG_SECONDARY = "#FEF9E7"  # soft yellow
_BG_REACH = "#FADBD8"      # soft red


# -- 1. Round-by-round matrix ------------------------------------------------
def render_round_matrix(
    sheet_df: pd.DataFrame,
    draft_slot: int,
    output_path: str | None = None,
) -> plt.Figure:
    """Render a 15-round cheat-sheet table for *draft_slot*.

    Parameters
    ----------
    sheet_df : DataFrame produced by ``generate_contingency_sheet``.
    draft_slot : Draft position (1-indexed).
    output_path : Where to save the PNG. Defaults to
        ``data/cheat_sheet_slot_{draft_slot}.png``.

    Returns
    -------
    matplotlib Figure.
    """
    if output_path is None:
        output_path = f"data/cheat_sheet_slot_{draft_slot}.png"

    max_rounds = int(sheet_df["round"].max()) if not sheet_df.empty else 15
    max_rounds = max(max_rounds, 15)

    # Build per-round summary rows
    rows: list[dict] = []
    for rnd in range(1, max_rounds + 1):
        rnd_df = sheet_df[sheet_df["round"] == rnd] if not sheet_df.empty else pd.DataFrame()
        pick = int(rnd_df["pick"].iloc[0]) if not rnd_df.empty else "--"

        def _fmt_players(tier: str) -> str:
            t_df = rnd_df[rnd_df["tier"] == tier] if not rnd_df.empty else pd.DataFrame()
            if t_df.empty:
                return "--"
            parts = []
            for _, p in t_df.iterrows():
                parts.append(f"{p['pos_label']} {p['player']}")
            return ", ".join(parts)

        rows.append({
            "Round": rnd,
            "Pick": pick,
            "Primary Targets": _fmt_players("Primary"),
            "Secondary Pivots": _fmt_players("Secondary"),
            "Reaches/Fades": _fmt_players("Value"),
        })

    fig, ax = plt.subplots(figsize=(20, max_rounds * 0.55 + 1.5))
    ax.axis("off")
    ax.set_title(
        f"Draft Cheat Sheet -- Slot {draft_slot}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    table = ax.table(
        cellText=[list(r.values()) for r in rows],
        colLabels=list(rows[0].keys()) if rows else [],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    # Style header row
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 2:  # Primary Targets
            cell.set_facecolor(_BG_PRIMARY)
        elif c == 3:  # Secondary Pivots
            cell.set_facecolor(_BG_SECONDARY)
        elif c == 4:  # Reaches/Fades
            cell.set_facecolor(_BG_REACH)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# -- 2. Market-arbitrage scatter ---------------------------------------------
def render_arbitrage_scatter(
    board_df: pd.DataFrame,
    top_n: int = 120,
    output_path: str | None = None,
) -> plt.Figure:
    """Scatter: Sleeper ``search_rank`` (X) vs ``vorp_rank`` (Y, inverted).

    The y = x parity line divides *Value* (below) from *Reach* (above).
    Top-8 positive and top-5 negative ``adp_delta`` outliers are annotated.

    Parameters
    ----------
    board_df : Full VORP draft board.
    top_n : Restrict to the top *top_n* players by vorp_rank.
    output_path : Save path. Defaults to ``data/market_arbitrage.png``.

    Returns
    -------
    matplotlib Figure.
    """
    if output_path is None:
        output_path = "data/market_arbitrage.png"

    df = board_df[board_df["vorp_rank"] <= top_n].copy()
    if df.empty:
        fig, ax = plt.subplots()
        ax.set_title("No data")
        return fig

    df["search_rank"] = pd.to_numeric(df["search_rank"], errors="coerce")
    df["vorp_rank"] = pd.to_numeric(df["vorp_rank"], errors="coerce")
    df = df.dropna(subset=["search_rank", "vorp_rank"])

    fig, ax = plt.subplots(figsize=(12, 10))

    # Quadrant fills — y-axis is inverted, so "above" the parity line
    # (green = value) is where vorp_rank < search_rank, and "below"
    # (red = reach) is where vorp_rank > search_rank.
    max_val = max(df["search_rank"].max(), df["vorp_rank"].max()) * 1.05
    ax.fill_between(
        [0, max_val], [0, max_val], [max_val, max_val],
        color="#FDEDEC", alpha=0.5, label="_nolegend_",
    )  # Reach quadrant (below parity in data coords / below on screen)
    ax.fill_between(
        [0, max_val], [0, 0], [0, max_val],
        color="#E8F8F5", alpha=0.5, label="_nolegend_",
    )  # Value quadrant (above parity in data coords / above on screen)

    # Scatter by position
    for pos, color in _POS_COLORS.items():
        mask = df["position_proj"] == pos
        if mask.any():
            ax.scatter(
                df.loc[mask, "search_rank"],
                df.loc[mask, "vorp_rank"],
                c=color, label=pos, alpha=0.7, edgecolors="white",
                linewidths=0.5, s=50, zorder=3,
            )

    # Parity line
    ax.plot(
        [0, max_val], [0, max_val],
        color="grey", linestyle="--", linewidth=1, alpha=0.6, zorder=2,
    )

    # Annotate top positive-delta (value) outliers
    top_pos = df.nlargest(8, "adp_delta")
    for _, row in top_pos.iterrows():
        ax.annotate(
            row["player_name"],
            (row["search_rank"], row["vorp_rank"]),
            fontsize=7, fontstyle="italic",
            xytext=(5, 5), textcoords="offset points",
        )

    # Annotate top negative-delta (reach) outliers
    top_neg = df.nsmallest(5, "adp_delta")
    for _, row in top_neg.iterrows():
        ax.annotate(
            row["player_name"],
            (row["search_rank"], row["vorp_rank"]),
            fontsize=7, fontstyle="italic", color="darkred",
            xytext=(5, -10), textcoords="offset points",
        )

    ax.set_xlabel("Sleeper Search Rank (ADP proxy)", fontsize=11)
    ax.set_ylabel("VORP Rank (1 = best)", fontsize=11)
    ax.set_title("Market Arbitrage: ADP vs. VORP", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(0, max_val)
    ax.set_ylim(max_val, 0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# -- 3. Positional cliffs ----------------------------------------------------
def render_positional_cliffs(
    board_df: pd.DataFrame,
    top_n: int = 120,
    output_path: str | None = None,
) -> plt.Figure:
    """Step-down line plots showing VORP drop-off per position.

    Parameters
    ----------
    board_df : Full VORP draft board.
    top_n : Restrict to the top *top_n* players by vorp_rank.
    output_path : Save path. Defaults to ``data/positional_cliffs.png``.

    Returns
    -------
    matplotlib Figure.
    """
    if output_path is None:
        output_path = "data/positional_cliffs.png"

    df = board_df[board_df["vorp_rank"] <= top_n].copy()

    fig, ax = plt.subplots(figsize=(14, 7))

    max_pos_rank = 0
    for pos, color in _POS_COLORS.items():
        pos_df = df[df["position_proj"] == pos].sort_values("pos_rank")
        if pos_df.empty:
            continue
        # Only plot players with positive VORP to avoid the massive flat tail
        pos_df_vorp = pos_df[pos_df["vorp"] > 0]
        if pos_df_vorp.empty:
            continue
        max_pos_rank = max(max_pos_rank, pos_df_vorp["pos_rank"].max())
        ax.step(
            pos_df_vorp["pos_rank"],
            pos_df_vorp["vorp"],
            where="post",
            label=pos,
            color=color,
            linewidth=2,
        )
        ax.scatter(
            pos_df_vorp["pos_rank"],
            pos_df_vorp["vorp"],
            color=color, s=25, zorder=3, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("Positional Rank", fontsize=11)
    ax.set_ylabel("VORP (Value Over Replacement)", fontsize=11)
    ax.set_title("Positional Value Cliffs -- Top 120 Players", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(left=0, right=max_pos_rank + 2 if max_pos_rank else 30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# -- 4. Styled HTML board ---------------------------------------------------
def export_styled_html_board(
    board_df: pd.DataFrame,
    output_path: str = "data/draft_board_styled.html",
) -> str:
    """Write a responsive standalone HTML file with colour-coded ADP delta.

    Parameters
    ----------
    board_df : Full VORP draft board.
    output_path : File path to write the HTML to.

    Returns
    -------
    The resolved *output_path*.
    """
    display_cols = [
        "vorp_rank",
        "pos_label",
        "player_name",
        "position_proj",
        "proj_points",
        "vorp",
        "search_rank",
        "adp_delta",
        "signal",
    ]
    cols_available = [c for c in display_cols if c in board_df.columns]
    out = board_df[cols_available].copy()

    # Ensure numeric for gradient
    for col in ["proj_points", "vorp", "search_rank", "adp_delta"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    def _adp_bg(val: float) -> str:
        """Return CSS background-color for an adp_delta value."""
        if pd.isna(val):
            return ""
        if val >= 10:
            return "background-color: #27AE60; color: white"  # strong green
        if val >= 5:
            return "background-color: #82E0AA"  # light green
        if val > -5:
            return ""
        if val > -10:
            return "background-color: #F9E79F"  # light yellow/red
        return "background-color: #E74C3C; color: white"  # strong red

    styler = out.style
    if "adp_delta" in out.columns:
        styler = styler.map(_adp_bg, subset=["adp_delta"])

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Fantasy Draft Board</title>\n"
        "<style>\n"
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, sans-serif; margin: 2rem; background: #f9f9f9; }\n"
        "h1 { color: #2C3E50; }\n"
        "table { border-collapse: collapse; width: 100%; background: white; "
        "box-shadow: 0 1px 3px rgba(0,0,0,0.12); }\n"
        "th { background: #2C3E50; color: white; padding: 10px 14px; "
        "text-align: left; font-size: 0.9rem; }\n"
        "td { padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 0.85rem; }\n"
        "tr:hover td { background: #eef6ff; }\n"
        "@media (max-width: 768px) { table { font-size: 0.75rem; } }\n"
        "</style>\n</head>\n<body>\n"
        "<h1>Fantasy Football Draft Board</h1>\n"
    )
    html += styler.to_html()
    html += "\n</body>\n</html>\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return output_path
