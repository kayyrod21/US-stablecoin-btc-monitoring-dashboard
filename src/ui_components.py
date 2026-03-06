from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import METRIC_TOOLTIPS, SLIPPAGE_NOTIONAL_SIZES
from .metrics import _to_dataframe, depth_at_bps


def _fmt(value: Optional[float], pattern: str) -> str:
    if value is None:
        return "—"
    try:
        return pattern.format(value)
    except Exception:
        return "—"


def _format_compact(value: Optional[float]) -> str:
    """Compact formatter for depth/volume: 9.6M, 158.3K."""
    if value is None:
        return "—"
    try:
        v = float(value)
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v / 1_000:.1f}K"
        return f"{v:,.0f}"
    except Exception:
        return "—"


def _fmt_bp_small(value: Optional[float]) -> str:
    """Format bp; show '<0.1' when nonzero but rounds to 0.0."""
    if value is None:
        return "—"
    try:
        v = float(value)
        if v != 0 and round(v, 1) == 0.0:
            return "<0.1"
        return f"{v:.1f}"
    except Exception:
        return "—"


def _tooltip(metric_key: str) -> None:
    text = METRIC_TOOLTIPS.get(metric_key)
    if not text:
        return
    st.caption(
        f"<span title='{text}'>ℹ️ Hover for metric details</span>",
        unsafe_allow_html=True,
    )


def _short_notional_label(size: float) -> str:
    """Labels: $50k, $250k, $1M."""
    if size >= 1_000_000:
        return f"${size/1e6:.0f}M"
    if size >= 1_000:
        return f"${size/1e3:.0f}k"
    return f"${size:.0f}"


def _fmt_slippage_bp(val: Optional[float]) -> str:
    """Human-readable slippage (bp): compact if large, em dash if unavailable or nonsensical."""
    if val is None or (isinstance(val, float) and (val < 0 or val != val)):
        return "—"
    try:
        v = float(val)
        if v > 999:
            return f"{v/1e3:.1f}k"
        if v > 99:
            return f"{v:.0f}"
        return _fmt_bp_small(v)
    except Exception:
        return "—"


def get_liquidity_wall_figure(
    order_book: Dict[str, Any],
    mid: float,
    height: int = 340,
    wide_view: bool = False,
    is_majors: bool = False,
):
    """Build and return Plotly figure for liquidity wall (depth vs distance from mid). Returns None if no data."""
    bids_df = _to_dataframe(order_book.get("bids", []))
    asks_df = _to_dataframe(order_book.get("asks", []))
    if bids_df.empty or asks_df.empty or mid is None or mid <= 0:
        return None
    bids_df = bids_df.copy()
    asks_df = asks_df.copy()
    bids_df["notional"] = bids_df["price"] * bids_df["volume"]
    asks_df["notional"] = asks_df["price"] * asks_df["volume"]
    # Bids on left (negative bp); asks on right (positive bp)
    bids_df["distance_bp"] = -1.0 * ((mid - bids_df["price"]) / mid * 10_000.0)
    asks_df["distance_bp"] = (asks_df["price"] - mid) / mid * 10_000.0
    bids_df = bids_df[bids_df["distance_bp"] <= 0].sort_values("distance_bp")
    asks_df = asks_df[asks_df["distance_bp"] >= 0].sort_values("distance_bp")
    bid_cum = bids_df["notional"].cumsum()
    ask_cum = asks_df["notional"].cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bids_df["distance_bp"].tolist(),
            y=bid_cum.tolist(),
            mode="lines",
            name="Bid Support",
            line=dict(color="#1f77b4", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.22)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=asks_df["distance_bp"].tolist(),
            y=ask_cum.tolist(),
            mode="lines",
            name="Ask Resistance",
            line=dict(color="#d62728", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.22)",
        )
    )
    # Midline at 0 bp: light gray, dot dash (no annotation to reduce clutter)
    fig.add_vline(x=0, line_dash="dot", line_color="#9CA3AF", line_width=1)

    # Adaptive y-axis for thin books: base range on near-peg (±100bp) cumulative depth
    y_upper = None
    if not wide_view:
        bid_mask = bids_df["distance_bp"] >= -100
        ask_mask = asks_df["distance_bp"] <= 100
        y_bid = float(bid_cum[bid_mask].max()) if bid_mask.any() else 0
        y_ask = float(ask_cum[ask_mask].max()) if ask_mask.any() else 0
        y_near_peg_max = max(y_bid, y_ask)
        if y_near_peg_max > 0:
            y_upper = y_near_peg_max * 1.15
        else:
            y_full = max(bid_cum.max() if len(bid_cum) else 0, ask_cum.max() if len(ask_cum) else 0)
            y_upper = max(float(y_full) * 0.2, 1000)

    title_text = "Order Book Depth Profile" if is_majors else "Liquidity Wall (Depth vs Distance from Mid)"
    subtitle_text = "Cumulative bid/ask depth relative to mid price (±100bp view by default)" if is_majors else "Near-peg liquidity (±100bp view by default)"
    layout = dict(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=13),
        ),
        annotations=[
            dict(
                text=subtitle_text,
                x=0.5,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=9, color="#9CA3AF"),
            )
        ],
        xaxis_title="Distance from Mid (bp)",
        yaxis_title="Cumulative Notional (USD)",
        height=height,
        margin=dict(l=10, r=10, t=52, b=10),
        legend=dict(orientation="h", yanchor="top", y=1.14, xanchor="right", x=1, font=dict(size=10)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1F2937", size=12),
        xaxis=dict(zeroline=False, gridcolor="#E5E7EB", linecolor="#E5E7EB"),
        yaxis=dict(
            zeroline=False,
            gridcolor="#E5E7EB",
            linecolor="#E5E7EB",
            tickformat=",.0s",
        ),
    )
    if not wide_view:
        layout["xaxis"]["range"] = [-100, 100]
    if y_upper is not None and y_upper > 0:
        layout["yaxis"]["range"] = [0, y_upper]
    fig.update_layout(**layout)
    return fig


def _liquidity_wall_chart(
    order_book: Dict[str, Any],
    mid: float,
    wide_view: bool = False,
    is_majors: bool = False,
) -> None:
    """Plot cumulative depth vs distance from mid (bp): bids and asks."""
    fig = get_liquidity_wall_figure(order_book, mid, wide_view=wide_view, is_majors=is_majors)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key="liquidity_wall_tab_chart")
        st.caption("Cumulative bid/ask depth relative to mid price." if is_majors else "Negative values show bid-side depth; positive values show ask-side depth.")


def _render_depth_heatmap(cross_pair_df: pd.DataFrame, is_majors: bool = False) -> None:
    """
    Render Plotly heatmap: rows=pairs, columns=depth bands (±5bp, ±10bp, ±25bp, ±50bp).
    Uses Blues colorscale; darker = deeper. Overlays compact numeric values.
    Expects cross_pair_df with Pair and numeric Depth ±Nbp columns.
    """
    if cross_pair_df.empty:
        return
    band_cols = ["Depth ±5bp", "Depth ±10bp", "Depth ±25bp", "Depth ±50bp"]
    available = [c for c in band_cols if c in cross_pair_df.columns]
    if not available:
        return
    pair_col = "Pair" if "Pair" in cross_pair_df.columns else cross_pair_df.columns[0]
    pairs = cross_pair_df[pair_col].astype(str).tolist()
    z = cross_pair_df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    text_arr = [[_format_compact(v) if v and v > 0 else "—" for v in row] for row in z.values]
    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=available,
            y=pairs,
            colorscale="Blues",
            text=text_arr,
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
            colorbar=dict(title="Depth (USD)"),
        )
    )
    title_main = "Near-Mid Liquidity Depth" if is_majors else "Near-Peg Liquidity Depth Comparison"
    title_sub = "Darker cells = deeper cumulative depth."
    fig.update_layout(
        title=dict(
            text=f"{title_main}<br><sub style='font-size:10px;color:#9CA3AF'>{title_sub}</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=12),
        ),
        height=min(260, 72 + len(pairs) * 46),
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1F2937", size=11),
        xaxis=dict(side="bottom", tickangle=-25),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, key="overview_depth_heatmap")


def _liquidity_summary_table(
    order_book: Dict[str, Any],
    mid: float,
) -> None:
    """Table: Band (±10bp, ±25bp, ±50bp, ±100bp), Bid Depth, Ask Depth, Total Depth, Imbalance."""
    bids_df = _to_dataframe(order_book.get("bids", []))
    asks_df = _to_dataframe(order_book.get("asks", []))
    if bids_df.empty or asks_df.empty or mid is None or mid <= 0:
        return
    rows = []
    for bps, band_label in [(10.0, "±10bp"), (25.0, "±25bp"), (50.0, "±50bp"), (100.0, "±100bp")]:
        d = depth_at_bps(bids_df, asks_df, mid, bps)
        bid_n = d["bid_notional"]
        ask_n = d["ask_notional"]
        total = d["total"]
        imb = (bid_n - ask_n) / total if total else None
        rows.append({
            "Band": band_label,
            "Bid Depth": _format_compact(bid_n),
            "Ask Depth": _format_compact(ask_n),
            "Total Depth": _format_compact(total),
            "Imbalance": f"{imb:.2%}" if imb is not None else "—",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=min(160, 40 + len(rows) * 36), hide_index=True)


def render_liquidity_kpis(
    kpis: Dict[str, Any],
    order_book: Optional[Dict[str, Any]] = None,
    mid: Optional[float] = None,
    is_majors: bool = False,
) -> None:
    """
    Liquidity tab: Row A strip, Row B wall, Row C (summary table | slippage cards), Row D 24h volume.
    """
    # LIQUIDITY ROW A — Compact strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Spread (bp)", _fmt_bp_small(kpis.get("spread_bps")))
    with c2:
        depth_10 = kpis.get("depth_10bp_total") or kpis.get("depth_usd")
        st.metric("Depth ±10bp", _format_compact(depth_10))
    with c3:
        st.metric("Depth ±25bp", _format_compact(kpis.get("depth_25bp_total")))
    with c4:
        imb = kpis.get("order_book_imbalance_10bp")
        st.metric("Order Book Imbalance", _fmt(imb, "{:.2f}") if imb is not None else "—")

    # LIQUIDITY ROW B — Main Liquidity Wall (Order Book Depth Profile for Majors)
    if order_book and mid is not None and mid > 0:
        _liquidity_wall_chart(order_book, mid, is_majors=is_majors)

    # LIQUIDITY ROW C — Band Summary (left) + Slippage mini cards (right)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("**Liquidity Summary**")
        if order_book and mid is not None and mid > 0:
            _liquidity_summary_table(order_book, mid)
    with col_right:
        st.markdown('<span class="k-slippage-compact" aria-hidden="true"></span>', unsafe_allow_html=True)
        st.markdown("**Estimated Slippage (bp)**")
        for size in SLIPPAGE_NOTIONAL_SIZES:
            key = f"slippage_bps_{int(size)}"
            val = kpis.get(key)
            st.metric(_short_notional_label(size), _fmt_slippage_bp(val))

    # LIQUIDITY ROW D — 24h Volume (small)
    st.metric("24h Volume", _format_compact(kpis.get("volume_24h")))

