"""
Multi-pair metrics engine: snapshot per pair for cross-pair liquidity summary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from .config import STABLECOINS
from .kraken_client import get_order_book, get_ticker
from .metrics import (
    _to_dataframe,
    classify_liquidity_regime,
    compute_stress_score,
    depth_at_bps,
    mid_price,
    parse_ticker_for_mid,
    peg_deviation_bps,
    spread_bps,
)


def _is_stablecoin_usd(pair_wsname: str) -> bool:
    """True if pair is base/USD with base in STABLECOINS."""
    if "/" not in pair_wsname:
        return False
    base, quote = pair_wsname.strip().upper().split("/", 1)
    return quote == "USD" and base in STABLECOINS


@st.cache_data(ttl=25, show_spinner=False)
def compute_pair_snapshot(pair_wsname: str) -> Dict[str, Any]:
    """
    Compute a liquidity snapshot for one pair. Uses Kraken Depth + Ticker.

    Returns:
        pair, mid, spread_bp, depth_10bp, volume_24h, peg_deviation_bp (or None),
        stress_score, liquidity_regime.
    """
    out: Dict[str, Any] = {
        "pair": pair_wsname,
        "mid": None,
        "spread_bp": None,
        "depth_10bp": None,
        "volume_24h": None,
        "peg_deviation_bp": None,
        "stress_score": 0.0,
        "liquidity_regime": "normal",
    }
    order_book, ob_err = get_order_book(pair_wsname)
    ticker_result, ticker_err = get_ticker(pair_wsname)
    if ob_err or not order_book or ticker_err or not ticker_result:
        return out

    first_payload = next(iter(ticker_result.values()), {}) or {}
    best_bid_t, best_ask_t, last_t, volume_24h = parse_ticker_for_mid(first_payload)
    out["volume_24h"] = volume_24h

    bids_df = _to_dataframe(order_book.get("bids", []))
    asks_df = _to_dataframe(order_book.get("asks", []))
    if bids_df.empty or asks_df.empty:
        return out

    mid = mid_price(bids_df, asks_df)
    best_bid = float(bids_df["price"].max())
    best_ask = float(asks_df["price"].min())
    if mid is None and last_t is not None:
        mid = last_t

    out["mid"] = mid
    out["spread_bp"] = spread_bps(best_bid, best_ask, mid)

    d10 = depth_at_bps(bids_df, asks_df, mid, 10.0)
    total_val = d10.get("total")
    out["depth_10bp"] = total_val if total_val is not None else None

    if _is_stablecoin_usd(pair_wsname) and mid is not None:
        out["peg_deviation_bp"] = peg_deviation_bps(mid, 1.0)

    out["stress_score"] = compute_stress_score(
        spread_bp=out["spread_bp"],
        depth_10bp=out["depth_10bp"],
        peg_deviation_bp=out["peg_deviation_bp"],
    )
    out["liquidity_regime"] = classify_liquidity_regime(out["stress_score"])
    return out
