"""
Metric definitions: peg, spread, depth, depeg events, liquidity KPIs.

Inputs: ticker (bid, ask, last, volume), order book (bids/asks), OHLC (close).
All thresholds and order sizes are configurable in config.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    DEPEG_MIN_DURATION_MINUTES,
    DEPEG_THRESHOLD_BPS,
    DEPTH_BAND_10_BPS,
    DEPTH_BAND_25_BPS,
    LIQUIDITY_DEPTH_BPS,
    LIQUIDITY_SLIPPAGE_NOTIONAL,
    PEG_TARGET_STABLECOIN_USD,
    SLIPPAGE_NOTIONAL_SIZES,
)


def _to_dataframe(side: Any) -> pd.DataFrame:
    """Convert book side to DataFrame. Accepts list of [price, vol, ts] or existing DataFrame."""
    if isinstance(side, pd.DataFrame):
        return side.copy()
    if not side:
        return pd.DataFrame(columns=["price", "volume", "timestamp"])
    df = pd.DataFrame(side, columns=["price", "volume", "timestamp"])
    if df.empty:
        return df
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df.dropna(subset=["price", "volume"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Ticker parsing (Kraken: b=bid, a=ask, c=last trade close, v=volume)
# ---------------------------------------------------------------------------


def parse_ticker_for_mid(ticker: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract best bid, best ask, last, and 24h volume from Kraken ticker payload.
    Returns (best_bid, best_ask, last, volume_24h). Any can be None.
    """
    if not ticker:
        return None, None, None, None
    try:
        bid_str = (ticker.get("b") or [None])[0]
        ask_str = (ticker.get("a") or [None])[0]
        last_str = (ticker.get("c") or [None])[0]
        vol_str = (ticker.get("v") or [None, None])[1]
        best_bid = float(bid_str) if bid_str is not None else None
        best_ask = float(ask_str) if ask_str is not None else None
        last = float(last_str) if last_str is not None else None
        volume_24h = float(vol_str) if vol_str is not None else None
        return best_bid, best_ask, last, volume_24h
    except (TypeError, ValueError, IndexError):
        return None, None, None, None


def mid_price(bids_df: pd.DataFrame, asks_df: pd.DataFrame) -> Optional[float]:
    """Mid price = (best_bid + best_ask) / 2 from order book."""
    if bids_df.empty or asks_df.empty:
        return None
    best_bid = bids_df["price"].max()
    best_ask = asks_df["price"].min()
    if pd.isna(best_bid) or pd.isna(best_ask):
        return None
    return float(best_bid + best_ask) / 2.0


def spread_bps(best_bid: Optional[float], best_ask: Optional[float], mid: Optional[float]) -> Optional[float]:
    """Spread (bp) = (best_ask - best_bid) / mid * 10,000."""
    if mid is None or mid <= 0 or best_bid is None or best_ask is None or best_ask < best_bid:
        return None
    return (best_ask - best_bid) / mid * 10_000.0


def peg_deviation_bps(
    mid: Optional[float],
    peg_target: Optional[float] = PEG_TARGET_STABLECOIN_USD,
) -> Optional[float]:
    """
    Peg deviation (bp) = |mid - peg_target| / peg_target * 10,000.
    For stablecoin/USD use peg_target=1.0. If peg_target is None, returns None (disabled).
    """
    if peg_target is None or peg_target == 0 or mid is None:
        return None
    return abs(mid - peg_target) / peg_target * 10_000.0


def depth_at_bps(
    bids_df: pd.DataFrame,
    asks_df: pd.DataFrame,
    mid: Optional[float],
    bps: float,
) -> Dict[str, float]:
    """
    Cumulative notional within ±bps of mid.
    Bids: price >= mid * (1 - bps/10_000)
    Asks: price <= mid * (1 + bps/10_000)
    Returns dict: bid_notional, ask_notional, total (USD notional).
    """
    out = {"bid_notional": 0.0, "ask_notional": 0.0, "total": 0.0}
    if mid is None or mid <= 0 or (bids_df.empty and asks_df.empty):
        return out
    band = bps / 10_000.0
    low = mid * (1 - band)
    high = mid * (1 + band)
    if not bids_df.empty:
        bid_band = bids_df[bids_df["price"] >= low]
        out["bid_notional"] = float((bid_band["price"] * bid_band["volume"]).sum())
    if not asks_df.empty:
        ask_band = asks_df[asks_df["price"] <= high]
        out["ask_notional"] = float((ask_band["price"] * ask_band["volume"]).sum())
    out["total"] = out["bid_notional"] + out["ask_notional"]
    return out


def order_book_imbalance(
    bids_df: pd.DataFrame,
    asks_df: pd.DataFrame,
    mid: Optional[float],
    bps: float,
) -> Optional[float]:
    """
    (bid_depth - ask_depth) / (bid_depth + ask_depth) in notional terms within ±bps.
    Range -1 to 1; positive = more bid liquidity.
    """
    d = depth_at_bps(bids_df, asks_df, mid, bps)
    bid_n = d["bid_notional"]
    ask_n = d["ask_notional"]
    total = bid_n + ask_n
    if total == 0:
        return None
    return (bid_n - ask_n) / total


def compute_stress_score(
    spread_bp: Optional[float] = None,
    depth_10bp: Optional[float] = None,
    peg_deviation_bp: Optional[float] = None,
) -> float:
    """
    Composite stress score in [0.00, 1.00]: 0 = lowest stress, 1 = highest.
    Uses spread, depth, and peg deviation; result is clamped to [0, 1].
    """
    raw = 0.0
    # Spread: up to ~0.35 (e.g. 25 bp -> 0.35)
    if spread_bp is not None and spread_bp > 0:
        raw += min(0.35, (spread_bp / 100.0) * 1.4)
    # Depth: low depth -> higher stress
    if depth_10bp is not None:
        if depth_10bp <= 0:
            raw += 0.35
        elif depth_10bp < 100_000:
            raw += 0.28
        elif depth_10bp < 500_000:
            raw += 0.14
    # Peg deviation: up to ~0.30
    if peg_deviation_bp is not None and peg_deviation_bp > 0:
        raw += min(0.30, (abs(peg_deviation_bp) / 100.0) * 1.0)
    return min(1.0, max(0.0, raw))


def classify_liquidity_regime(
    stress_score: float,
) -> str:
    """
    Classify regime from normalized stress score [0, 1]:
    0.00–0.25 -> deep/stable, 0.25–0.50 -> normal, 0.50–0.75 -> tight, 0.75–1.00 -> stress.
    """
    s = max(0.0, min(1.0, stress_score))
    if s < 0.25:
        return "deep_stable"
    if s < 0.50:
        return "normal"
    if s < 0.75:
        return "tight"
    return "stress"


def slippage_bps_for_notional(
    asks_df: pd.DataFrame,
    notional: float,
    best_ask: Optional[float],
) -> Optional[float]:
    """
    Price impact (bps) for a hypothetical buy of `notional` walking the ask book.
    Uses (volume * price) cumsum to fill notional; then (fill_price - best_ask) / best_ask * 10_000.
    """
    if notional <= 0 or asks_df.empty or best_ask is None or best_ask <= 0:
        return None
    asks_sorted = asks_df.sort_values("price")
    asks_sorted = asks_sorted.copy()
    asks_sorted["notional"] = asks_sorted["price"] * asks_sorted["volume"]
    asks_sorted["cum_notional"] = asks_sorted["notional"].cumsum()
    filled = asks_sorted[asks_sorted["cum_notional"] >= notional].head(1)
    if filled.empty:
        return None
    level_price = float(filled["price"].iloc[0])
    return (level_price - best_ask) / best_ask * 10_000.0


# ---------------------------------------------------------------------------
# Depeg events from OHLC close series
# ---------------------------------------------------------------------------


def depeg_events(
    ohlc_df: pd.DataFrame,
    threshold_bps: float = DEPEG_THRESHOLD_BPS,
    min_duration_minutes: float = DEPEG_MIN_DURATION_MINUTES,
    peg_target: float = PEG_TARGET_STABLECOIN_USD,
    time_col: str = "time",
    close_col: str = "close",
) -> List[Dict[str, Any]]:
    """
    Count events where peg deviation exceeds threshold for a continuous duration.

    Uses OHLC close series. Expects ohlc_df with time_col (datetime) and close_col (numeric).
    Returns list of events: start_time, end_time, duration_minutes, max_deviation_bp.
    """
    events: List[Dict[str, Any]] = []
    if ohlc_df.empty or time_col not in ohlc_df.columns or close_col not in ohlc_df.columns:
        return events

    df = ohlc_df[[time_col, close_col]].copy()
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    if df.empty or peg_target == 0:
        return events

    df["deviation_bp"] = (df["close"] - peg_target).abs() / peg_target * 10_000.0
    df["above"] = df["deviation_bp"] > threshold_bps

    # Ensure time is datetime and sorted
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).reset_index(drop=True)

    in_run = False
    run_start: Optional[pd.Timestamp] = None
    run_devs: List[float] = []

    for _, row in df.iterrows():
        t = row[time_col]
        above = row["above"]
        dev = row["deviation_bp"]
        if above:
            if not in_run:
                in_run = True
                run_start = t
                run_devs = [dev]
            else:
                run_devs.append(dev)
        else:
            if in_run and run_start is not None:
                duration_min = (t - run_start).total_seconds() / 60.0
                if duration_min >= min_duration_minutes:
                    events.append({
                        "start_time": run_start,
                        "end_time": t,
                        "duration_minutes": round(duration_min, 2),
                        "max_deviation_bp": round(max(run_devs), 2),
                    })
                in_run = False
                run_start = None
                run_devs = []

    if in_run and run_start is not None:
        # run extends to end of series
        last_t = df[time_col].iloc[-1]
        duration_min = (last_t - run_start).total_seconds() / 60.0
        if duration_min >= min_duration_minutes:
            events.append({
                "start_time": run_start,
                "end_time": last_t,
                "duration_minutes": round(duration_min, 2),
                "max_deviation_bp": round(max(run_devs), 2),
            })

    return events


# ---------------------------------------------------------------------------
# Combined liquidity KPIs (Liquidity tab)
# ---------------------------------------------------------------------------


def compute_liquidity_kpis(
    order_book: Dict[str, Any],
    ticker: Dict[str, Any],
    notional: Optional[float] = None,
    depth_bps: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute liquidity KPIs from order book + ticker.

    Returns:
    - spread_bps
    - mid (from book)
    - depth_10bp_total, depth_10bp_bid, depth_10bp_ask
    - depth_25bp_total, depth_25bp_bid, depth_25bp_ask
    - slippage_bps_50k, slippage_bps_250k, slippage_bps_1m (keys by size from config)
    - order_book_imbalance_10bp, order_book_imbalance_25bp
    - volume_24h
    """
    default: Dict[str, Any] = {
        "spread_bps": None,
        "mid": None,
        "depth_10bp_total": None,
        "depth_10bp_bid": None,
        "depth_10bp_ask": None,
        "depth_25bp_total": None,
        "depth_25bp_bid": None,
        "depth_25bp_ask": None,
        "order_book_imbalance_10bp": None,
        "order_book_imbalance_25bp": None,
        "volume_24h": None,
        "slippage_bps": None,
    }
    for size in SLIPPAGE_NOTIONAL_SIZES:
        default[f"slippage_bps_{int(size)}"] = None

    if not order_book:
        return default

    bids_df = _to_dataframe(order_book.get("bids", []))
    asks_df = _to_dataframe(order_book.get("asks", []))

    if bids_df.empty or asks_df.empty:
        return default

    best_bid = float(bids_df["price"].max())
    best_ask = float(asks_df["price"].min())
    mid = mid_price(bids_df, asks_df)

    default["mid"] = mid
    default["spread_bps"] = spread_bps(best_bid, best_ask, mid)

    # Depth @ ±10bp and ±25bp
    d10 = depth_at_bps(bids_df, asks_df, mid, DEPTH_BAND_10_BPS)
    d25 = depth_at_bps(bids_df, asks_df, mid, DEPTH_BAND_25_BPS)
    default["depth_10bp_total"] = d10["total"]
    default["depth_10bp_bid"] = d10["bid_notional"]
    default["depth_10bp_ask"] = d10["ask_notional"]
    default["depth_25bp_total"] = d25["total"]
    default["depth_25bp_bid"] = d25["bid_notional"]
    default["depth_25bp_ask"] = d25["ask_notional"]

    # Legacy single depth/slippage for backward compatibility
    depth_bps_use = depth_bps if depth_bps is not None else LIQUIDITY_DEPTH_BPS
    d_legacy = depth_at_bps(bids_df, asks_df, mid, depth_bps_use)
    default["depth_usd"] = d_legacy["total"]
    notional_use = notional if notional is not None else LIQUIDITY_SLIPPAGE_NOTIONAL
    default["slippage_bps"] = slippage_bps_for_notional(asks_df, notional_use, best_ask)

    # Slippage for each configured size
    for size in SLIPPAGE_NOTIONAL_SIZES:
        default[f"slippage_bps_{int(size)}"] = slippage_bps_for_notional(asks_df, size, best_ask)

    default["order_book_imbalance_10bp"] = order_book_imbalance(bids_df, asks_df, mid, DEPTH_BAND_10_BPS)
    default["order_book_imbalance_25bp"] = order_book_imbalance(bids_df, asks_df, mid, DEPTH_BAND_25_BPS)

    _, _, _, vol = parse_ticker_for_mid(ticker)
    default["volume_24h"] = vol

    return default


# ---------------------------------------------------------------------------
# Peg + main KPI row metrics
# ---------------------------------------------------------------------------


def compute_peg_and_main_kpis(
    order_book: Dict[str, Any],
    ticker: Dict[str, Any],
    is_stablecoin_usd: bool = True,
    peg_target: Optional[float] = PEG_TARGET_STABLECOIN_USD,
) -> Dict[str, Any]:
    """
    Mid, spread, peg deviation, depth @ ±10bp (total), 24h volume.
    For peg deviation: use peg_target only if is_stablecoin_usd or peg_target is set.
    """
    out: Dict[str, Any] = {
        "mid": None,
        "spread_bps": None,
        "peg_deviation_bps": None,
        "depth_10bp_total": None,
        "volume_24h": None,
    }
    if not order_book:
        return out

    bids_df = _to_dataframe(order_book.get("bids", []))
    asks_df = _to_dataframe(order_book.get("asks", []))
    if bids_df.empty or asks_df.empty:
        return out

    best_bid = float(bids_df["price"].max())
    best_ask = float(asks_df["price"].min())
    mid = mid_price(bids_df, asks_df)

    out["mid"] = mid
    out["spread_bps"] = spread_bps(best_bid, best_ask, mid)

    pt = peg_target if (is_stablecoin_usd or peg_target is not None) else None
    out["peg_deviation_bps"] = peg_deviation_bps(mid, pt)

    d10 = depth_at_bps(bids_df, asks_df, mid, DEPTH_BAND_10_BPS)
    out["depth_10bp_total"] = d10["total"]

    _, _, _, vol = parse_ticker_for_mid(ticker)
    out["volume_24h"] = vol

    return out
