"""
Kraken public API client.

All endpoints return parsed, pandas-friendly data with robust error handling.
Uses st.cache_data with short TTLs to limit rate pressure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from .config import API_BASE_URL

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10


def _request(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    GET a Kraken public REST endpoint.

    Handles:
    - Timeouts (configurable)
    - Non-2xx responses (raise_for_status)
    - Kraken "error" array in JSON body
    - Invalid or non-JSON response

    Returns (result dict, error message or None).
    """
    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.get(
            url,
            params=params or None,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as e:
        logger.warning("Kraken request timeout: %s %s", endpoint, e)
        return {}, f"Request timeout ({timeout}s)"
    except requests.RequestException as e:
        logger.warning("Kraken request failed: %s %s", endpoint, e)
        return {}, str(e)
    except ValueError as e:
        logger.warning("Kraken invalid JSON: %s %s", endpoint, e)
        return {}, "Invalid response JSON"

    if not isinstance(data, dict):
        return {}, "Unexpected Kraken response format."

    errors = data.get("error") or []
    if errors:
        msg = "; ".join(str(e) for e in errors)
        logger.warning("Kraken API errors: %s %s", endpoint, msg)
        return {}, msg

    result = data.get("result")
    if not isinstance(result, dict):
        return {}, "Missing or invalid result field in Kraken response."

    return result, None


@st.cache_data(ttl=3600, show_spinner=False)
def get_asset_pairs(
    pair: Optional[str] = None,
    info: Optional[str] = None,
    country_code: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Get tradable asset pairs.

    GET https://api.kraken.com/0/public/AssetPairs

    Optional params:
    - pair: comma-separated pair names to filter
    - info: "info" | "leverage" | "fees" | "margin"
    - country_code: 2-letter country code filter

    Returns (result dict, error). result is keyed by pair name; each value
    is pair metadata (altname, wsname, base, quote, etc.). Pandas-friendly:
    pd.DataFrame.from_dict(result, orient='index').
    Cache TTL: 1 hour.
    """
    params: Dict[str, Any] = {}
    if pair is not None and str(pair).strip():
        params["pair"] = pair.strip()
    if info is not None and str(info).strip():
        params["info"] = info.strip()
    if country_code is not None and str(country_code).strip():
        params["country_code"] = country_code.strip()

    return _request("AssetPairs", params=params if params else None)


def _pair_for_rest(pair: str) -> str:
    """Kraken REST API expects pair without slash (e.g. XBTUSD, XBTUSDT)."""
    if not pair or not str(pair).strip():
        return ""
    return str(pair).strip().replace("/", "").upper()


@st.cache_data(ttl=60, show_spinner=False)
def get_ticker(pair: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Get ticker information for one or more pairs.

    GET https://api.kraken.com/0/public/Ticker?pair=

    pair: wsname (e.g. "XBT/USDT") or canonical pair key; normalized to REST format.
    Comma-separated for multiple pairs.

    Returns (result dict, error). result is keyed by pair id; each value
    has keys like c (last trade), v (volume), b (bid), a (ask), o (open).
    Cache TTL: 60 seconds.
    """
    if not pair or not str(pair).strip():
        return {}, "pair is required"
    rest_pair = _pair_for_rest(pair)
    if not rest_pair:
        return {}, "pair is required"
    return _request("Ticker", params={"pair": rest_pair})


@st.cache_data(ttl=120, show_spinner=False)
def get_ohlc(
    pair: str,
    interval: int,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Get OHLC candles.

    GET https://api.kraken.com/0/public/OHLC?pair=&interval=

    pair: wsname or pair key.
    interval: candle interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600).

    Returns (DataFrame, error). DataFrame columns: time, open, high, low,
    close, vwap, volume, count. time is datetime64. Empty DataFrame on error.
    Cache TTL: 120 seconds.
    """
    if not pair or not str(pair).strip():
        return pd.DataFrame(), "pair is required"

    raw, err = _request(
        "OHLC",
        params={"pair": pair.strip(), "interval": interval},
    )
    if err:
        return pd.DataFrame(), err

    if not raw:
        return pd.DataFrame(), "Empty OHLC result."

    try:
        _, rows = next(iter(raw.items()))
    except StopIteration:
        return pd.DataFrame(), "No OHLC rows in response."

    if not rows:
        return pd.DataFrame(), "No OHLC rows in response."

    columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "count",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, None


def _book_side_to_df(side: List[Any]) -> pd.DataFrame:
    """Convert raw book side [[price, vol, time], ...] to DataFrame."""
    if not side:
        return pd.DataFrame(columns=["price", "volume", "timestamp"])
    df = pd.DataFrame(side, columns=["price", "volume", "timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.dropna(subset=["price", "volume"], inplace=True)
    return df


@st.cache_data(ttl=20, show_spinner=False)
def get_order_book(
    pair: str,
    count: int = 100,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Get order book (depth).

    GET https://api.kraken.com/0/public/Depth?pair=&count=

    pair: wsname or pair key.
    count: number of levels per side (default 100).

    Returns (book, error). book has keys:
    - "bids": DataFrame with columns price, volume, timestamp
    - "asks": DataFrame with columns price, volume, timestamp
    Raw list form also under "bids_raw" / "asks_raw" for compatibility.
    Empty dict on error. Cache TTL: 20 seconds.
    """
    if not pair or not str(pair).strip():
        return {}, "pair is required"

    rest_pair = _pair_for_rest(pair)
    if not rest_pair:
        return {}, "pair is required"
    raw, err = _request(
        "Depth",
        params={"pair": rest_pair, "count": count},
    )
    if err:
        return {}, err

    if not raw:
        return {}, "Empty order book result."

    try:
        _, book = next(iter(raw.items()))
    except StopIteration:
        return {}, "No order book in response."

    if not isinstance(book, dict):
        return {}, "Unexpected order book format."

    bids = book.get("bids") or []
    asks = book.get("asks") or []

    return {
        "bids": _book_side_to_df(bids),
        "asks": _book_side_to_df(asks),
        "bids_raw": bids,
        "asks_raw": asks,
    }, None
