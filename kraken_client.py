import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


BASE_API_URL = "https://api.kraken.com/0/public/"


def _request(
    endpoint: str, params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Perform a GET request against a Kraken public endpoint."""
    url = f"{BASE_API_URL}{endpoint}"

    try:
        response = requests.get(url, params=params or None, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logging.warning("Kraken request to %s failed: %s", endpoint, exc)
        return {}, str(exc)

    if not isinstance(data, dict):
        return {}, "Unexpected response format from Kraken."

    errors = data.get("error") or []
    if errors:
        msg = "; ".join(errors)
        logging.warning("Kraken API %s returned errors: %s", endpoint, msg)
        return {}, msg

    result = data.get("result")
    if not isinstance(result, dict):
        return {}, "Missing result field in Kraken response."

    return result, None


@st.cache_data(ttl=600)
def get_asset_pairs() -> Tuple[Dict[str, Any], Optional[str]]:
    """Fetch available asset pairs from Kraken's public API."""
    return _request("AssetPairs")


@st.cache_data(ttl=30)
def get_ticker(pair: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Fetch ticker information for a specific pair.

    `pair` typically uses the altname form, e.g. "XBTUSD".
    """
    return _request("Ticker", params={"pair": pair})


@st.cache_data(ttl=120)
def get_ohlc(pair: str, interval: int) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch OHLC candles for a specific pair and interval.

    Returns a DataFrame with columns:
    time, open, high, low, close, vwap, volume, count
    or an empty DataFrame on failure.
    """
    raw, err = _request("OHLC", params={"pair": pair, "interval": interval})
    if err:
        return pd.DataFrame(), err

    if not raw:
        return pd.DataFrame(), "Empty OHLC result."

    # OHLC returns {pair_name: [[time, open, high, low, close, vwap, volume, count], ...]}
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
    # Convert timestamp to datetime for display and charting
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df, None

