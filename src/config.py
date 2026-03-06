from __future__ import annotations

from typing import Dict, Tuple

API_BASE_URL = "https://api.kraken.com/0/public/"

# BTC symbol as used on Kraken spot markets
BTC_SYMBOL = "XBT"

# Stablecoins we care about for peg & liquidity monitoring
STABLECOINS = ("USDT", "USDC", "USD1", "USAT")

# Default BTC/stablecoin pair (wsname form)
DEFAULT_WSNAME = "XBT/USDT"

# --- Peg & mid price ---
# For stablecoin/USD pairs, target peg (e.g. 1.0). None = peg deviation disabled.
PEG_TARGET_STABLECOIN_USD: float = 1.0

# --- Depth bands (basis points around mid) ---
DEPTH_BAND_10_BPS: float = 10.0
DEPTH_BAND_25_BPS: float = 25.0
# Legacy / generic band (used where a single band is needed)
LIQUIDITY_DEPTH_BPS: float = 50.0

# --- Slippage proxy: hypothetical market order sizes (quote notional, e.g. USD) ---
SLIPPAGE_NOTIONAL_SIZES: Tuple[float, ...] = (50_000.0, 250_000.0, 1_000_000.0)
LIQUIDITY_SLIPPAGE_NOTIONAL: float = 100_000.0  # default single size

# --- Depeg event detection ---
DEPEG_THRESHOLD_BPS: float = 50.0
DEPEG_MIN_DURATION_MINUTES: float = 5.0

# --- Alerting thresholds ---
MAX_SPREAD_BPS: float = 20.0
MIN_DEPTH_USD: float = 250_000.0
MAX_SLIPPAGE_BPS: float = 30.0


METRIC_TOOLTIPS: Dict[str, str] = {
    "spread_bps": "Bid/ask spread at top of book, expressed in basis points relative to mid price.",
    "depth_usd": (
        "Cumulative notional (price × size) within ±10bp of mid. "
        "Bids: price >= mid×(1-10bp); asks: price <= mid×(1+10bp). Total = bid + ask."
    ),
    "depth_10bp": "Total USD notional within ±10bp of mid (both sides).",
    "depth_25bp": "Total USD notional within ±25bp of mid (both sides).",
    "slippage_bps": (
        "Estimated price impact (bps) for a simulated buy of the given notional "
        "walking the ask book. Configurable sizes in config (e.g. $50k, $250k, $1m)."
    ),
    "order_book_imbalance": (
        "(bid_depth - ask_depth) / (bid_depth + ask_depth) within the depth band. "
        "Range -1 to 1; positive = more bid liquidity."
    ),
    "volume_24h": (
        "24h traded volume from Kraken Ticker endpoint (quote terms)."
    ),
    "peg_deviation_bps": (
        "|mid - peg_target| / peg_target × 10,000. For stablecoin/USD, peg_target = 1.0. "
        "Disabled for non–stablecoin/USD unless peg_target is set."
    ),
    "depeg_events": (
        "Count of contiguous periods where peg deviation exceeded threshold for at least "
        "the minimum duration (threshold and duration configurable in config)."
    ),
}

