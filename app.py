from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from kraken_client import get_asset_pairs, get_ohlc, get_ticker
from src.audit import (
    audit_log_append,
    EVENT_ALERT_CLEARED,
    EVENT_ALERT_TRIGGERED,
    EVENT_API_ERROR,
    EVENT_PAIR_CHANGE,
    EVENT_SURVEILLANCE,
    EVENT_THRESHOLD_CHANGE,
    get_audit_log,
)
from src.config import (
    LIQUIDITY_DEPTH_BPS,
    LIQUIDITY_SLIPPAGE_NOTIONAL,
    STABLECOINS,
)
from src.kraken_client import get_order_book
from src.metrics import (
    _to_dataframe,
    compute_liquidity_kpis,
    depth_at_bps,
    mid_price,
    order_book_imbalance,
    parse_ticker_for_mid,
    peg_deviation_bps,
    spread_bps,
)
from src.peg_events import detect_depeg_events
from src.snapshot import compute_pair_snapshot
from src.ui_components import _render_depth_heatmap, get_liquidity_wall_figure, render_liquidity_kpis

# In-scope pairs for cross-pair liquidity summary
STABLECOIN_USD_PAIRS = ["USDC/USD", "USDT/USD", "USAT/USD", "USD1/USD"]
CROSS_PAIR_SCOPE = STABLECOIN_USD_PAIRS + ["XBT/USD", "XBT/USDT"]

# Majors: user-facing labels -> Kraken exchange symbols (API uses exchange only)
MAJORS_DISPLAY_TO_EXCHANGE = {"BTC": "XBT", "Gold": "XAU"}
MAJORS_EXCHANGE_TO_DISPLAY = {"XBT": "BTC", "XAU": "Gold"}


def format_compact_number(v: Optional[float]) -> str:
    """Format large numbers as 13.2M, 158.3K; peg/spread stay numeric elsewhere."""
    if v is None:
        return "—"
    try:
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v / 1_000:.1f}K"
        return f"{v:,.0f}"
    except Exception:
        return "—"


def format_percent_like_bp(v: Optional[float]) -> str:
    """Format basis points for display (e.g. 12.5)."""
    if v is None:
        return "—"
    try:
        return f"{v:.1f}"
    except Exception:
        return "—"


def format_bp_with_small(v: Optional[float]) -> str:
    """Format bp for display; show '<0.1' when nonzero but rounds to 0.0."""
    if v is None:
        return "—"
    try:
        if v != 0 and round(v, 1) == 0.0:
            return "<0.1"
        return f"{v:.1f}"
    except Exception:
        return "—"


def _market_quality_label(
    stress_score: Optional[float],
    spread_bp: Optional[float],
    depth_10bp: Optional[float],
) -> str:
    """Market Quality classification: ORDERLY, MONITOR, or ELEVATED RISK (display only)."""
    stress = stress_score if stress_score is not None else 1.0
    spread = spread_bp if spread_bp is not None else 999.0
    depth = depth_10bp if depth_10bp is not None else 0.0
    if stress > 0.5 or spread > 10 or depth < 250_000:
        return "ELEVATED RISK"
    if stress < 0.25 and spread < 5 and depth > 1_000_000:
        return "ORDERLY"
    return "MONITOR"


def _format_regime_label(regime: Optional[str]) -> str:
    """Display label for liquidity regime: DEEP / STABLE, NORMAL, TIGHT, STRESS."""
    if not regime:
        return "—"
    r = regime.strip().lower()
    if r == "deep_stable":
        return "DEEP / STABLE"
    if r in ("normal", "tight", "stress"):
        return r.upper()
    return regime.upper()


def build_market_summary_text(
    peg_dev_bp: Optional[float],
    spread_bp: Optional[float],
    depth_10bp: Optional[float],
    stress_score: Optional[float],
    regime: Optional[str] = None,
) -> str:
    """One-line interpretation for stablecoin/USD; narrative matches score/regime."""
    regime = (regime or "").lower()
    if regime == "deep_stable":
        return "Stablecoin market appears liquid and tightly pegged."
    if regime == "normal":
        return "Market conditions appear stable with moderate liquidity."
    if regime == "tight":
        return "Liquidity is thinning or spread is widening; monitor more closely."
    if regime == "stress":
        return "Conditions indicate elevated peg and liquidity stress."
    # Fallback from score if regime not passed
    stress = stress_score if stress_score is not None else 0.0
    if stress < 0.25:
        return "Stablecoin market appears liquid and tightly pegged."
    if stress < 0.50:
        return "Market conditions appear stable with moderate liquidity."
    if stress < 0.75:
        return "Liquidity is thinning or spread is widening; monitor more closely."
    return "Conditions indicate elevated peg and liquidity stress."

# Session state keys for alerts and audit
ALERTS_DEPEG_KEY = "kraken_alerts_depeg"  # dict: (pair_wsname, timeframe) -> list of event dicts
ALERTS_LIQUIDITY_KEY = "kraken_alerts_liquidity"  # list of {pair, message, ts}
ALERTS_SHOW_ALL_KEY = "kraken_alerts_show_all"
DEPTH_WARNING_FIRED_KEY = "kraken_depth_warning_fired"  # set of pair_wsname
LAST_SELECTIONS_KEY = "kraken_last_selections"

DEFAULT_PAIR: Tuple[str, str] = ("BTC", "USD")


# Kraken public REST is wired: AssetPairs, Ticker, OHLC, Depth (see kraken_client + src.kraken_client).
# Peg deviation applies to stablecoin/USD only; BTC pairs show "—".


def configure_page() -> None:
    """Set base Streamlit page configuration and lightweight CSS."""
    st.set_page_config(
        page_title="US Stablecoin Monitoring Dashboard",
        layout="wide",
    )

    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1200px;
          }

          [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
          }

          [data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
          }

          .k-card {
            background: #ffffff;
            border-radius: 6px;
            border: 1px solid #E5E7EB;
            box-shadow: none;
            padding: 12px 14px;
            margin-bottom: 0.75rem;
          }

          .k-title {
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin: 0;
            color: #1F2937;
          }

          .k-subtitle {
            font-size: 12px;
            color: #6B7280;
            margin-top: 2px;
            margin-bottom: 0;
          }

          .k-section-title {
            font-size: 12px;
            font-weight: 500;
            margin: 0 0 6px 0;
            color: #374151;
          }
          .k-section-title.k-section-spaced {
            margin-top: 0.5rem;
          }

          .k-pill-up {
            display: inline-flex;
            align-items: center;
            padding: 2px 6px;
            border-radius: 4px;
            background: transparent;
            color: #6B7280;
            font-size: 10px;
            font-weight: 500;
          }

          .k-pill-status {
            display: inline-flex;
            align-items: center;
            padding: 2px 6px;
            border-radius: 4px;
            background: #F3F4F6;
            color: #6B7280;
            font-size: 10px;
            font-weight: 500;
          }

          .k-depeg-badge {
            display: inline-block;
            font-size: 11px;
            color: #DC2626;
            font-weight: 500;
            padding: 1px 0;
          }

          [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.75rem;
          }

          /* Primary KPI strip: compact */
          [data-testid="stMetric"] label {
            color: #6B7280 !important;
            font-weight: 500 !important;
            font-size: 11px !important;
          }
          [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1F2937 !important;
            font-size: 15px !important;
          }

          /* Supporting strip: even smaller */
          .k-supporting-strip-marker + div [data-testid="stMetric"] label {
            font-size: 10px !important;
            color: #9CA3AF !important;
            font-weight: 400 !important;
          }
          .k-supporting-strip-marker + div [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 10px !important;
            color: #6B7280 !important;
            word-break: break-word;
            max-width: 100%;
          }
          .k-supporting-strip-marker + div [data-testid="column"] {
            padding-top: 2px;
            padding-bottom: 2px;
          }

          .k-strip {
            display: flex;
            gap: 1rem;
            padding: 8px 0;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
          }

          .k-compact-card {
            background: #FAFAFA;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            padding: 8px 12px;
            min-width: 0;
          }

          div[data-testid="stDataFrame"] {
            font-size: 12px;
          }
          div[data-testid="stDataFrame"] th {
            background: #F9FAFB !important;
            font-weight: 500 !important;
            color: #6B7280 !important;
            font-size: 11px !important;
          }
          div[data-testid="stDataFrame"] td {
            font-size: 12px !important;
          }

          [data-testid="stExpander"] {
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
          }
          [data-testid="stExpander"] summary {
            padding: 0.2rem 0;
            font-size: 11px !important;
          }
          [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
            padding-top: 0.25rem;
          }
          .k-alerts-inner {
            padding: 0.25rem 0;
          }
          /* Liquidity tab: compact slippage metrics (sibling of marker in same column) */
          .k-slippage-compact ~ [data-testid="stMetric"] label { font-size: 10px !important; }
          .k-slippage-compact ~ [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 13px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_base_symbol(symbol: str) -> str:
    """Normalize base asset symbol to Kraken's conventions where needed (for API resolution)."""
    u = symbol.upper()
    if u == "BTC":
        return "XBT"
    if u == "GOLD":
        return "XAU"
    return u


def _display_pair_for_ui(wsname: Optional[str]) -> str:
    """Convert exchange wsname to user-facing pair label; no-op for non-Majors pairs."""
    if not wsname or "/" not in wsname:
        return wsname or "—"
    base_ws, quote = wsname.strip().upper().split("/", 1)
    display_base = MAJORS_EXCHANGE_TO_DISPLAY.get(base_ws)
    if display_base is None:
        return wsname
    return f"{display_base}/{quote}"


def _display_pair_to_exchange_wsname(display_pair: str) -> str:
    """Convert user-facing pair (e.g. BTC/USD, Gold/USD) to Kraken wsname for API."""
    if not display_pair or "/" not in display_pair:
        return display_pair
    base_display, quote = display_pair.strip().split("/", 1)
    base_u = base_display.upper()
    exchange_base = MAJORS_DISPLAY_TO_EXCHANGE.get(base_u) if base_u in MAJORS_DISPLAY_TO_EXCHANGE else base_u
    return f"{exchange_base}/{quote.upper()}"


def _pair_for_rest(wsname: Optional[str]) -> str:
    """Kraken REST API expects pair without slash (e.g. XBTUSD). Used for root kraken_client calls."""
    if not wsname or not str(wsname).strip():
        return ""
    return str(wsname).strip().replace("/", "").upper()


def resolve_pair_code(
    asset_pairs: Dict[str, Any],
    base: str,
    quote: str,
) -> str | None:
    """
    Determine the correct Kraken pair identifier from AssetPairs metadata.

    Resolution order:
    1) Direct altname match (e.g. BTC/USD -> XBTUSD)
    2) wsname match (e.g. XBT/USD)
    3) Match normalized base/quote fields (strip leading X/Z)
    """
    if not asset_pairs:
        return None

    base_u = base.upper()
    quote_u = quote.upper()

    # 1) altname match
    base_norm_alt = _normalize_base_symbol(base_u)
    target_altname = f"{base_norm_alt}{quote_u}"
    for pair_key, meta in asset_pairs.items():
        altname = (meta or {}).get("altname")
        if isinstance(altname, str) and altname.upper() == target_altname:
            return pair_key

    # 2) wsname match (e.g. XBT/USD)
    target_wsname = f"{base_norm_alt}/{quote_u}"
    for pair_key, meta in asset_pairs.items():
        wsname = (meta or {}).get("wsname")
        if isinstance(wsname, str) and wsname.upper() == target_wsname:
            return pair_key

    # 3) match by normalized base/quote fields
    def _normalize_field(sym: Optional[str]) -> str:
        if not sym:
            return ""
        s = sym.upper()
        if s.startswith(("X", "Z")) and len(s) > 3:
            s = s[1:]
        if s == "XBT":
            s = "BTC"
        return s

    target_base = "BTC" if base_u == "BTC" else base_u
    target_quote = quote_u

    for pair_key, meta in asset_pairs.items():
        if not isinstance(meta, dict):
            continue
        base_field = _normalize_field(meta.get("base"))
        quote_field = _normalize_field(meta.get("quote"))
        if base_field == target_base and quote_field == target_quote:
            return pair_key

    return None


def _map_timeframe_to_interval(timeframe: str, hi_res: bool = False) -> int:
    """
    Map UI time window to Kraken OHLC interval (minutes).
    - 1D + 5m granularity -> 5; else 1D -> 15.
    - 1W -> 60, 1M -> 240, 3M -> 1440.
    """
    if timeframe == "1D" and hi_res:
        return 5
    mapping = {
        "1D": 15,
        "1W": 60,
        "1M": 240,
        "3M": 1440,
    }
    return mapping.get(timeframe, 60)


def _get_active_chart_mode(timeframe: str, hi_res: bool) -> Tuple[int, str, bool]:
    """
    Active chart mode for OHLC fetch and chart horizon.
    When 5m (1D) is checked: force 5m interval and last 24h window (intraday).
    When unchecked: use radio timeframe and default interval for that window.
    Returns (active_interval_minutes, window_label, is_intraday_24h).
    """
    if hi_res:
        return 5, "last 24h", True
    interval = _map_timeframe_to_interval(timeframe, False)
    return interval, timeframe, False


def _trim_ohlc_to_last_24h(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Trim OHLC DataFrame to last 24 hours (by 'time' column). Returns trimmed copy."""
    if ohlc_df.empty or "time" not in ohlc_df.columns:
        return ohlc_df
    now = pd.Timestamp.utcnow()
    cutoff = now - pd.Timedelta(hours=24)
    time_vals = ohlc_df["time"]
    if getattr(time_vals.dt, "tz", None) is None:
        cutoff = cutoff.tz_localize(None) if cutoff.tz is not None else cutoff
    else:
        cutoff = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff
    return ohlc_df.loc[ohlc_df["time"] >= cutoff].copy()


def _prepare_ticker_metrics(ticker: Dict[str, Any]) -> Dict[str, str]:
    """Compute human-readable KPI values from Kraken ticker payload."""
    metrics = {
        "price": "$ —",
        "volume": "—",
        "change": "—",
        "spread": "—",
    }

    if not ticker:
        return metrics

    try:
        last_str = (ticker.get("c") or [None])[0]
        open_str = ticker.get("o")
        bid_str = (ticker.get("b") or [None])[0]
        ask_str = (ticker.get("a") or [None])[0]
        vol_24h_str = (ticker.get("v") or [None, None])[1]

        last = float(last_str) if last_str is not None else None
        open_price = float(open_str) if open_str is not None else None
        bid = float(bid_str) if bid_str is not None else None
        ask = float(ask_str) if ask_str is not None else None
        vol_24h = float(vol_24h_str) if vol_24h_str is not None else None

        if last is not None:
            metrics["price"] = f"${last:,.2f}"

        if vol_24h is not None:
            metrics["volume"] = f"{vol_24h:,.0f}"

        if last is not None and open_price not in (None, 0):
            diff = last - open_price
            pct = diff / open_price * 100
            metrics["change"] = f"{diff:+.2f} ({pct:+.2f}%)"

        if bid is not None and ask is not None:
            metrics["spread"] = f"{ask - bid:.2f}"
    except Exception:
        # On any parsing error, fall back to placeholders.
        return metrics

    return metrics


def _build_pairs_dataframe(
    asset_pairs: Dict[str, Any],
    base_asset: str,
    quote_asset: str,
) -> pd.DataFrame:
    """
    Build a small DataFrame of pairs matching the selected base/quote.
    Falls back to top-N pairs if there is no direct match.
    """
    if not asset_pairs:
        return pd.DataFrame()

    base_norm = _normalize_base_symbol(base_asset)
    quote_norm = quote_asset.upper()
    target_altname = f"{base_norm}{quote_norm}"

    rows = []
    for pair_name, meta in asset_pairs.items():
        if not isinstance(meta, dict):
            continue
        altname = meta.get("altname", "")
        if altname == target_altname or (
            altname.endswith(quote_norm) and base_norm in altname
        ):
            rows.append(
                {
                    "pair": pair_name,
                    "altname": altname,
                    "wsname": meta.get("wsname"),
                    "base": meta.get("base"),
                    "quote": meta.get("quote"),
                    "status": meta.get("status"),
                }
            )

    if not rows:
        # Fallback: show a small sample of tradable pairs.
        for pair_name, meta in list(asset_pairs.items())[:20]:
            if not isinstance(meta, dict):
                continue
            rows.append(
                {
                    "pair": pair_name,
                    "altname": meta.get("altname"),
                    "wsname": meta.get("wsname"),
                    "base": meta.get("base"),
                    "quote": meta.get("quote"),
                    "status": meta.get("status"),
                }
            )

    return pd.DataFrame(rows)


def _derive_group_pairs(
    asset_pairs: Dict[str, Any],
    asset_group: str,
    quote_asset: str,
) -> Tuple[list[str], Dict[str, list[str]]]:
    """
    From AssetPairs, derive options for the selected group and quote.
    - Stablecoins: base and pair options use exchange symbols (e.g. USDC/USD).
    - Majors: BTC and Gold only; base and pair options use display labels (BTC, Gold, BTC/USD, Gold/USD).

    Returns:
    - sorted list of base asset symbols (display form for Majors)
    - mapping base_symbol -> list of pair strings (display form for Majors, wsname for Stablecoins)
    """
    if not asset_pairs:
        return [], {}

    quote_u = quote_asset.upper()
    bases: set[str] = set()
    pairs_by_base: Dict[str, list[str]] = {}

    if asset_group == "Majors":
        # Majors: BTC (XBT) and Gold (XAU) only if Kraken metadata contains them
        allowed_exchange_bases = {"XBT", "XAU"}
        online_by_base: Dict[str, list[str]] = {}
        other_by_base: Dict[str, list[str]] = {}
        for meta in asset_pairs.values():
            if not isinstance(meta, dict):
                continue
            wsname = meta.get("wsname")
            if not isinstance(wsname, str) or "/" not in wsname:
                continue
            base_ws, quote_ws = [p.strip().upper() for p in wsname.split("/", 1)]
            if quote_ws != quote_u or base_ws not in allowed_exchange_bases:
                continue
            status = str(meta.get("status", "")).lower()
            target = online_by_base if status == "online" else other_by_base
            target.setdefault(base_ws, []).append(wsname)
        for base_ws in sorted(set(online_by_base) | set(other_by_base)):
            display_base = MAJORS_EXCHANGE_TO_DISPLAY.get(base_ws, base_ws)
            bases.add(display_base)
            pairs_by_base[display_base] = [f"{display_base}/{quote_asset}"]
        return sorted(bases), pairs_by_base

    # Stablecoins: original logic
    online_by_base = {}
    other_by_base = {}
    for meta in asset_pairs.values():
        if not isinstance(meta, dict):
            continue
        wsname = meta.get("wsname")
        if not isinstance(wsname, str) or "/" not in wsname:
            continue
        base_ws, quote_ws = [p.strip().upper() for p in wsname.split("/", 1)]
        if quote_ws != quote_u:
            continue
        if base_ws not in STABLECOINS:
            continue
        status = str(meta.get("status", "")).lower()
        target = online_by_base if status == "online" else other_by_base
        target.setdefault(base_ws, []).append(wsname)

    for base_sym in set(online_by_base) | set(other_by_base):
        online = sorted(online_by_base.get(base_sym, []))
        other = sorted(other_by_base.get(base_sym, []))
        merged = online + other
        if merged:
            bases.add(base_sym)
            pairs_by_base[base_sym] = merged

    return sorted(bases), pairs_by_base


def render_top_controls(
    connection_status: str,
    status_message: Optional[str] = None,
    asset_pairs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render compact horizontal control bar at top of main content; return current selections."""
    asset_pairs = asset_pairs or {}

    # Row 1: Asset Group, Quote, Base, Pair (Pair given more space)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2.5])
    with c1:
        asset_group = st.selectbox(
            "Asset Group",
            options=["Stablecoins", "Majors"],
            index=0,
            key="top_asset_group",
        )
    with c2:
        quote_asset = st.selectbox(
            "Quote Asset",
            options=["USD", "USDT"],
            index=0,
            key="top_quote_asset",
        )
    bases, pairs_by_base = _derive_group_pairs(asset_pairs, asset_group, quote_asset)
    if asset_group == "Stablecoins":
        default_base = "USDC" if "USDC" in bases else (bases[0] if bases else "USDC")
    else:
        default_base = "BTC" if "BTC" in bases else (bases[0] if bases else "BTC")
    with c3:
        base_options = bases or [default_base]
        base_idx = base_options.index(default_base) if default_base in base_options else 0
        base_asset = st.selectbox(
            "Base Asset",
            options=base_options,
            index=min(base_idx, len(base_options) - 1),
            key="top_base_asset",
        )
    with c4:
        pair_options = pairs_by_base.get(base_asset, [])
        if not pair_options:
            pair_options = [f"{base_asset}/{quote_asset}"]
        pair_selected = st.selectbox(
            "Pair",
            options=pair_options,
            index=0,
            key="top_pair",
        )
    # For Majors, selection is display form (e.g. BTC/USD); use exchange wsname for API
    pair_wsname = _display_pair_to_exchange_wsname(pair_selected) if asset_group == "Majors" else pair_selected

    # Row 2: Time Window only (5m and Country controls removed from UI; logic preserved for future use)
    d1, d2 = st.columns([3, 1])
    with d1:
        timeframe = st.radio(
            "Time Window",
            options=["1D", "1W", "1M", "3M"],
            index=0,
            horizontal=True,
            key="top_timeframe",
        )
    with d2:
        if status_message:
            st.caption(status_message)
    st.write("")

    return {
        "asset_group": asset_group,
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "pair_wsname": pair_wsname,
        "timeframe": timeframe,
        "hi_res": False,
        "country_code": None,
    }


def render_sidebar_minimal(
    connection_status: str,
    status_message: Optional[str] = None,
) -> None:
    """Render minimal sidebar: status only (controls moved to top bar)."""
    with st.sidebar:
        st.markdown("**Status**")
        st.markdown(
            f'<span class="k-pill-status">API: {connection_status}</span>',
            unsafe_allow_html=True,
        )
        if status_message:
            st.caption(status_message)




def render_primary_kpi_strip(metrics: Dict[str, str]) -> None:
    """Single horizontal KPI strip: 6 metrics for Stablecoins (peg, spread, depth, combined, volume, depeg)."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Peg Deviation (bp)", metrics.get("peg_deviation_bps", "—")),
        (c2, "Spread (bp)", metrics.get("spread_bps", "—")),
        (c3, "Depth ±10bp", metrics.get("depth_usd_10bp", "—")),
        (c4, "Combined Depth ±10bp", metrics.get("combined_stablecoin_depth_10bp", "—")),
        (c5, "24h Volume", metrics.get("volume_24h", "—")),
        (c6, "Depeg Alerts", metrics.get("depeg_events", "—")),
    ]
    for col, label, value in kpis:
        with col:
            st.metric(label=label, value=value)
    st.write("")


def render_majors_kpi_strip(metrics: Dict[str, str]) -> None:
    """Majors KPI strip: Spread, Depth ±10bp, Depth ±25bp, 24h Volume, Order Book Imbalance, Regime."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Spread (bp)", metrics.get("spread_bps", "—")),
        (c2, "Depth ±10bp", metrics.get("depth_usd_10bp", "—")),
        (c3, "Depth ±25bp", metrics.get("depth_25_val", "—")),
        (c4, "24h Volume", metrics.get("volume_24h", "—")),
        (c5, "Order Book Imbalance", metrics.get("imbalance_val", "—")),
        (c6, "Regime", metrics.get("regime_label", "—")),
    ]
    for col, label, value in kpis:
        with col:
            st.metric(label=label, value=value)
    st.write("")


def _build_peg_deviation_figure(
    df: pd.DataFrame,
    is_stablecoin_usd: bool,
    height: int = 320,
    title: Optional[str] = None,
) -> Optional[Any]:
    """Build Plotly peg deviation (or price) line chart; thin lines, no fill. Returns figure or None."""
    if df.empty or "date" not in df.columns:
        return None
    if is_stablecoin_usd and "price" in df.columns:
        df = df.copy()
        df["deviation_bp"] = (df["price"].astype(float) - 1.0) * 10_000.0
        fig = px.line(df, x="date", y="deviation_bp")
        fig.update_traces(line=dict(color="#2F6BFF", width=1.5), fill=None)
        fig.update_layout(
            margin=dict(l=10, r=10, t=8, b=10),
            height=height,
            yaxis_title="Deviation (bp)",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1F2937", size=12),
            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
        )
        for y_val, dash in [(0, "solid"), (20, "dash"), (-20, "dash"), (50, "dot"), (-50, "dot")]:
            fig.add_hline(y=y_val, line_dash=dash, line_color="#F3F4F6", line_width=0.8)
        return fig
    if "price" in df.columns:
        fig = px.line(df, x="date", y="price")
        fig.update_traces(line=dict(color="#2F6BFF", width=1.5), fill=None)
        layout = dict(
            margin=dict(l=10, r=10, t=42, b=10),
            height=height,
            yaxis_title="Price",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1F2937", size=12),
            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
        )
        if title:
            layout["title"] = dict(text=title, font=dict(size=13), x=0.5, xanchor="center")
        fig.update_layout(**layout)
        return fig
    return None


def make_placeholder_price_series(timeframe: str) -> pd.DataFrame:
    """Generate a simple placeholder price series for the chart."""
    periods_map = {
        "1D": 24,
        "1W": 7,
        "1M": 30,
        "3M": 90,
        "1Y": 365,
    }
    periods = periods_map.get(timeframe, 30)

    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="H")
    base_price = 100
    prices = [
        base_price + i * 0.3 + (i % 5) * 0.8 for i in range(len(dates))
    ]

    return pd.DataFrame({"date": dates, "price": prices})


def _ensure_alerts_state() -> None:
    """Initialize session state for alerts and depth warning."""
    if ALERTS_DEPEG_KEY not in st.session_state:
        st.session_state[ALERTS_DEPEG_KEY] = {}
    if ALERTS_LIQUIDITY_KEY not in st.session_state:
        st.session_state[ALERTS_LIQUIDITY_KEY] = []
    if ALERTS_SHOW_ALL_KEY not in st.session_state:
        st.session_state[ALERTS_SHOW_ALL_KEY] = False
    if DEPTH_WARNING_FIRED_KEY not in st.session_state:
        st.session_state[DEPTH_WARNING_FIRED_KEY] = set()


def render_dashboard_header(connection_status: str) -> None:
    """Render page title, subtitle, and status (controls are in top bar below)."""
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(
            """
            <p class="k-title">US Stablecoin Monitoring Dashboard</p>
            <p class="k-subtitle">
              Market structure, peg stability, and liquidity monitoring for U.S.-quoted stablecoin pairs.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(
            f'<span class="k-pill-status">API: {connection_status}</span>',
            unsafe_allow_html=True,
        )
    st.write("")


def render_main_area(
    timeframe: str,
    base_asset: str,
    quote_asset: str,
    asset_pairs: Dict[str, Any],
    pair_wsname: str,
    hi_res: bool = False,
    connection_status: str = "connected",
    asset_group: str = "Stablecoins",
) -> Dict[str, Any]:
    """Render dashboard: primary KPI strip only. Returns overview_context for Overview tab (hero + strip)."""
    _ensure_alerts_state()
    peg_threshold_bp = st.session_state.get("peg_threshold_bp", 20.0)
    depth_warning_threshold = st.session_state.get("depth_warning_threshold", 250_000.0)

    # Active chart mode: 5m (1D) overrides to 5m interval + last 24h; else use radio timeframe
    active_interval, active_window_label, is_intraday_24h = _get_active_chart_mode(timeframe, hi_res)

    # --- Fetch data (non-fatal: fallbacks on API failure) ---
    rest_pair = _pair_for_rest(pair_wsname) if pair_wsname else ""
    ohlc_df, ohlc_err = get_ohlc(rest_pair or pair_wsname or "", active_interval) if pair_wsname else (pd.DataFrame(), "No pair")
    if ohlc_err and pair_wsname:
        audit_log_append(EVENT_API_ERROR, f"OHLC {_display_pair_for_ui(pair_wsname)}: {ohlc_err}")
    if not ohlc_df.empty and is_intraday_24h:
        ohlc_df = _trim_ohlc_to_last_24h(ohlc_df)

    order_book, ob_err = get_order_book(pair_wsname) if pair_wsname else ({}, "No pair")
    if ob_err and pair_wsname:
        audit_log_append(EVENT_API_ERROR, f"Depth {_display_pair_for_ui(pair_wsname)}: {ob_err}")

    ticker_result, ticker_err = get_ticker(rest_pair or pair_wsname or "") if pair_wsname else ({}, "No pair")
    if ticker_err and pair_wsname:
        audit_log_append(EVENT_API_ERROR, f"Ticker {_display_pair_for_ui(pair_wsname)}: {ticker_err}")
    first_payload = next(iter(ticker_result.values()), {}) or {} if ticker_result else {}
    best_bid_t, best_ask_t, last_t, volume_24h = parse_ticker_for_mid(first_payload)

    # Mid: from orderbook when available, else ticker last
    mid: Optional[float] = None
    best_bid: Optional[float] = best_bid_t
    best_ask: Optional[float] = best_ask_t
    if order_book and order_book.get("bids") is not None and order_book.get("asks") is not None:
        bids_df = _to_dataframe(order_book.get("bids", []))
        asks_df = _to_dataframe(order_book.get("asks", []))
        if not bids_df.empty and not asks_df.empty:
            mid = mid_price(bids_df, asks_df)
            best_bid = float(bids_df["price"].max())
            best_ask = float(asks_df["price"].min())
    if mid is None and last_t is not None:
        mid = last_t

    # Peg deviation: stablecoin/USD only (peg=1.0); BTC pairs show "—"
    is_stablecoin_usd = base_asset.upper() in STABLECOINS and quote_asset.upper() == "USD"
    peg_dev_bps: Optional[float] = None
    if is_stablecoin_usd and mid is not None:
        peg_dev_bps = peg_deviation_bps(mid, 1.0)
    # else: BTC/USD, BTC/USDT etc. -> leave None so we show "—"

    spread_bps_val: Optional[float] = spread_bps(best_bid, best_ask, mid) if mid else None

    depth_usd_10bp: Optional[float] = None
    if order_book and mid is not None:
        bids_df = _to_dataframe(order_book.get("bids", []))
        asks_df = _to_dataframe(order_book.get("asks", []))
        d10 = depth_at_bps(bids_df, asks_df, mid, 10.0)
        total_d10 = d10.get("total")
        depth_usd_10bp = total_d10 if total_d10 is not None else None

    # Depeg events: run detector for stablecoin pairs; store by (pair_wsname, timeframe)
    depeg_events_list: List[Dict[str, Any]] = []
    if is_stablecoin_usd and not ohlc_df.empty and "close" in ohlc_df.columns:
        depeg_events_list = detect_depeg_events(
            ohlc_df,
            threshold_bp=peg_threshold_bp,
            peg_target=1.0,
            n_consecutive_start=3,
            n_consecutive_end=2,
        )
    alert_key = f"{pair_wsname}|{timeframe}"
    st.session_state[ALERTS_DEPEG_KEY][alert_key] = depeg_events_list
    if depeg_events_list and pair_wsname:
        depeg_logged = st.session_state.setdefault("_depeg_alert_logged", set())
        if alert_key not in depeg_logged:
            audit_log_append(
                EVENT_ALERT_TRIGGERED,
                f"Depeg events detected: {_display_pair_for_ui(pair_wsname)} ({len(depeg_events_list)} events in window).",
            )
            depeg_logged.add(alert_key)
            st.session_state["_depeg_alert_logged"] = depeg_logged

    # Depth warning: trigger once per pair when below threshold; clear when above
    depth_warning_fired = st.session_state[DEPTH_WARNING_FIRED_KEY]
    if depth_usd_10bp is not None and depth_warning_threshold is not None:
        if depth_usd_10bp < depth_warning_threshold:
            if pair_wsname not in depth_warning_fired:
                depth_warning_fired.add(pair_wsname)
                st.session_state[ALERTS_LIQUIDITY_KEY].append({
                    "pair": _display_pair_for_ui(pair_wsname),
                    "message": f"Liquidity Warning: Depth @ ±10bp ({depth_usd_10bp:,.0f}) < {depth_warning_threshold:,.0f}",
                })
                audit_log_append(EVENT_ALERT_TRIGGERED, f"Depth warning {_display_pair_for_ui(pair_wsname)}: depth {depth_usd_10bp:,.0f} < {depth_warning_threshold:,.0f}")
        else:
            depth_warning_fired.discard(pair_wsname)

    def _fmt_bp(v: Optional[float]) -> str:
        if v is None:
            return "—"
        try:
            return f"{v:.1f}"
        except Exception:
            return "—"

    def _fmt_usd(v: Optional[float]) -> str:
        if v is None:
            return "—"
        try:
            return f"{v:,.0f}"
        except Exception:
            return "—"

    # Combined stablecoin depth ±10bp (sum across USDC/USD, USDT/USD, USAT/USD, USD1/USD)
    combined_stablecoin_depth: Optional[float] = None
    try:
        total = 0.0
        for p in STABLECOIN_USD_PAIRS:
            snap = compute_pair_snapshot(p)
            d = snap.get("depth_10bp")
            if d is not None:
                total += float(d)
        combined_stablecoin_depth = total if total > 0 else None
    except Exception:
        combined_stablecoin_depth = None

    is_majors = asset_group == "Majors"
    # Depth ±25bp and imbalance (needed for both strips)
    depth_25_val: Optional[float] = None
    imbalance_val: Optional[float] = None
    if order_book and mid is not None:
        _bids = _to_dataframe(order_book.get("bids", []))
        _asks = _to_dataframe(order_book.get("asks", []))
        if not _bids.empty and not _asks.empty:
            d25 = depth_at_bps(_bids, _asks, mid, 25.0)
            depth_25_val = d25.get("total")
            imbalance_val = order_book_imbalance(_bids, _asks, mid, 10.0)
    snap_current = compute_pair_snapshot(pair_wsname) if pair_wsname else {}
    regime_raw = snap_current.get("liquidity_regime") or "normal"
    regime_label = _format_regime_label(regime_raw)
    stress_val = snap_current.get("stress_score")
    stress_str = f"{stress_val:.2f}" if stress_val is not None else "—"

    # ROW 2 — KPI strip (Majors: market-structure; Stablecoins: peg-focused)
    if is_majors:
        majors_kpis = {
            "spread_bps": _fmt_bp(spread_bps_val),
            "depth_usd_10bp": format_compact_number(depth_usd_10bp),
            "depth_25_val": format_compact_number(depth_25_val),
            "volume_24h": format_compact_number(volume_24h),
            "imbalance_val": f"{imbalance_val:.2f}" if imbalance_val is not None else "—",
            "regime_label": regime_label or "—",
        }
        render_majors_kpi_strip(majors_kpis)
    else:
        main_kpis = {
            "peg_deviation_bps": _fmt_bp(peg_dev_bps),
            "spread_bps": _fmt_bp(spread_bps_val),
            "depth_usd_10bp": format_compact_number(depth_usd_10bp),
            "volume_24h": format_compact_number(volume_24h),
            "depeg_events": str(len(depeg_events_list)),
            "combined_stablecoin_depth_10bp": format_compact_number(combined_stablecoin_depth),
        }
        render_primary_kpi_strip(main_kpis)

    # Build chart_df for Overview tab (ohlc_df already trimmed to 24h when 5m mode)
    chart_df = pd.DataFrame()
    if not ohlc_df.empty and "close" in ohlc_df.columns:
        chart_df = pd.DataFrame({
            "date": ohlc_df["time"],
            "price": pd.to_numeric(ohlc_df["close"], errors="coerce"),
        })

    overview_context: Dict[str, Any] = {
        "chart_df": chart_df,
        "ohlc_chart": ohlc_df.copy() if not ohlc_df.empty else None,
        "active_interval": active_interval,
        "active_window_label": active_window_label,
        "is_intraday_24h": is_intraday_24h,
        "is_stablecoin_usd": is_stablecoin_usd,
        "is_majors": is_majors,
        "order_book": order_book,
        "mid": mid,
        "depth_usd_10bp": depth_usd_10bp,
        "depth_25_val": depth_25_val,
        "imbalance_val": imbalance_val,
        "regime_label": regime_label,
        "stress_str": stress_str,
        "peg_dev_bps": peg_dev_bps,
        "spread_bps_val": spread_bps_val,
        "depeg_count": len(depeg_events_list),
        "stress_val": stress_val,
    }
    return overview_context


def render_alerts_expander(asset_group: str = "Stablecoins") -> None:
    """Render Alerts & thresholds expander below tabs (no analytics change). Majors: market-structure only."""
    _ensure_alerts_state()
    is_majors = asset_group == "Majors"
    peg_threshold_bp = st.session_state.get("peg_threshold_bp", 20.0)
    depth_warning_threshold = st.session_state.get("depth_warning_threshold", 250_000.0)
    with st.expander("Alerts & thresholds", expanded=False):
        if is_majors:
            depth_warning_threshold = st.number_input(
                "Depth Warning Threshold (USD)",
                min_value=0.0,
                value=250_000.0,
                step=50_000.0,
                key="depth_warning_threshold",
            )
            if st.session_state.get("last_depth_threshold") is not None and st.session_state["last_depth_threshold"] != depth_warning_threshold:
                audit_log_append(EVENT_THRESHOLD_CHANGE, f"Depth warning (USD): {depth_warning_threshold:,.0f}")
            st.session_state["last_depth_threshold"] = depth_warning_threshold
            st.caption("Thresholds used to flag liquidity deterioration and spread deterioration (client-side).")
            liquidity_alerts = st.session_state.get(ALERTS_LIQUIDITY_KEY, [])
            if liquidity_alerts:
                st.markdown('<p class="k-section-title">Liquidity Alerts</p>', unsafe_allow_html=True)
                for a in liquidity_alerts:
                    msg = a.get("message", a) if isinstance(a, dict) else str(a)
                    st.markdown(f'<span class="k-depeg-badge">{msg}</span>', unsafe_allow_html=True)
            else:
                st.caption("No liquidity alerts for monitored major pairs.")
            if st.button("Clear all alerts", key="clear_alerts_majors"):
                st.session_state[ALERTS_LIQUIDITY_KEY] = []
                st.session_state[DEPTH_WARNING_FIRED_KEY] = set()
                audit_log_append(EVENT_ALERT_CLEARED, "User cleared all alerts")
                st.rerun()
        else:
            peg_threshold_bp = st.number_input(
                "Peg Deviation Threshold (bp)",
                min_value=0.0,
                value=20.0,
                step=1.0,
                key="peg_threshold_bp",
            )
            depth_warning_threshold = st.number_input(
                "Depth Warning Threshold (USD)",
                min_value=0.0,
                value=250_000.0,
                step=50_000.0,
                key="depth_warning_threshold",
            )
            if st.session_state.get("last_peg_threshold") is not None and st.session_state["last_peg_threshold"] != peg_threshold_bp:
                audit_log_append(EVENT_THRESHOLD_CHANGE, f"Peg threshold (bp): {peg_threshold_bp}")
            if st.session_state.get("last_depth_threshold") is not None and st.session_state["last_depth_threshold"] != depth_warning_threshold:
                audit_log_append(EVENT_THRESHOLD_CHANGE, f"Depth warning (USD): {depth_warning_threshold:,.0f}")
            st.session_state["last_peg_threshold"] = peg_threshold_bp
            st.session_state["last_depth_threshold"] = depth_warning_threshold
            st.caption("Thresholds used to flag depeg or liquidity deterioration (client-side).")
            st.markdown('<p class="k-section-title">Depeg Alerts</p>', unsafe_allow_html=True)
            all_depeg: List[Dict[str, Any]] = []
            for k, evts in st.session_state[ALERTS_DEPEG_KEY].items():
                all_depeg.extend(evts)
            liquidity_alerts = st.session_state.get(ALERTS_LIQUIDITY_KEY, [])
            show_all = st.session_state.get(ALERTS_SHOW_ALL_KEY, False)
            if not all_depeg and not liquidity_alerts:
                st.caption("No depeg events detected in the selected window.")
            else:
                if all_depeg:
                    st.caption(f"{len(all_depeg)} depeg event(s) in window.")
                    to_show = all_depeg if show_all else all_depeg[:10]
                    rows = []
                    for e in to_show:
                        start = e.get("start_time", "")
                        if hasattr(start, "strftime"):
                            start = start.strftime("%Y-%m-%d %H:%M") if start else "—"
                        rows.append({
                            "Start Time": start,
                            "Duration": f"{e.get('duration_minutes', 0)} min",
                            "Max Deviation (bp)": e.get("max_deviation_bp", "—"),
                            "Spread at Peak": e.get("spread_at_peak", "—"),
                            "Depth ±10bp at Peak": e.get("depth_10bp_at_peak", "—"),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=min(180, 28 + len(rows) * 20))
                for a in liquidity_alerts:
                    msg = a.get("message", a) if isinstance(a, dict) else str(a)
                    st.markdown(f'<span class="k-depeg-badge">{msg}</span>', unsafe_allow_html=True)
                if len(all_depeg) > 10 and not show_all:
                    if st.button("Show all", key="show_all_alerts"):
                        st.session_state[ALERTS_SHOW_ALL_KEY] = True
                        st.rerun()
                elif show_all and len(all_depeg) > 10:
                    if st.button("Show less", key="show_less"):
                        st.session_state[ALERTS_SHOW_ALL_KEY] = False
                        st.rerun()
            if st.button("Clear all alerts", key="clear_alerts_stable"):
                st.session_state[ALERTS_DEPEG_KEY] = {}
                st.session_state[ALERTS_LIQUIDITY_KEY] = []
                st.session_state[DEPTH_WARNING_FIRED_KEY] = set()
                st.session_state[ALERTS_SHOW_ALL_KEY] = False
                audit_log_append(EVENT_ALERT_CLEARED, "User cleared all alerts")
                st.rerun()


def _market_narrative_sentence(
    peg_dev_bps: Optional[float],
    spread_bps_val: Optional[float],
    depth_10bp: Optional[float],
    stress_val: Optional[float],
    depeg_count: int,
) -> str:
    """One rules-based narrative sentence from existing metrics only (display-only)."""
    depth_ok = depth_10bp is not None and depth_10bp > 1_000_000
    spread_ok = spread_bps_val is not None and spread_bps_val < 5
    peg_ok = peg_dev_bps is not None and abs(peg_dev_bps) < 20
    stress_ok = stress_val is not None and stress_val < 0.25
    if depeg_count > 0:
        return "Depeg events detected in window; review Peg & Stability tab."
    if peg_ok and spread_ok and depth_ok and stress_ok:
        return "Market appears orderly with tight spread and deep near-peg liquidity."
    if peg_ok and (not depth_ok or not spread_ok):
        return "Pair remains near peg, but thinner depth suggests higher execution sensitivity."
    return "Wide spreads and shallow near-peg depth indicate elevated execution risk."


def render_tabs(
    selections: Dict[str, str],
    asset_pairs: Dict[str, Any],
    asset_pairs_error: Optional[str] = None,
    overview_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Render tab row. Majors: Overview, Liquidity, Surveillance, Audit Log. Stablecoins: + Peg & Stability."""
    st.write("")
    is_majors = selections.get("asset_group") == "Majors"
    if is_majors:
        tab_overview, tab_liquidity, tab_surveillance, tab_audit = st.tabs(
            ["Overview", "Liquidity", "Surveillance", "Audit Log"]
        )
        tab_peg = None
    else:
        tab_overview, tab_peg, tab_liquidity, tab_surveillance, tab_audit = st.tabs(
            ["Overview", "Peg & Stability", "Liquidity", "Surveillance", "Audit Log"]
        )

    base_asset = selections["base_asset"]
    quote_asset = selections["quote_asset"]
    timeframe = selections["timeframe"]
    pair_key = resolve_pair_code(asset_pairs, base_asset, quote_asset)
    pair_wsname = selections.get("pair_wsname", f"{base_asset}/{quote_asset}")
    overview_context = overview_context if overview_context is not None else {}
    oc_is_majors = overview_context.get("is_majors", is_majors)

    with tab_overview:
        # ROW A — Hero: left = peg deviation (Stablecoins) or price (Majors), right = liquidity wall
        if overview_context:
            hero_left, hero_right = st.columns([1, 1])
            with hero_left:
                maybe_chart = overview_context.get("chart_df")
                chart_df = maybe_chart if maybe_chart is not None else pd.DataFrame()
                is_sc = overview_context.get("is_stablecoin_usd", False)
                if oc_is_majors:
                    fig_main = _build_peg_deviation_figure(
                        chart_df, False, height=300, title="Price Over Time"
                    )
                    if fig_main is not None:
                        st.plotly_chart(fig_main, use_container_width=True, key="hero_peg_chart")
                    else:
                        st.caption("Price data unavailable for selected major pair.")
                else:
                    fig_main = _build_peg_deviation_figure(chart_df, is_sc, height=300)
                    if fig_main is not None:
                        st.plotly_chart(fig_main, use_container_width=True, key="hero_peg_chart")
                    else:
                        st.caption("No price data for chart.")
            with hero_right:
                ob = overview_context.get("order_book")
                mid = overview_context.get("mid")
                fig_lw = get_liquidity_wall_figure(ob, mid, height=300, is_majors=oc_is_majors) if (ob is not None and mid is not None) else None
                if fig_lw is not None:
                    st.plotly_chart(fig_lw, use_container_width=True, key="hero_liquidity_wall")
                    st.caption("Cumulative bid/ask depth relative to mid price." if oc_is_majors else "Negative = bid depth; positive = ask depth from mid.")
                else:
                    st.caption("No order book data for liquidity wall.")
            st.write("")

            # ROW B — Supporting microstructure strip
            st.markdown('<div class="k-supporting-strip-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                st.metric("Depth ±10bp", format_compact_number(overview_context.get("depth_usd_10bp")))
            with r2:
                st.metric("Depth ±25bp", format_compact_number(overview_context.get("depth_25_val")))
            with r3:
                imb = overview_context.get("imbalance_val")
                st.metric("Imbalance", f"{imb:.2f}" if imb is not None else "—")
            with r4:
                st.metric("Regime", overview_context.get("regime_label", "—"))
            with r5:
                st.metric("Stress", overview_context.get("stress_str", "—"))
            if oc_is_majors:
                narrative = "Market structure and liquidity view for selected major pair."
            else:
                narrative = _market_narrative_sentence(
                    overview_context.get("peg_dev_bps"),
                    overview_context.get("spread_bps_val"),
                    overview_context.get("depth_usd_10bp"),
                    overview_context.get("stress_val"),
                    overview_context.get("depeg_count", 0),
                )
            st.caption(narrative)
            depth_10 = overview_context.get("depth_usd_10bp")
            depth_25 = overview_context.get("depth_25_val")
            if (depth_10 is None or depth_10 == 0) and depth_25 and depth_25 > 0:
                st.caption("Depth ±10bp = 0 indicates no liquidity within ±10bp; wider bands may still show depth.")
        st.write("")

        # ROW C — Cross-Pair Comparison (Monitoring State + disclaimer)
        st.markdown('<p class="k-section-title">Cross-Pair Comparison</p>', unsafe_allow_html=True)
        cross_order = (
            ["XBT/USD", "XAU/USD"] if oc_is_majors else ["USDT/USD", "USDC/USD", "USD1/USD", "USAT/USD"]
        )
        cross_rows = []
        heatmap_rows = []
        for p in cross_order:
            try:
                s = compute_pair_snapshot(p)
                if s.get("mid") is None and s.get("depth_10bp") is None:
                    continue
                mq = _market_quality_label(
                    s.get("stress_score"),
                    s.get("spread_bp"),
                    s.get("depth_10bp"),
                )
                cross_rows.append({
                    "Pair": _display_pair_for_ui(p),
                    "Peg Deviation (bp)": "—" if oc_is_majors else format_percent_like_bp(s.get("peg_deviation_bp")),
                    "Spread (bp)": format_percent_like_bp(s.get("spread_bp")),
                    "Depth ±10bp": format_compact_number(s.get("depth_10bp")),
                    "Stress Score": f"{s.get('stress_score'):.2f}" if s.get("stress_score") is not None else "—",
                    "Regime": _format_regime_label(s.get("liquidity_regime")),
                    "Monitoring State": mq,
                })
                # Build heatmap data: depth at 5, 10, 25, 50 bp
                order_book, _ = get_order_book(p)
                if order_book and s.get("mid"):
                    bids_df = _to_dataframe(order_book.get("bids", []))
                    asks_df = _to_dataframe(order_book.get("asks", []))
                    mid = s["mid"]
                    if not bids_df.empty and not asks_df.empty:
                        d5 = depth_at_bps(bids_df, asks_df, mid, 5.0)
                        d10 = depth_at_bps(bids_df, asks_df, mid, 10.0)
                        d25 = depth_at_bps(bids_df, asks_df, mid, 25.0)
                        d50 = depth_at_bps(bids_df, asks_df, mid, 50.0)
                        heatmap_rows.append({
                            "Pair": _display_pair_for_ui(p),
                            "Depth ±5bp": d5.get("total") or 0,
                            "Depth ±10bp": d10.get("total") or 0,
                            "Depth ±25bp": d25.get("total") or 0,
                            "Depth ±50bp": d50.get("total") or 0,
                        })
            except Exception:
                continue
        if cross_rows:
            df_cross = pd.DataFrame(cross_rows)
            # For Majors, drop Peg Deviation column so table is cleaner
            if oc_is_majors and "Peg Deviation (bp)" in df_cross.columns:
                df_cross = df_cross.drop(columns=["Peg Deviation (bp)"])
            def _mq_badge_style(v):
                if v == "ORDERLY":
                    return "background-color: #059669; color: #fff; padding: 2px 5px; border-radius: 3px; font-size: 11px; font-weight: 500;"
                if v == "MONITOR":
                    return "background-color: #CA8A04; color: #1f2937; padding: 2px 5px; border-radius: 3px; font-size: 11px; font-weight: 500;"
                if v == "ELEVATED RISK":
                    return "background-color: #B91C1C; color: #fff; padding: 2px 5px; border-radius: 3px; font-size: 11px; font-weight: 500;"
                return ""
            styled = df_cross.style.apply(
                lambda s: [_mq_badge_style(v) if c == "Monitoring State" else "" for c, v in s.items()],
                axis=1,
            )
            st.dataframe(styled, use_container_width=True, height=min(180, 40 + len(cross_rows) * 32), hide_index=True)
            st.caption(
                "Internal monitoring label from spread, near-peg depth, and stress. Separate from Regime; not a legal or solvency view."
            )
        else:
            st.caption("No stablecoin pair data available." if not oc_is_majors else "No major pair data available.")

        # ROW D — Near-Peg Liquidity Depth heat map
        if heatmap_rows:
            st.markdown('<p class="k-section-title">Near-Peg Liquidity Depth</p>', unsafe_allow_html=True)
            df_heatmap = pd.DataFrame(heatmap_rows)
            _render_depth_heatmap(df_heatmap, is_majors=oc_is_majors)

        # ROW E — BTC Context strip (hidden for Majors; redundant when viewing BTC/Gold)
        if not oc_is_majors:
            st.markdown('<p class="k-section-title k-section-spaced">BTC Context</p>', unsafe_allow_html=True)
            try:
                snap_btc_usd = compute_pair_snapshot("XBT/USD")
                snap_btc_usdt = compute_pair_snapshot("XBT/USDT")
                mid_usd = snap_btc_usd.get("mid")
                mid_usdt = snap_btc_usdt.get("mid")
                basis_ov = (mid_usdt - mid_usd) / mid_usd * 10_000.0 if (mid_usd and mid_usdt and mid_usd != 0) else None
            except Exception:
                snap_btc_usd = {}
                basis_ov = None
            bd1, bd2, bd3 = st.columns(3)
            with bd1:
                st.metric("BTC Depth ±10bp", format_compact_number(snap_btc_usd.get("depth_10bp") if snap_btc_usd else None))
            with bd2:
                st.metric("BTC Spread (bp)", format_bp_with_small(snap_btc_usd.get("spread_bp") if snap_btc_usd else None))
            with bd3:
                st.metric("BTC Stablecoin Basis (bp)", format_bp_with_small(basis_ov))

    if tab_peg is not None:
        with tab_peg:
            is_stablecoin_peg = base_asset.upper() in STABLECOINS and quote_asset.upper() == "USD"
            if not is_stablecoin_peg:
                st.info("Peg stability analytics apply only to stablecoin/USD pairs.")
            else:
                hi_res_peg = selections.get("hi_res", False)
                active_interval_peg, active_window_label_peg, is_intraday_24h_peg = _get_active_chart_mode(
                    selections.get("timeframe", "1D"), hi_res_peg
                )
                # Use same OHLC as main area when available (already trimmed when 5m mode)
                ohlc_peg = None
                if overview_context and overview_context.get("ohlc_chart") is not None and not overview_context["ohlc_chart"].empty:
                    ohlc_peg = overview_context["ohlc_chart"]
                if ohlc_peg is None and pair_wsname:
                    ohlc_peg, _ = get_ohlc(_pair_for_rest(pair_wsname) or pair_wsname, active_interval_peg)
                    if not ohlc_peg.empty and is_intraday_24h_peg:
                        ohlc_peg = _trim_ohlc_to_last_24h(ohlc_peg)
                if ohlc_peg is None:
                    ohlc_peg = pd.DataFrame()
                snap_peg = compute_pair_snapshot(pair_wsname) if pair_wsname else {}
                current_peg_bp = snap_peg.get("peg_deviation_bp")
                alert_key_peg = f"{pair_wsname}|{selections.get('timeframe', '1D')}"
                depeg_peg_list = st.session_state.get(ALERTS_DEPEG_KEY, {}).get(alert_key_peg, [])
                max_dev_bp: Optional[float] = None
                mean_dev_bp: Optional[float] = None
                if not ohlc_peg.empty and "close" in ohlc_peg.columns:
                    closes = pd.to_numeric(ohlc_peg["close"], errors="coerce").dropna()
                    if not closes.empty:
                        dev_bp = (closes - 1.0) * 10_000.0
                        max_dev_bp = float(dev_bp.abs().max())
                        mean_dev_bp = float(dev_bp.abs().mean())

                # PEG ROW A — Summary metrics
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.metric("Current Deviation", format_percent_like_bp(current_peg_bp))
                with p2:
                    st.metric("Max in Window", format_percent_like_bp(max_dev_bp))
                with p3:
                    st.metric("Mean in Window", format_percent_like_bp(mean_dev_bp))
                with p4:
                    st.metric("Depeg Events", str(len(depeg_peg_list)))

                # PEG ROW B — Deviation Over Time
                st.markdown('<p class="k-section-title">Deviation Over Time</p>', unsafe_allow_html=True)
                if not ohlc_peg.empty and "close" in ohlc_peg.columns:
                    df_peg = pd.DataFrame({
                        "date": ohlc_peg["time"],
                        "price": pd.to_numeric(ohlc_peg["close"], errors="coerce"),
                    }).dropna(subset=["price"])
                    if not df_peg.empty:
                        fig_peg_line = _build_peg_deviation_figure(df_peg, True, height=220)
                        if fig_peg_line:
                            st.plotly_chart(fig_peg_line, use_container_width=True, key="peg_stability_chart")

                # PEG ROW C — Compact Distribution (smaller than main chart)
                st.markdown('<p class="k-section-title">Distribution</p>', unsafe_allow_html=True)
                if not ohlc_peg.empty and "close" in ohlc_peg.columns:
                    dev_series = (pd.to_numeric(ohlc_peg["close"], errors="coerce") - 1.0) * 10_000.0
                    dev_series = dev_series.dropna()
                    if not dev_series.empty:
                        n_vals = len(dev_series)
                        nbins = min(24, max(6, n_vals // 4))
                        fig_peg_hist = px.histogram(
                            x=dev_series,
                            nbins=nbins,
                            labels={"x": "Peg deviation (bp)", "y": "Count"},
                        )
                        fig_peg_hist.update_traces(marker=dict(color="#E5E7EB", line=dict(color="#E5E7EB", width=1)))
                        fig_peg_hist.update_layout(
                            height=130,
                            margin=dict(l=10, r=10, t=4, b=10),
                            showlegend=False,
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            font=dict(color="#1F2937", size=11),
                            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
                            yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB"),
                        )
                        st.plotly_chart(fig_peg_hist, use_container_width=True, key="peg_distribution_chart")

                # PEG ROW D — Recent Depeg Events
                st.markdown('<p class="k-section-title">Recent Depeg Events</p>', unsafe_allow_html=True)
                if depeg_peg_list:
                    rows_ev = [
                        {
                            "Start Time": e.get("start_time").strftime("%Y-%m-%d %H:%M") if hasattr(e.get("start_time"), "strftime") else str(e.get("start_time", "—")),
                            "Duration": f"{e.get('duration_minutes', 0)} min",
                            "Max Deviation (bp)": e.get("max_deviation_bp", "—"),
                        }
                        for e in depeg_peg_list
                    ]
                    st.dataframe(pd.DataFrame(rows_ev), use_container_width=True, height=min(130, 28 + len(rows_ev) * 22), hide_index=True)
                else:
                    st.caption("No depeg events detected in the selected window.")

    with tab_liquidity:
        if not pair_wsname:
            st.error(f"No tradable pair found for {base_asset}/{quote_asset}")
        else:
            st.caption(f"Liquidity snapshot for {_display_pair_for_ui(pair_wsname)}")
            order_book, ob_error = get_order_book(pair_wsname)
            ticker_result, ticker_error = get_ticker(_pair_for_rest(pair_wsname) or pair_wsname)
            if ob_error or ticker_error or not order_book or not ticker_result:
                st.error("Liquidity data unavailable for selected pair.")
            else:
                first_payload = next(iter(ticker_result.values()), {}) or {}
                kpis = compute_liquidity_kpis(
                    order_book=order_book,
                    ticker=first_payload,
                    notional=LIQUIDITY_SLIPPAGE_NOTIONAL,
                    depth_bps=LIQUIDITY_DEPTH_BPS,
                )
                mid_liq = kpis.get("mid")
                render_liquidity_kpis(kpis, order_book=order_book, mid=mid_liq, is_majors=is_majors)

    with tab_surveillance:
        peg_threshold_bp = st.session_state.get("peg_threshold_bp", 20.0)
        depth_warning_threshold = st.session_state.get("depth_warning_threshold", 250_000.0)
        snap_current = compute_pair_snapshot(pair_wsname) if pair_wsname else {}
        is_stablecoin_peg = base_asset.upper() in STABLECOINS and quote_asset.upper() == "USD"
        surv_is_majors = is_majors
        peg_bp = snap_current.get("peg_deviation_bp")
        spread_bp = snap_current.get("spread_bp")
        depth_10bp = snap_current.get("depth_10bp")
        stress_score = snap_current.get("stress_score")
        abs_peg = abs(peg_bp) if peg_bp is not None else 0.0
        basis_bp_surv: Optional[float] = None
        try:
            s_usd = compute_pair_snapshot("XBT/USD")
            s_usdt = compute_pair_snapshot("XBT/USDT")
            mid_usd = s_usd.get("mid")
            mid_usdt = s_usdt.get("mid")
            if mid_usd and mid_usdt and mid_usd != 0:
                basis_bp_surv = (mid_usdt - mid_usd) / mid_usd * 10_000.0
        except Exception:
            pass
        abs_basis = abs(basis_bp_surv) if basis_bp_surv is not None else 0.0
        signal_rows: List[Dict[str, Any]] = []
        active_signals: List[str] = []
        pairs_with_peg: List[str] = []
        pairs_with_liquidity: List[str] = []
        basis_active = False
        # 1. Peg Deviation Warning (stablecoin/USD only; skip for Majors)
        if not surv_is_majors and is_stablecoin_peg and peg_bp is not None:
            if abs_peg >= peg_threshold_bp:
                status = "ACTIVE"
                active_signals.append("peg")
                pairs_with_peg.append(pair_wsname)
                if abs_peg >= 2 * peg_threshold_bp:
                    sev = "HIGH"
                else:
                    sev = "WARNING"
            else:
                status = "CLEAR"
                sev = "INFO"
            signal_rows.append({
                "Signal": "Peg Deviation Warning",
                "Pair": _display_pair_for_ui(pair_wsname) or "—",
                "Severity": sev,
                "Current Value": format_percent_like_bp(peg_bp),
                "Threshold": f"{peg_threshold_bp:.1f} bp",
                "Status": status,
                "Notes": "Stablecoin peg deviation vs threshold.",
            })
        # 2. Liquidity Warning (selected pair)
        if depth_10bp is not None and depth_warning_threshold is not None:
            if depth_10bp < depth_warning_threshold:
                status = "ACTIVE"
                active_signals.append("liquidity")
                pairs_with_liquidity.append(pair_wsname)
                if depth_10bp <= depth_warning_threshold * 0.5:
                    sev = "HIGH"
                else:
                    sev = "WARNING"
            else:
                status = "CLEAR"
                sev = "INFO"
            signal_rows.append({
                "Signal": "Liquidity Warning",
                "Pair": _display_pair_for_ui(pair_wsname) or "—",
                "Severity": sev,
                "Current Value": format_compact_number(depth_10bp),
                "Threshold": format_compact_number(depth_warning_threshold),
                "Status": status,
                "Notes": "Depth ±10bp vs warning threshold.",
            })
        # 3. Wide Spread (selected pair)
        if spread_bp is not None:
            if spread_bp > 15:
                sev = "HIGH"
                status = "ACTIVE"
                active_signals.append("spread")
            elif spread_bp > 5:
                sev = "WARNING"
                status = "ACTIVE"
                active_signals.append("spread")
            else:
                sev = "INFO"
                status = "CLEAR"
            signal_rows.append({
                "Signal": "Wide Spread",
                "Pair": _display_pair_for_ui(pair_wsname) or "—",
                "Severity": sev,
                "Current Value": format_percent_like_bp(spread_bp),
                "Threshold": "5 bp (WARNING) / 15 bp (HIGH)",
                "Status": status,
                "Notes": "Bid-ask spread in bp.",
            })
        # 4. BTC Stablecoin Basis Divergence (Stablecoins only)
        if not surv_is_majors and basis_bp_surv is not None:
            if abs_basis > 10:
                sev = "HIGH"
                status = "ACTIVE"
                basis_active = True
                active_signals.append("basis")
            elif abs_basis > 5:
                sev = "WARNING"
                status = "ACTIVE"
                basis_active = True
                active_signals.append("basis")
            else:
                sev = "INFO"
                status = "CLEAR"
            signal_rows.append({
                "Signal": "BTC Stablecoin Basis Divergence",
                "Pair": "BTC/USD vs BTC/USDT",
                "Severity": sev,
                "Current Value": format_percent_like_bp(basis_bp_surv),
                "Threshold": "5 bp (WARNING) / 10 bp (HIGH)",
                "Status": status,
                "Notes": "Basis between BTC/USDT and BTC/USD.",
            })
        # 5. Elevated Stress Score (selected pair)
        if stress_score is not None:
            if stress_score > 0.75:
                sev = "HIGH"
                status = "ACTIVE"
                active_signals.append("stress")
            elif stress_score > 0.50:
                sev = "WARNING"
                status = "ACTIVE"
                active_signals.append("stress")
            else:
                sev = "INFO"
                status = "CLEAR"
            signal_rows.append({
                "Signal": "Elevated Stress Score",
                "Pair": _display_pair_for_ui(pair_wsname) or "—",
                "Severity": sev,
                "Current Value": f"{stress_score:.2f}",
                "Threshold": "0.50 (WARNING) / 0.75 (HIGH)",
                "Status": status,
                "Notes": "Composite stress from spread, depth." if surv_is_majors else "Composite stress from spread, depth, peg.",
            })
        # Summary sentence
        if not active_signals:
            summary_sentence = "No active market-quality signals detected across monitored major pairs." if surv_is_majors else "No active surveillance signals detected across monitored stablecoin pairs."
        elif surv_is_majors:
            summary_sentence = "Some market-quality signals are elevated; review Active Signals below."
        elif basis_active and not pairs_with_peg:
            summary_sentence = "BTC-stablecoin basis divergence is elevated while stablecoin pegs remain orderly."
        elif pairs_with_peg or pairs_with_liquidity:
            concentrated = list(dict.fromkeys([_display_pair_for_ui(p) for p in pairs_with_peg + pairs_with_liquidity]))
            if len(concentrated) == 1:
                summary_sentence = f"Liquidity and peg warnings are currently concentrated in {concentrated[0]}."
            else:
                summary_sentence = f"Multiple surveillance signals active across {', '.join(concentrated)}."
        else:
            summary_sentence = "Some surveillance signals are elevated; review Active / Recent Signals below."
        st.markdown(f"**{summary_sentence}**")
        st.write("")
        active_rows = [r for r in signal_rows if r.get("Status") == "ACTIVE"]
        monitoring_rows = [r for r in signal_rows if r.get("Status") != "ACTIVE"]
        st.markdown('<p class="k-section-title">Active Signals</p>', unsafe_allow_html=True)
        if active_rows:
            df_active = pd.DataFrame(active_rows)
            st.dataframe(
                df_active,
                use_container_width=True,
                height=min(180, 28 + len(active_rows) * 28),
                hide_index=True,
                column_config={"Notes": st.column_config.TextColumn("Notes", width="large")},
            )
        else:
            st.caption("No active market-quality signals detected across monitored major pairs." if surv_is_majors else "No active surveillance signals detected across monitored stablecoin pairs.")
        st.markdown('<p class="k-section-title">Monitoring Checks</p>', unsafe_allow_html=True)
        if monitoring_rows:
            df_monitoring = pd.DataFrame(monitoring_rows)
            st.dataframe(
                df_monitoring,
                use_container_width=True,
                height=min(180, 28 + len(monitoring_rows) * 28),
                hide_index=True,
                column_config={"Notes": st.column_config.TextColumn("Notes", width="large")},
            )
        else:
            st.caption("No monitoring check rows.")
        st.markdown('<p class="k-section-title">Cross-Pair Monitoring</p>', unsafe_allow_html=True)
        cross_surv_rows = []
        surv_pair_list = ["XBT/USD", "XAU/USD"] if surv_is_majors else STABLECOIN_USD_PAIRS
        for p in surv_pair_list:
            try:
                s = compute_pair_snapshot(p)
                if s.get("mid") is None and s.get("depth_10bp") is None:
                    continue
                peg = s.get("peg_deviation_bp")
                spr = s.get("spread_bp")
                dep = s.get("depth_10bp")
                stress = s.get("stress_score")
                regime = s.get("liquidity_regime")
                count = 0
                if not surv_is_majors and peg is not None and abs(peg) >= peg_threshold_bp:
                    count += 1
                if dep is not None and depth_warning_threshold is not None and dep < depth_warning_threshold:
                    count += 1
                if spr is not None and spr > 5:
                    count += 1
                if stress is not None and stress > 0.50:
                    count += 1
                cross_surv_rows.append({
                    "Pair": _display_pair_for_ui(p),
                    "Peg Deviation (bp)": "—" if surv_is_majors else format_percent_like_bp(peg),
                    "Spread (bp)": format_percent_like_bp(spr),
                    "Depth ±10bp": format_compact_number(dep),
                    "Stress Score": f"{stress:.2f}" if stress is not None else "—",
                    "_stress_num": stress if stress is not None else -1.0,
                    "Regime": _format_regime_label(regime) if regime else "—",
                    "Signal Count": count,
                })
            except Exception:
                continue
        if cross_surv_rows:
            df_cross_surv = pd.DataFrame(cross_surv_rows)
            if surv_is_majors and "Peg Deviation (bp)" in df_cross_surv.columns:
                df_cross_surv = df_cross_surv.drop(columns=["Peg Deviation (bp)"])
            df_cross_surv = df_cross_surv.sort_values(
                ["Signal Count", "_stress_num"],
                ascending=[False, False],
            )
            df_cross_surv = df_cross_surv.drop(columns=["_stress_num"])
            st.dataframe(df_cross_surv, use_container_width=True, height=min(200, 32 + len(cross_surv_rows) * 30), hide_index=True)
        else:
            st.caption("No major pair data available for cross-pair monitoring." if surv_is_majors else "No stablecoin pair data available for cross-pair monitoring.")
        st.caption(
            "Rule-based surveillance from current order book and ticker data. "
            "Thresholds match sidebar and main dashboard settings."
        )

    with tab_audit:
        log_entries = get_audit_log()
        st.caption("Session log. Recent entries shown below.")
        if log_entries:
            recent_n = 15
            recent = list(log_entries)[-recent_n:] if len(log_entries) > recent_n else log_entries
            df_audit = pd.DataFrame(recent)
            st.dataframe(df_audit, use_container_width=True, height=min(300, 32 + len(recent) * 24), hide_index=True)
            if len(log_entries) > recent_n:
                with st.expander("Show full log"):
                    st.dataframe(pd.DataFrame(log_entries), use_container_width=True, height=400, hide_index=True)
        else:
            st.caption("No events yet. Pair changes, threshold changes, and API errors are logged here.")


def main() -> None:
    configure_page()
    asset_pairs, asset_pairs_error = get_asset_pairs()
    connection_status = "connected" if asset_pairs else "not connected"
    status_message: Optional[str] = None
    if asset_pairs_error:
        status_message = "Kraken AssetPairs error. See Pairs tab for details."
    elif not asset_pairs:
        status_message = "Kraken AssetPairs returned no data."

    with st.sidebar:
        render_sidebar_minimal(connection_status=connection_status, status_message=status_message)
    render_dashboard_header(connection_status)
    selections = render_top_controls(
        connection_status=connection_status,
        status_message=status_message,
        asset_pairs=asset_pairs,
    )
    # Log pair/time window changes to audit log
    last = st.session_state.get(LAST_SELECTIONS_KEY, {})
    cur_pair = selections.get("pair_wsname", "")
    cur_tf = selections.get("timeframe", "")
    if last.get("pair_wsname") != cur_pair or last.get("timeframe") != cur_tf:
        audit_log_append(
            EVENT_PAIR_CHANGE,
            f"Pair/time: {_display_pair_for_ui(cur_pair)} · {cur_tf}",
        )
    st.session_state[LAST_SELECTIONS_KEY] = {"pair_wsname": cur_pair, "timeframe": cur_tf}

    overview_context = render_main_area(
        timeframe=selections["timeframe"],
        base_asset=selections["base_asset"],
        quote_asset=selections["quote_asset"],
        asset_pairs=asset_pairs,
        pair_wsname=selections["pair_wsname"],
        hi_res=selections.get("hi_res", False),
        connection_status=connection_status,
        asset_group=selections.get("asset_group", "Stablecoins"),
    )
    render_tabs(
        selections=selections,
        asset_pairs=asset_pairs,
        asset_pairs_error=asset_pairs_error,
        overview_context=overview_context,
    )
    render_alerts_expander(asset_group=selections.get("asset_group", "Stablecoins"))


if __name__ == "__main__":
    main()

