"""
Depeg event detection for stablecoin peg monitoring.

Uses OHLC close series: event starts when deviation_bp >= threshold for >= N
consecutive points; ends when deviation_bp < threshold for >= 2 consecutive points.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# Default: 3 consecutive points above threshold to start event; 2 below to end
DEPEG_N_CONSECUTIVE_START = 3
DEPEG_N_CONSECUTIVE_END = 2


def detect_depeg_events(
    ohlc_df: pd.DataFrame,
    threshold_bp: float,
    peg_target: float = 1.0,
    n_consecutive_start: int = DEPEG_N_CONSECUTIVE_START,
    n_consecutive_end: int = DEPEG_N_CONSECUTIVE_END,
    time_col: str = "time",
    close_col: str = "close",
) -> List[Dict[str, Any]]:
    """
    Detect depeg events from OHLC close series.

    Event start: deviation_bp >= threshold_bp for >= n_consecutive_start points.
    Event end: deviation_bp < threshold_bp for >= n_consecutive_end points.

    Returns list of events: {start_time, end_time, duration_minutes, max_deviation_bp}.
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
    df["above"] = df["deviation_bp"] >= threshold_bp

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).reset_index(drop=True)

    i = 0
    while i < len(df):
        # Look for run of >= n_consecutive_start True
        run_start_idx = None
        run_devs: List[float] = []
        j = i
        while j < len(df) and df["above"].iloc[j]:
            if run_start_idx is None:
                run_start_idx = j
            run_devs.append(float(df["deviation_bp"].iloc[j]))
            j += 1
            if len(run_devs) >= n_consecutive_start:
                break
        if run_start_idx is None or len(run_devs) < n_consecutive_start:
            i = j if j > i else i + 1
            continue

        # We have an event start at run_start_idx. Now find end: >= n_consecutive_end False
        start_time = df[time_col].iloc[run_start_idx]
        max_dev = max(run_devs)
        end_idx = j
        below_count = 0
        while end_idx < len(df):
            if df["above"].iloc[end_idx]:
                below_count = 0
                max_dev = max(max_dev, float(df["deviation_bp"].iloc[end_idx]))
            else:
                below_count += 1
                if below_count >= n_consecutive_end:
                    end_time = df[time_col].iloc[end_idx]
                    duration_min = (end_time - start_time).total_seconds() / 60.0
                    events.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_minutes": round(duration_min, 2),
                        "max_deviation_bp": round(max_dev, 2),
                    })
                    end_idx += 1
                    break
            end_idx += 1
        else:
            # Run to end of series
            end_time = df[time_col].iloc[-1]
            duration_min = (end_time - start_time).total_seconds() / 60.0
            events.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": round(duration_min, 2),
                "max_deviation_bp": round(max_dev, 2),
            })
        i = end_idx

    return events
