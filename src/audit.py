"""
Session-scoped audit log for dashboard actions and API events.

Event types: PAIR_CHANGE, THRESHOLD_CHANGE, ALERT_TRIGGERED, ALERT_CLEARED, API_ERROR.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import streamlit as st

AUDIT_LOG_KEY = "kraken_audit_log"
EVENT_PAIR_CHANGE = "PAIR_CHANGE"
EVENT_THRESHOLD_CHANGE = "THRESHOLD_CHANGE"
EVENT_ALERT_TRIGGERED = "ALERT_TRIGGERED"
EVENT_ALERT_CLEARED = "ALERT_CLEARED"
EVENT_API_ERROR = "API_ERROR"
EVENT_SURVEILLANCE = "SURVEILLANCE"


def _ensure_audit_log() -> List[Dict[str, Any]]:
    if AUDIT_LOG_KEY not in st.session_state:
        st.session_state[AUDIT_LOG_KEY] = []
    return st.session_state[AUDIT_LOG_KEY]


def audit_log_append(event_type: str, details: str) -> None:
    """Append an audit entry: timestamp, event_type, details."""
    log = _ensure_audit_log()
    log.append({
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": event_type,
        "details": details,
    })


def get_audit_log() -> List[Dict[str, Any]]:
    """Return the current audit log (newest last if appended in order)."""
    return list(_ensure_audit_log())


def clear_audit_log() -> None:
    """Clear all audit entries (e.g. for testing)."""
    st.session_state[AUDIT_LOG_KEY] = []
