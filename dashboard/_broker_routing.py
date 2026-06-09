from __future__ import annotations

import streamlit as st

import api

_ROUTING_LABELS = {
    "auto": "Auto (use active_broker)",
    "schwab": "Schwab only",
    "webull": "Webull only",
    "zerodha": "Zerodha (India) only",
    "both": "Both (Schwab + Webull)",
    "paper": "Paper",
}
_OPTIONS = ["auto", "schwab", "webull", "zerodha", "both", "paper"]

# Session-state key for the day-trading-specific broker selection.
# This does NOT write to .env — it is local to the dashboard session and
# is passed per-request to the autotrader, keeping it fully isolated from
# the Strategy page's global broker routing.
_DT_BROKER_KEY = "_dt_broker_routing"


def render_broker_routing_toggle(*, key_suffix: str = "") -> None:
    """Render the GLOBAL broker-routing radio (Strategy page and other pages).

    Writes to .env via the settings API — affects all live order routing
    platform-wide. Do NOT call this from the Day Trading page; use
    render_daytrading_broker_toggle() there instead.

    `key_suffix` keeps Streamlit widget keys unique across pages.
    """
    try:
        resp = api.settings_get_trade_routing() or {}
    except Exception as e:
        st.warning(
            f"Broker-routing toggle unavailable: {e}. "
            "Restart the FastAPI backend to pick up the /settings route."
        )
        return

    current = resp.get("trade_routing", "auto")
    idx = _OPTIONS.index(current) if current in _OPTIONS else 0
    cols = st.columns([3, 2])
    with cols[0]:
        picked = st.radio(
            "Broker routing",
            options=_OPTIONS,
            index=idx,
            format_func=lambda v: _ROUTING_LABELS.get(v, v),
            horizontal=True,
            key=f"trade_routing_radio_{key_suffix}" if key_suffix else "trade_routing_radio",
            help=(
                "Where live orders are sent. 'Both' fans out to Schwab + Webull (US). "
                "'Zerodha (India) only' routes ALL orders to the India broker. "
                "Webull live trading is not yet fully implemented."
            ),
        )
    with cols[1]:
        eff = resp.get("effective_brokers", [])
        st.markdown(
            f"<div style='padding-top:1.6em;color:#888;font-size:0.85em;'>"
            f"Routes to: <b>{', '.join(eff) or '—'}</b></div>",
            unsafe_allow_html=True,
        )
    if picked != current:
        try:
            api.settings_set_trade_routing(picked)
            st.success(f"Broker routing → {picked}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to update broker routing: {e}")


def render_daytrading_broker_toggle(*, key: str = "dt_broker_select") -> str:
    """Render the DAY TRADING broker selector.

    Isolated from the global platform routing — stores selection in
    Streamlit session_state only. Returns the selected broker name so
    the caller can pass it directly to the autotrader start request.

    Changing this does NOT affect the Strategy page or any other
    live swing-trading orders.
    """
    # Default: inherit the current global setting on first render
    if _DT_BROKER_KEY not in st.session_state:
        try:
            resp = api.settings_get_trade_routing() or {}
            st.session_state[_DT_BROKER_KEY] = resp.get("trade_routing", "paper")
        except Exception:
            st.session_state[_DT_BROKER_KEY] = "paper"

    current = st.session_state[_DT_BROKER_KEY]
    idx = _OPTIONS.index(current) if current in _OPTIONS else _OPTIONS.index("paper")

    cols = st.columns([3, 1])
    with cols[0]:
        picked = st.radio(
            "Day trading broker",
            options=_OPTIONS,
            index=idx,
            format_func=lambda v: _ROUTING_LABELS.get(v, v),
            horizontal=True,
            key=key,
            help=(
                "Broker for day trading orders only. "
                "This is isolated from the Strategy page — changes here do not "
                "affect swing-trade order routing. "
                "'Paper' = simulation, no real orders sent."
            ),
        )
    with cols[1]:
        _broker_label = _ROUTING_LABELS.get(picked, picked)
        st.markdown(
            f"<div style='padding-top:1.6em;color:#888;font-size:0.85em'>"
            f"Day trades → <b>{_broker_label}</b></div>",
            unsafe_allow_html=True,
        )

    st.session_state[_DT_BROKER_KEY] = picked
    return picked


def get_daytrading_broker() -> str:
    """Return the current day-trading broker selection (session-local)."""
    return st.session_state.get(_DT_BROKER_KEY, "paper")
