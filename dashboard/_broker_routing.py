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


def render_broker_routing_toggle(*, key_suffix: str = "") -> None:
    """Render the broker-routing radio. Safe to call from multiple pages.

    `key_suffix` keeps Streamlit widget keys unique when this is rendered on
    more than one page in the same session.
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
                "'Zerodha (India) only' routes ALL orders to the India broker — US "
                "symbols won't trade; for a mix of US + India use per-assignment "
                "broker routing on the Strategy page instead. "
                "Webull live trading is not yet implemented — selecting Webull or "
                "Both will currently fail at the Webull leg."
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
            st.success(f"Broker routing -> {picked}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to update broker routing: {e}")
