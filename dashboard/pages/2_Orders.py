from __future__ import annotations

import json
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme

import streamlit as st
import pandas as pd


def _derive_source(row: pd.Series) -> str:
    """Label where an order came from.

    Reads the authoritative `source` column written by ExecutionService. The
    older rows (before the column existed) are tagged `unknown_pre_migration`
    in the DB and surfaced as "Unknown (pre-fix)" so users see they're not
    confirmed manual.
    """
    src = row.get("source")
    strategy = row.get("strategy_name") if pd.notna(row.get("strategy_name")) else None
    if src == "scheduler":
        return f"Scheduler: {strategy}" if strategy else "Scheduler"
    if src == "scanner":
        return f"Scanner: {strategy}" if strategy else "Scanner"
    if src == "autotrader":
        return "Autotrader"
    if src == "unknown_pre_migration":
        return "Unknown (pre-fix)"
    if src == "manual":
        return "Manual"
    # Fallback for rows missing the column entirely — shouldn't happen.
    if strategy:
        return f"Strategy: {strategy}"
    return "Manual"


def _planned_exits(preview_json: object) -> str:
    """Pull TP / SL from preview_json if the strategy wrote them, else em-dash."""
    if not preview_json or not isinstance(preview_json, str):
        return "—"
    try:
        pj = json.loads(preview_json)
    except Exception:
        return "—"
    tp = pj.get("take_profit") or pj.get("tp") or pj.get("target_price")
    sl = pj.get("stop_loss") or pj.get("sl") or pj.get("stop_price")
    if tp is None and sl is None:
        return "—"
    parts = []
    if tp is not None:
        parts.append(f"TP ${float(tp):.2f}")
    if sl is not None:
        parts.append(f"SL ${float(sl):.2f}")
    return " · ".join(parts)

apply_theme("Orders")
st.title("Orders")

# ══════════════════════════════════════════════════════════════
# SECTION 1 — PLACE MANUAL ORDER
# ══════════════════════════════════════════════════════════════
with st.expander("Place Manual Order", expanded=False):
    st.caption("Routed through the full risk engine — order will be blocked if kill switch is active or outside market hours.")
    with st.form("manual_order"):
        c1, c2, c3, c4, c5 = st.columns(5)
        symbol     = c1.text_input("Symbol", value="SPY").upper()
        side       = c2.selectbox("Side", ["BUY", "SELL"])
        order_type = c3.selectbox("Type", ["MARKET", "LIMIT"])
        quantity   = c4.number_input("Qty", min_value=0.01, value=1.0, step=1.0)
        limit_price = c5.number_input("Limit Price", min_value=0.0, value=0.0, step=0.01)

        submitted = st.form_submit_button("Submit Order", type="primary")
        if submitted:
            try:
                lp = limit_price if order_type == "LIMIT" and limit_price > 0 else None
                result = api.place_order(symbol, side, order_type, quantity, lp)
                st.success("Order submitted.")
                st.json(result)
            except Exception as e:
                st.error(f"Order failed: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — ORDER HISTORY (DB)
# ══════════════════════════════════════════════════════════════
st.subheader("Order History (this session)")
st.caption("Orders placed through this system — stored in the local database.")

try:
    ords = api.orders()
    if ords:
        df = pd.DataFrame(ords)

        statuses = ["All"] + sorted(df["status"].unique().tolist()) if "status" in df.columns else ["All"]
        chosen = st.selectbox("Filter by status", statuses, key="db_filter")
        if chosen != "All":
            df = df[df["status"] == chosen]

        def _status_tag(v):
            v = str(v)
            if v == "filled":    return "✅ filled"
            if v == "rejected":  return "❌ rejected"
            if v == "cancelled": return "🚫 cancelled"
            if v in ("pending", "working"): return "⏳ " + v
            return v

        if "status" in df.columns:
            df["status"] = df["status"].apply(_status_tag)

        # Derive the two columns that disambiguate manual vs. strategy orders.
        # Manual orders have no signal_id / no preview_json; strategy orders
        # carry both and we surface their planned TP/SL inline.
        df["source"] = df.apply(_derive_source, axis=1)
        df["planned_exit"] = df["preview_json"].apply(_planned_exits) \
            if "preview_json" in df.columns else "—"

        # Show most useful columns first. Source and planned_exit go right next
        # to status so the operator can answer "where did this come from?" and
        # "what's the exit?" without scrolling.
        # Synthesise a human-readable "trail" column for TRAILING_STOP orders.
        if "order_type" in df.columns:
            def _trail_label(row):
                if str(row.get("order_type", "")).upper() != "TRAILING_STOP":
                    return "—"
                val = row.get("trail_value")
                typ = str(row.get("trail_type", "PERCENT")).upper()
                if val is None:
                    return "trailing"
                return f"{val:.1f}%" if typ == "PERCENT" else f"${val:.2f}"
            df["trail"] = df.apply(_trail_label, axis=1)

        priority = ["created_at", "symbol", "side", "order_type", "trail", "quantity",
                    "fill_price", "status", "source", "planned_exit",
                    "broker_order_id"]
        # Hide raw JSON / internal plumbing — noise in the table.
        hidden = {"preview_json", "signal_id", "strategy_name", "source",
                  "trail_type", "trail_value"}
        show_cols = [c for c in priority if c in df.columns] + \
                    [c for c in df.columns if c not in priority and c not in hidden]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
        st.caption(
            "**Source** shows whether an order came from a strategy (with the strategy name) "
            "or was placed manually. **Planned exit** shows the TP / SL the strategy recorded; "
            "manual orders have no planned exit because no strategy set one."
        )

        # Cancel button
        if "broker_order_id" in df.columns:
            pending_ids = [
                str(r["broker_order_id"]) for _, r in df.iterrows()
                if "pending" in str(r.get("status", ""))
            ]
            if pending_ids:
                st.markdown("**Cancel a pending order**")
                to_cancel = st.selectbox("Select order to cancel", pending_ids, key="cancel_sel")
                confirm = st.checkbox(
                    f"Confirm cancellation of order {to_cancel}",
                    key=f"cancel_confirm_{to_cancel}",
                )
                if st.button("🚫 Cancel Order", key="cancel_btn", disabled=not confirm):
                    try:
                        api._post(f"/orders/{to_cancel}/cancel", {})
                        st.success(f"Cancellation sent for {to_cancel}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Cancel failed: {e}")
    else:
        st.info("No orders in database yet. Place a manual order above or wait for the scheduler to fire.")
except Exception as e:
    st.warning(f"Could not load order history: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — LIVE BROKER ORDERS (Schwab / paper)
# ══════════════════════════════════════════════════════════════
st.subheader("Live Broker Orders")
st.caption(
    "Orders fetched directly from your broker (Schwab or paper). "
    "This shows what the broker actually has on record — including orders placed outside this system."
)

col1, col2 = st.columns([1, 4])
with col1:
    refresh = st.button("🔄 Fetch from Broker", type="primary", key="broker_refresh", use_container_width=True)

if refresh:
    with st.spinner("Fetching orders from broker..."):
        try:
            broker_orders = api._get("/orders/broker")
            st.session_state["broker_orders"] = broker_orders
        except Exception as e:
            # If endpoint doesn't exist yet, explain clearly
            st.error(f"Could not fetch broker orders: {e}")

broker_ords = st.session_state.get("broker_orders")
if broker_ords is not None:
    if not broker_ords:
        st.info("Broker reports no orders on record.")
    else:
        bdf = pd.DataFrame(broker_ords)

        if "status" in bdf.columns:
            bdf["status"] = bdf["status"].apply(_status_tag)

        st.dataframe(bdf, use_container_width=True, hide_index=True)
        st.caption(f"{len(broker_ords)} order(s) on broker record.")
else:
    st.info("Click 'Fetch from Broker' to load live order status from Schwab.")
