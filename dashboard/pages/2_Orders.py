from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Orders", page_icon="📋", layout="wide")
st.title("📋 Orders")

# ══════════════════════════════════════════════════════════════
# SECTION 1 — PLACE MANUAL ORDER
# ══════════════════════════════════════════════════════════════
with st.expander("➕ Place Manual Order", expanded=False):
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

        # Show most useful columns first
        priority = ["created_at", "symbol", "side", "order_type", "quantity",
                    "fill_price", "status", "broker_order_id"]
        show_cols = [c for c in priority if c in df.columns] + \
                    [c for c in df.columns if c not in priority]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

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
