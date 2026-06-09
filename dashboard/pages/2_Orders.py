from __future__ import annotations

import json
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, divider, pill, empty_state
from _components import page_header, stat_band, filter_cols

import streamlit as st
import pandas as pd


def _derive_source(row: pd.Series) -> str:
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
    if strategy:
        return f"Strategy: {strategy}"
    return "Manual"


def _planned_exits(preview_json: object) -> str:
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


def _status_tag(v: str) -> str:
    v = str(v)
    if v == "filled":    return "✅ filled"
    if v == "rejected":  return "❌ rejected"
    if v == "cancelled": return "🚫 cancelled"
    if v in ("pending", "working"): return "⏳ " + v
    return v


apply_theme("Orders")

from _sidebar import render_sidebar
render_sidebar()

page_header(
    "Orders",
    subtitle="Order history from the local database and live broker. Routed through the full risk engine.",
)

# ── Manual order form ─────────────────────────────────────────────────────────
with st.expander("Place Manual Order", expanded=False):
    st.caption("Blocked if kill switch is active or outside market hours.")
    with st.form("manual_order"):
        c1, c2, c3, c4, c5 = st.columns(5)
        symbol      = c1.text_input("Symbol", value="SPY").upper()
        side        = c2.selectbox("Side", ["BUY", "SELL"])
        order_type  = c3.selectbox("Type", ["MARKET", "LIMIT"])
        quantity    = c4.number_input("Qty", min_value=0.01, value=1.0, step=1.0)
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

divider()

# ── Order history ─────────────────────────────────────────────────────────────
section("Order History", "Orders from this system — stored in the local database.")

try:
    ords = api.orders()
    if ords:
        df = pd.DataFrame(ords)

        # Status filter
        statuses = ["All"] + sorted(df["status"].unique().tolist()) if "status" in df.columns else ["All"]
        fil_col, _ = st.columns([2, 6])
        chosen = fil_col.selectbox("Filter by status", statuses, key="db_filter")
        if chosen != "All":
            df = df[df["status"] == chosen]

        if "status" in df.columns:
            df["status"] = df["status"].apply(_status_tag)

        df["source"] = df.apply(_derive_source, axis=1)
        df["planned_exit"] = df["preview_json"].apply(_planned_exits) \
            if "preview_json" in df.columns else "—"

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
                    "fill_price", "status", "source", "planned_exit", "broker_order_id"]
        hidden = {"preview_json", "signal_id", "strategy_name",
                  "trail_type", "trail_value"}
        show_cols = [c for c in priority if c in df.columns] + \
                    [c for c in df.columns if c not in priority and c not in hidden]

        st.dataframe(
            df[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "created_at":      st.column_config.TextColumn("Time",         width="medium"),
                "symbol":          st.column_config.TextColumn("Symbol",       width="small"),
                "side":            st.column_config.TextColumn("Side",         width="small"),
                "order_type":      st.column_config.TextColumn("Type",         width="small"),
                "trail":           st.column_config.TextColumn("Trail",        width="small"),
                "quantity":        st.column_config.NumberColumn("Qty",        format="%.2f", width="small"),
                "fill_price":      st.column_config.NumberColumn("Fill $",     format="$%.4f", width="small"),
                "status":          st.column_config.TextColumn("Status",       width="medium"),
                "source":          st.column_config.TextColumn("Source",       width="large"),
                "planned_exit":    st.column_config.TextColumn("Exit plan",    width="medium"),
                "broker_order_id": st.column_config.TextColumn("Broker ID",   width="medium"),
            },
        )
        st.caption(
            "**Source** shows whether an order came from a strategy or was placed manually. "
            "**Exit plan** shows TP / SL the strategy recorded; manual orders have none."
        )

        # Cancel pending
        if "broker_order_id" in df.columns:
            pending_ids = [
                str(r["broker_order_id"]) for _, r in df.iterrows()
                if "pending" in str(r.get("status", ""))
            ]
            if pending_ids:
                section("Cancel a Pending Order", level=3)
                to_cancel = st.selectbox("Select order to cancel", pending_ids, key="cancel_sel")
                confirm = st.checkbox(f"Confirm cancellation of order {to_cancel}", key=f"cancel_confirm_{to_cancel}")
                if st.button("🚫 Cancel Order", key="cancel_btn", disabled=not confirm):
                    try:
                        api._post(f"/orders/{to_cancel}/cancel", {})
                        st.success(f"Cancellation sent for {to_cancel}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Cancel failed: {e}")
    else:
        empty_state(
            "No orders yet",
            "Place a manual order above or wait for the scheduler to fire on an assigned symbol.",
            icon="📋",
        )
except Exception as e:
    st.warning(f"Could not load order history: {e}", icon="⚠️")

divider()

# ── Live broker orders ────────────────────────────────────────────────────────
section("Live Broker Orders", "Fetched directly from your broker — includes orders placed outside this system.")

fetch_col, _ = st.columns([2, 8])
refresh = fetch_col.button("🔄 Fetch from Broker", type="primary", key="broker_refresh", use_container_width=True)

if refresh:
    with st.spinner("Fetching orders from broker…"):
        try:
            broker_orders = api._get("/orders/broker")
            st.session_state["broker_orders"] = broker_orders
        except Exception as e:
            st.error(f"Could not fetch broker orders: {e}")

broker_ords = st.session_state.get("broker_orders")
if broker_ords is not None:
    if not broker_ords:
        empty_state("No broker orders", "The broker reports no orders on record.", icon="📭")
    else:
        bdf = pd.DataFrame(broker_ords)
        if "status" in bdf.columns:
            bdf["status"] = bdf["status"].apply(_status_tag)
        st.dataframe(bdf, use_container_width=True, hide_index=True)
        st.caption(f"{len(broker_ords)} order(s) on broker record.")
else:
    st.caption("Click **Fetch from Broker** above to load live order status from Schwab.")
