"""Effective-exit-policy descriptor for backtest results UI.

Renders a compact "which exit layers actually fired in this backtest" panel
so the trader can compare runs with/without the Chandelier trail (Layer 2) or
the Phase-1 exit_policy (Layer 3) — they only have to flip the override
toggles on the Single Strategy mode and re-run.

Layer model (mirrors the live engine in app/services/strategy/exits.py):
  * Layer 1 — the rule's own SELL signal. Always active.
  * Layer 2 — Chandelier trailing overlay. Active when params['trail_enabled'].
  * Layer 3 — Phase-1 exit_policy dict. Active when params['exit_policy'].

Anything else (broker-resting protective stop, daily-loss kill switch, the
backtest engine's end-of-data force-close) is outside the per-bar exit
decision and is not surfaced here.
"""
from __future__ import annotations

from typing import Any

# Layer-1 rule SELL descriptions, keyed by strategy_type. Values reach into
# the live params dict so per-symbol calibration is reflected.
_RULE_EXITS: dict[str, Any] = {
    # Live (in rules.py _RULE_REGISTRY)
    "rsi2_mean_reversion":  lambda p: f"RSI(2) > {p.get('rsi_exit_threshold', 70)} OR close > SMA({p.get('exit_sma', 5)})",
    "pullback_ema50":       lambda p: f"RSI > {p.get('exit_rsi', 65)} OR price > +{p.get('exit_extension_pct', 3.0)}% above EMA50",
    "vix_spike_reversal":   lambda p: f"ATR% < {p.get('atr_exit_threshold', 2.0)} OR RSI > {p.get('rsi_exit', 55)}",
    "ema_macd_crossover":   lambda p: "bearish EMA(9/21) cross OR MACD crosses below signal",
    "bb_squeeze_breakout":  lambda p: f"close < BB middle OR RSI > {p.get('rsi_overbought', 80)}",
    "bollinger":            lambda p: "at upper BB AND RSI > 70 (or mid-band break when not in uptrend)",
    "fib_pullback":         lambda p: "close >= swing high OR RSI > 70",
    "supertrend":           lambda p: "Supertrend flip bearish",
    "ema_ribbon":           lambda p: "fast EMA crosses below mid OR RSI > rsi_high",
    "breakout":             lambda p: "close drops below breakout level OR RSI > 75",
    "ema_crossover":        lambda p: "fast EMA crosses below slow",
    "macd":                 lambda p: "bearish MACD cross OR RSI > 75",
    "sma_rsi":              lambda p: "bearish SMA cross OR RSI > rsi_overbought",
    "vwap_rsi":             lambda p: "RSI crosses below 65 OR > 4% above VWAP",
    # Phase 1 unified (parallel, not live by default)
    "rsi2_reversion":       lambda p: f"RSI(2) > {p.get('rsi_exit_threshold', 65)} OR close > SMA({p.get('exit_sma', 5)})",
    "trend_pullback":       lambda p: f"RSI > {p.get('exit_rsi', 65)} OR +{p.get('exit_extension_pct', 3.0)}% above EMA50",
    "squeeze_breakout":     lambda p: f"close < EMA({p.get('exit_ema', 20)})",
    "momentum_breakout":    lambda p: "EMA9 < EMA21 (momentum loss)",
    "panic_reversal":       lambda p: f"ATR% < {p.get('atr_exit_threshold', 2.0)} OR RSI > {p.get('rsi_exit', 55)}",
    "trend_follow":         lambda p: "Supertrend flip bearish",
}


def describe_layers(strategy_type: str, params: dict[str, Any]) -> list[dict[str, str]]:
    """Return one dict per layer: {label, status, detail}.

    status is "ALWAYS" (Layer 1 — the rule always emits its own SELL),
    "ON" (active for this run), or "OFF" (inert).
    """
    out: list[dict[str, str]] = []

    rule_fn = _RULE_EXITS.get(strategy_type, lambda p: "(no description for this strategy type)")
    out.append({
        "label": "Layer 1: rule SELL",
        "status": "ALWAYS",
        "detail": rule_fn(params),
    })

    if params.get("trail_enabled"):
        trigger = params.get("trail_trigger_pct", 3.0)
        mult = params.get("atr_trail_mult", 3.0)
        period = params.get("atr_trail_period", 22)
        out.append({
            "label": "Layer 2: Chandelier trail",
            "status": "ON",
            "detail": f"trigger {trigger}% — ride winners until close < peak_close − {mult}×ATR({period})",
        })
    else:
        out.append({
            "label": "Layer 2: Chandelier trail",
            "status": "OFF",
            "detail": "—",
        })

    policy = params.get("exit_policy")
    if isinstance(policy, dict) and policy:
        parts: list[str] = []
        trail = policy.get("trail")
        if trail and trail != "none":
            parts.append(f"{trail} trail {policy.get('atr_mult', 3.0)}×ATR (trigger {policy.get('trigger_pct', 2.0)}%)")
        tsb = policy.get("time_stop_bars")
        if tsb:
            parts.append(f"time_stop {tsb} bars")
        tf = policy.get("trend_fail")
        if tf and tf != "none":
            parts.append(f"trend_fail = {tf}")
        re_ = policy.get("regime_exit")
        if re_ and re_ != "none":
            parts.append(f"regime_exit = {re_}")
        out.append({
            "label": "Layer 3: exit_policy",
            "status": "ON",
            "detail": " + ".join(parts) if parts else "(empty policy)",
        })
    else:
        out.append({"label": "Layer 3: exit_policy", "status": "OFF", "detail": "—"})

    # Layer 0: hard stop-loss (runs in engine before strategy signal)
    sl = params.get("stop_loss_pct", 8.0)
    if sl and float(sl) > 0:
        out.insert(0, {
            "label": "Layer 0: hard stop-loss",
            "status": "ON",
            "detail": f"exit if unrealized loss >= {sl:.0f}% from entry (engine-level, cannot be overridden)",
        })
    else:
        out.insert(0, {
            "label": "Layer 0: hard stop-loss",
            "status": "OFF",
            "detail": "stop_loss_pct=0 — no hard stop",
        })

    return out


def render(strategy_type: str, params: dict[str, Any], *, st=None, key_prefix: str = "") -> None:
    """Render the descriptor as a Streamlit caption block."""
    if st is None:
        import streamlit as st  # noqa: WPS433
    layers = describe_layers(strategy_type, params)
    active_n = sum(1 for l in layers if l["status"] in ("ON", "ALWAYS"))
    with st.expander(f"Effective exit policy — {active_n} layer(s) active", expanded=False):
        for l in layers:
            color = {"ALWAYS": "#90caf9", "ON": "#26a69a", "OFF": "#9ba3b1"}[l["status"]]
            st.markdown(
                f"<div style='margin:2px 0'><span style='display:inline-block;min-width:80px;"
                f"padding:1px 8px;border-radius:8px;background:#1a1f2c;color:{color};"
                f"font-size:11px;text-align:center'>{l['status']}</span> "
                f"<span style='color:#e6e6e6'>{l['label']}</span> — "
                f"<span style='color:#9ba3b1'>{l['detail']}</span></div>",
                unsafe_allow_html=True,
            )


def render_inline(strategy_type: str, params: dict[str, Any]) -> str:
    """Compact one-line summary for tables. Returns plain text."""
    layers = describe_layers(strategy_type, params)
    on = [l for l in layers if l["status"] == "ON"]
    if not on:
        return "Layer 1 only"
    return "+ " + " + ".join(l["label"].split(":", 1)[1].strip() for l in on)
