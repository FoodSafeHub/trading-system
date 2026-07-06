"""Shared CEEI-gate promote controls.

One widget set used by every "Promote to auto-trade" panel (Backtest page,
India Swing page) so the gate is configured identically everywhere. The gate
is opt-in: mode "none" (the default) promotes with all ceei_* fields None,
which is byte-identical to a pre-CEEI promote.
"""
from __future__ import annotations

import streamlit as st

CEEI_GATE_MODES = ["none", "setup", "trigger", "score"]

# Per-mode "when to use it" guidance, shown under the mode selector wherever the
# gate is configured. Sourced from the 49-symbol/5y confirmation study
# (output/ceei_meta/CEEI_CONFIRMATION.md).
CEEI_MODE_GUIDE = {
    "none": "Gate **off** — entries are unchanged (the default).",
    "trigger": "**Is CEEI firing right now?** Strictest — cuts 60-75% of trades for the "
               "biggest per-trade gains. **Use when** the strategy signals often and you "
               "want only the highest-conviction entries (confirmed for **sma_rsi** +51.7% "
               "and **momentum_breakout** +25.1% expectancy). Avoid on strategies that "
               "trade rarely — it may starve them.",
    "score": "**Is the environment good enough?** Mildest — keeps ~80% of trade flow. "
             "**Use when** trade frequency matters and you just want the worst-environment "
             "entries trimmed; the reasonable default if unsure. Threshold is the dial: "
             "55-60 behaves like trigger, lower is barely a filter.",
    "setup": "**Has it been coiling recently?** Passes if a compression setup occurred "
             "within the lookback — the breakout needn't be live yet. **Use when** the "
             "strategy's signal is slow or lagging (crossovers, ribbons — confirmed for "
             "**ema_crossover** +11.7%), where requiring a live trigger would reject good "
             "entries. Shorter lookback (5) = fresh coil only; longer (15-20) = permissive.",
}

# Name fragments of strategy families the CEEI meta-study showed the gate HARMS
# (mean-reversion / pullback / reversal). Warning badge only — never blocks.
_BAD_FAMILY_FRAGMENTS = (
    "rsi2", "pullback", "reversion", "reversal", "panic", "bollinger", "vwap",
    "fib", "dip", "bb_",
)


def ceei_family_incompatible(strategy_name: str) -> bool:
    name_l = (strategy_name or "").lower()
    return any(frag in name_l for frag in _BAD_FAMILY_FRAGMENTS)


def ceei_promote_controls(key_prefix: str, strategy_name: str = "",
                          default_gate: str = "none",
                          default_threshold: float = 48.0,
                          default_lookback: int = 10) -> tuple[str, float, int]:
    """Render the compact CEEI gate row (mode / threshold / lookback) and the
    incompatible-family warning. Returns (gate, threshold, lookback).
    Pass the backtest's tested values as defaults so a validated gate promotes
    with the same settings."""
    if default_gate not in CEEI_GATE_MODES:
        default_gate = "none"
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        gate = st.selectbox(
            "CEEI entry gate", CEEI_GATE_MODES,
            index=CEEI_GATE_MODES.index(default_gate),
            key=f"{key_prefix}_ceei_gate",
            help="Vetoes BUY entries unless the CEEI ignition condition holds "
                 "(exits never blocked; missing data fails open). trigger = "
                 "firing on the signal bar (confirmed best for sma_rsi / "
                 "momentum_breakout). score = composite above threshold "
                 "(milder). setup = coil within lookback. none = off (default). "
                 "Pre-filled from your backtest CEEI run if you did one.",
        )
    with c2:
        thr = st.number_input(
            "Score threshold", min_value=0.0, max_value=100.0,
            value=float(default_threshold), step=1.0,
            key=f"{key_prefix}_ceei_thr", disabled=(gate != "score"),
        )
    with c3:
        lb = st.number_input(
            "Setup lookback (bars)", min_value=1, max_value=60,
            value=int(default_lookback), step=1,
            key=f"{key_prefix}_ceei_lb", disabled=(gate != "setup"),
        )
    st.caption(CEEI_MODE_GUIDE[gate])
    if gate != "none" and ceei_family_incompatible(strategy_name):
        st.warning(
            "⚠️ This looks like a mean-reversion / pullback / reversal strategy — "
            "the CEEI meta-study showed the gate **reduces** expectancy for these "
            "families. You can still promote, but backtest the gated variant first.",
            icon="⚠️",
        )
    return gate, float(thr), int(lb)


def ceei_upsert_kwargs(gate: str, threshold: float, lookback: int) -> dict:
    """kwargs for api.upsert_assignment. All-None when the gate is off, so an
    ungated promote stays an exact no-op."""
    if not gate or gate == "none":
        return {"ceei_gate": None, "ceei_gate_enabled": None,
                "ceei_gate_threshold": None, "ceei_gate_lookback": None}
    return {"ceei_gate": gate,
            "ceei_gate_enabled": True,
            "ceei_gate_threshold": threshold if gate == "score" else None,
            "ceei_gate_lookback": lookback if gate == "setup" else None}


def ceei_note(gate: str, threshold: float, lookback: int) -> str:
    """Short human string for assignment notes / success messages."""
    if not gate or gate == "none":
        return "CEEI off"
    if gate == "score":
        return f"CEEI score≥{threshold:g}"
    if gate == "setup":
        return f"CEEI setup/{lookback}"
    return "CEEI trigger"
