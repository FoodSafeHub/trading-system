"""Shared page-level UI components for the trading dashboard.

These sit one level above the raw CSS helpers in _theme.py. Every significant
UI pattern that appears on two or more pages should live here so changes
propagate everywhere without hunting through 14 page files.

Usage pattern::

    from _components import page_header, stat_band, filter_row, result_panel

    page_header("Day Trading", subtitle="Intraday strategies on 5m and 15m bars")
    stat_band([("Market", "OPEN", "green"), ("Regime", "BULL_OPEN", "teal")])

    with filter_row():
        symbol = st.text_input("Symbol", value="SPY")
        period = st.selectbox("Period", ["1d", "5d", "30d"])
        run = st.button("Run", type="primary")
"""
from __future__ import annotations

from typing import Iterable, Sequence
import streamlit as st

from _theme import pill, divider, empty_state  # noqa: F401 — re-export for convenience


# ── Page shell ───────────────────────────────────────────────────────────────


def page_header(
    title: str,
    *,
    subtitle: str = "",
    badge: str = "",
    badge_color: str = "grey",
    actions: str = "",           # raw HTML injected to the right of the title
) -> None:
    """Render the standardised page title block.

    Produces a single visual unit: title + optional subtitle + optional badge.
    Replaces the per-page ``st.title() + st.caption()`` pair so every page has
    the same rhythm and spacing.

    Args:
        title:       Page name shown prominently.
        subtitle:    One-line description, rendered in muted text below the title.
        badge:       Short status text rendered as a pill next to the title.
        badge_color: Pill colour — green | red | amber | grey | blue | teal.
        actions:     Raw HTML snippet (e.g., link or pill) injected right-aligned.
    """
    badge_html = f"&nbsp;{pill(badge, badge_color)}" if badge else ""
    actions_html = f"<span style='margin-left:auto'>{actions}</span>" if actions else ""
    sub_html = (
        f"<div style='font-size:0.8rem;color:var(--text-3);margin-top:3px;"
        f"line-height:1.4'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(
        f"<div class='tx-page-header'>"
        f"  <div>"
        f"    <span class='tx-page-title'>{title}{badge_html}</span>"
        f"    {sub_html}"
        f"  </div>"
        f"  {actions_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Status / stat band ───────────────────────────────────────────────────────


def stat_band(items: Iterable[tuple[str, str, str]]) -> None:
    """Render the horizontal stat/status strip.

    Args:
        items: Sequence of ``(label, text, color)`` tuples. Color is the pill
               colour: green | red | amber | grey | blue | teal.

    This is the same as ``_theme.status_row()`` but uses the new CSS class name
    ``.tx-statband`` so it benefits from the refreshed token values.
    """
    parts = []
    for label, text, color in items:
        parts.append(
            f"<span class='tx-stat'>"
            f"<span class='tx-stat-label'>{label}</span>{pill(text, color)}"
            f"</span>"
        )
    st.markdown(
        "<div class='tx-statband'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


# ── Filter toolbar ────────────────────────────────────────────────────────────


class filter_row:
    """Context manager that wraps a row of filter controls in a styled toolbar.

    Usage::

        with filter_row():
            sym = st.text_input("Symbol", ...)
            per = st.selectbox("Period", [...])
            btn = st.button("Run", type="primary")
    """

    def __enter__(self):
        self._container = st.container()
        self._container.markdown(
            "<div class='tx-filterbar'>", unsafe_allow_html=True
        )
        return self._container

    def __exit__(self, *_):
        self._container.markdown("</div>", unsafe_allow_html=True)


def filter_cols(*ratios: float | int, gap: str = "small"):
    """Return columns pre-sized for a filter toolbar.

    Wraps ``st.columns()`` with sensible defaults. Pass relative widths::

        sym_col, strat_col, _, run_col = filter_cols(2, 2, 3, 1)
        sym = sym_col.text_input("Symbol")
        run_col.button("Run", type="primary", use_container_width=True)
    """
    return st.columns(list(ratios) if ratios else [1], gap=gap)


# ── Result panels ─────────────────────────────────────────────────────────────


def result_panel(
    label: str,
    *,
    border: bool = True,
    collapsed: bool = False,
) -> "st.delta_generator.DeltaGenerator":  # type: ignore[name-defined]
    """Return an expander styled as a result section.

    Use for collapsible sub-results (pipeline diagnostics, trade logs, etc.)::

        with result_panel("Pipeline Diagnostics", collapsed=True):
            st.dataframe(diag_df)
    """
    return st.expander(label, expanded=not collapsed)


def metric_band(
    items: Sequence[tuple[str, str]] | Sequence[tuple[str, str, str | None]],
    *,
    cols: int | None = None,
    gap: str = "small",
) -> None:
    """Render KPI metrics in a tight horizontal band.

    Accepts ``(label, value)`` or ``(label, value, delta)`` tuples. ``cols``
    overrides the automatic column count (useful for wide layouts where you want
    fixed-width columns rather than evenly split).
    """
    if not items:
        return
    n = cols or len(items)
    columns = st.columns(n, gap=gap)
    for i, item in enumerate(items):
        col = columns[i % n]
        if len(item) == 2:
            col.metric(item[0], item[1])
        else:
            col.metric(item[0], item[1], delta=item[2])


# ── Loading / error state helpers ─────────────────────────────────────────────


def loading_placeholder(message: str = "Loading…") -> None:
    """Show a consistent loading message."""
    st.markdown(
        f"<div class='tx-empty' style='border-style:solid'>"
        f"<div class='tx-empty-icon'>⏳</div>"
        f"<div class='tx-empty-title'>{message}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def error_state(
    title: str = "Something went wrong",
    body: str = "Check the backend is running and refresh the page.",
) -> None:
    """Show a formatted error state."""
    st.markdown(
        f"<div class='tx-empty' style='border-color:var(--neg-bd)'>"
        f"<div class='tx-empty-icon'>⚠️</div>"
        f"<div class='tx-empty-title' style='color:var(--neg)'>{title}</div>"
        f"<div class='tx-empty-body'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Section label (alternative to section() for sub-sections) ────────────────


def label(text: str, *, muted: bool = False) -> None:
    """Render a compact section label (smaller than H3, larger than a caption)."""
    color = "var(--text-3)" if muted else "var(--text-2)"
    st.markdown(
        f"<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.09em;color:{color};margin:var(--sp-4) 0 var(--sp-2) 0'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )


# ── Inline colored value helpers ─────────────────────────────────────────────


def colored_value(value: float | str, *, threshold: float = 0) -> str:
    """Return HTML-colored value — green if positive, red if negative/zero."""
    try:
        v = float(value)
        cls = "tx-pos" if v > threshold else "tx-neg"
    except (TypeError, ValueError):
        cls = "tx-muted"
    return f"<span class='{cls}'>{value}</span>"


def regime_badge(regime: str) -> str:
    """Return a pill HTML string for a regime name."""
    _map = {
        "BULL_OPEN":  ("BULL",  "green"),
        "BEAR_OPEN":  ("BEAR",  "red"),
        "CHOPPY":     ("CHOPPY","amber"),
        "PRE_MARKET": ("PRE",   "grey"),
        "TREND_UP":   ("TREND↑","green"),
        "TREND_DOWN": ("TREND↓","red"),
        "HIGH_VOL":   ("HI-VOL","amber"),
        "NEWS_RISK":  ("NEWS",  "red"),
        "UNKNOWN":    ("UNKN",  "grey"),
    }
    text, color = _map.get(regime, (regime[:6], "grey"))
    return pill(text, color)


def direction_badge(direction: str) -> str:
    """Return a pill for a trade direction."""
    _map = {
        "BUY":        ("BUY",   "green"),
        "SELL":       ("SELL",  "red"),
        "SELL_SHORT": ("SHORT", "red"),
        "HOLD":       ("HOLD",  "grey"),
    }
    text, color = _map.get(direction.upper(), (direction, "grey"))
    return pill(text, color)


# ── Trading workflow components ───────────────────────────────────────────────


def regime_chip(regime: str) -> str:
    """Return an HTML `.tx-regime` chip for a market regime string.

    Wider than a pill — meant for display inside a card header or stat band.
    """
    _map = {
        "BULL_OPEN":  ("BULL OPEN",  "bull"),
        "BEAR_OPEN":  ("BEAR OPEN",  "bear"),
        "CHOPPY":     ("CHOPPY",     "chop"),
        "PRE_MARKET": ("PRE-MARKET", "pre"),
        "TREND_UP":   ("TREND UP",   "bull"),
        "TREND_DOWN": ("TREND DOWN", "bear"),
        "HIGH_VOL":   ("HIGH VOL",   "chop"),
        "NEWS_RISK":  ("NEWS RISK",  "bear"),
        "UNKNOWN":    ("UNKNOWN",    "unkn"),
    }
    text, cls = _map.get(regime, (regime, "unkn"))
    return f"<span class='tx-regime {cls}'>{text}</span>"


def eligibility_chip(state: str, reason: str = "") -> str:
    """Return an HTML `.tx-chip` for a symbol's trading eligibility state.

    States: 'ready' | 'watch' | 'blocked' | 'idle'.
    """
    _map = {
        "ready":   ("● READY",   "ready"),
        "watch":   ("◐ WATCH",   "watch"),
        "blocked": ("✕ BLOCKED", "blocked"),
        "idle":    ("○ IDLE",    "idle"),
    }
    text, cls = _map.get(state.lower(), (state.upper(), "idle"))
    title = f' title="{reason}"' if reason else ""
    return f"<span class='tx-chip {cls}'{title}>{text}</span>"


def risk_gauge_html(label: str, used: float, maximum: float, *, prefix: str = "$") -> str:
    """Return an HTML string for a labelled risk gauge row (no Streamlit widgets).

    Use before ``st.progress()`` to get a styled label row above the bar::

        st.markdown(risk_gauge_html("Daily loss", 320, 1000), unsafe_allow_html=True)
        st.progress(320 / 1000)
    """
    pct = used / max(maximum, 1)
    cls = "danger" if pct >= 0.85 else ("warn" if pct >= 0.60 else "ok")
    val_str = f"{prefix}{abs(used):,.0f} / {prefix}{abs(maximum):,.0f}"
    return (
        f"<div class='tx-gauge-row'>"
        f"<span>{label}</span>"
        f"<span class='tx-gauge-val {cls}'>{val_str}</span>"
        f"</div>"
    )
