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

State vocabulary
----------------
The platform uses these states everywhere (Scanner, DayTrading, Risk, Strategy):

  READY    — conditions met, can enter/act
  WATCH    — marginal/partial conditions; monitor but be cautious
  ACTIVE   — long or short position currently open
  TRAILING — position in trailing-stop management phase
  PARTIAL  — partial exit taken, remainder being managed
  BLOCKED  — hard gate preventing action (kill switch, risk limit, regime)
  IDLE     — no position, no signal, normal standby
  COOLDOWN — post-trade lockout period before next entry allowed

Use eligibility_chip(state) to render any of these consistently.
Use bot_state_chip(raw_state) to map raw backend state strings to the vocabulary.
Use blocker_label(reason_key) to get a human-readable blocker description.
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
    actions: str = "",
) -> None:
    """Render the standardised page title block."""
    badge_html   = f"&nbsp;{pill(badge, badge_color)}" if badge else ""
    actions_html = f"<span style='margin-left:auto'>{actions}</span>" if actions else ""
    sub_html = (
        f"<div style='font-size:0.8rem;color:var(--text-3);margin-top:3px;line-height:1.4'>"
        f"{subtitle}</div>"
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
    """Render the horizontal stat/status strip from ``(label, text, color)`` tuples."""
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
    """Context manager that wraps filter controls in a styled toolbar."""

    def __enter__(self):
        self._container = st.container()
        self._container.markdown("<div class='tx-filterbar'>", unsafe_allow_html=True)
        return self._container

    def __exit__(self, *_):
        self._container.markdown("</div>", unsafe_allow_html=True)


def filter_cols(*ratios: float | int, gap: str = "small"):
    """Return columns pre-sized for a filter toolbar."""
    return st.columns(list(ratios) if ratios else [1], gap=gap)


# ── Result panels ─────────────────────────────────────────────────────────────


def result_panel(label: str, *, collapsed: bool = False):
    """Return an expander styled as a result section."""
    return st.expander(label, expanded=not collapsed)


def metric_band(
    items: Sequence[tuple[str, str]] | Sequence[tuple[str, str, str | None]],
    *,
    cols: int | None = None,
    gap: str = "small",
) -> None:
    """Render KPI metrics in a tight horizontal band."""
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
    st.markdown(
        f"<div class='tx-empty' style='border-color:var(--neg-bd)'>"
        f"<div class='tx-empty-icon'>⚠️</div>"
        f"<div class='tx-empty-title' style='color:var(--neg)'>{title}</div>"
        f"<div class='tx-empty-body'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Section label ─────────────────────────────────────────────────────────────


def label(text: str, *, muted: bool = False) -> None:
    color = "var(--text-3)" if muted else "var(--text-2)"
    st.markdown(
        f"<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.09em;color:{color};margin:var(--sp-4) 0 var(--sp-2) 0'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )


# ── Inline helpers ─────────────────────────────────────────────────────────────


def colored_value(value: float | str, *, threshold: float = 0) -> str:
    try:
        v = float(value)
        cls = "tx-pos" if v > threshold else "tx-neg"
    except (TypeError, ValueError):
        cls = "tx-muted"
    return f"<span class='{cls}'>{value}</span>"


def regime_badge(regime: str) -> str:
    """Return a pill HTML string for a regime name."""
    _map = {
        "BULL_OPEN":  ("BULL",   "green"),
        "BEAR_OPEN":  ("BEAR",   "red"),
        "CHOPPY":     ("CHOPPY", "amber"),
        "PRE_MARKET": ("PRE",    "grey"),
        "TREND_UP":   ("TREND↑", "green"),
        "TREND_DOWN": ("TREND↓", "red"),
        "HIGH_VOL":   ("HI-VOL", "amber"),
        "NEWS_RISK":  ("NEWS",   "red"),
        "UNKNOWN":    ("UNKN",   "grey"),
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


# ── Trading workflow: regime and eligibility chips ────────────────────────────


def regime_chip(regime: str) -> str:
    """Return a `.tx-regime` chip for a market regime string."""
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


# ── Unified state vocabulary ──────────────────────────────────────────────────
#
# States used across Scanner, DayTrading, Risk, Strategy pages.
# Always use eligibility_chip(state) to render — do NOT hardcode colors or text
# per page. Update the mapping here and all pages benefit automatically.
#
#   READY    — conditions met, can act
#   WATCH    — marginal; monitor but cautious
#   ACTIVE   — position open (any direction)
#   TRAILING — trailing-stop management phase
#   PARTIAL  — partial exit taken, remainder managed
#   BLOCKED  — hard gate: kill switch / risk cap / regime / spread
#   IDLE     — no position, no signal, standby
#   COOLDOWN — post-trade lockout before next entry

_ELIG_MAP: dict[str, tuple[str, str]] = {
    "ready":    ("● READY",    "ready"),
    "watch":    ("◐ WATCH",    "watch"),
    "active":   ("▶ ACTIVE",   "ready"),    # same green family as READY
    "trailing": ("↝ TRAILING", "watch"),    # amber — managing risk
    "partial":  ("◑ PARTIAL",  "watch"),    # amber — partial out
    "blocked":  ("✕ BLOCKED",  "blocked"),
    "idle":     ("○ IDLE",     "idle"),
    "cooldown": ("⏳ COOLDOWN", "idle"),
}


def eligibility_chip(state: str, reason: str = "") -> str:
    """Return an HTML `.tx-chip` for a trading eligibility/state.

    Standard states: ready | watch | active | trailing | partial | blocked | idle | cooldown

    Args:
        state:  One of the vocabulary states above (case-insensitive).
        reason: Optional tooltip text explaining the state (shown on hover).
    """
    text, cls = _ELIG_MAP.get(state.lower(), (state.upper(), "idle"))
    title = f' title="{reason}"' if reason else ""
    return f"<span class='tx-chip {cls}'{title}>{text}</span>"


# ── Bot state → eligibility mapping ──────────────────────────────────────────
#
# Maps raw backend SingleStockTrader state strings to the platform vocabulary.
# Import and call bot_state_chip(state_val, block_reason) in any page that
# displays auto-trader status so all pages stay consistent.

_BOT_STATE_TO_ELIG: dict[str, tuple[str, str]] = {
    "FLAT":               ("idle",     "Flat — scanning for entry"),
    "LONG":               ("active",   "Long position open"),
    "SHORT":              ("active",   "Short position open"),
    "PARTIAL_EXIT_TAKEN": ("partial",  "Partial exit taken — managing remainder"),
    "TRAILING":           ("trailing", "Trailing stop active"),
    "EXITED":             ("cooldown", "Trade closed — cooldown before next entry"),
    "PENDING_ENTRY":      ("watch",    "Order pending fill"),
    "BLOCKED":            ("blocked",  ""),   # reason filled at call-site
    "UNKNOWN":            ("idle",     "State unknown"),
}


def bot_state_chip(raw_state: str, block_reason: str = "") -> str:
    """Map a raw bot state string to a standard eligibility chip.

    Usage::
        st.markdown(bot_state_chip(status["state"], status.get("block_reason","")),
                    unsafe_allow_html=True)
    """
    elig, default_reason = _BOT_STATE_TO_ELIG.get(raw_state, ("idle", raw_state))
    reason = block_reason if raw_state == "BLOCKED" and block_reason else default_reason
    return eligibility_chip(elig, reason)


# ── Blocker reason vocabulary ─────────────────────────────────────────────────
#
# Canonical human-readable labels for every blocker reason that appears across
# pages. Internal reason keys (from the brain, risk engine, or scheduler) map
# to a short label + optional detail. Pages call blocker_label(key) instead of
# formatting reasons ad-hoc.

_BLOCKER_LABELS: dict[str, tuple[str, str]] = {
    # Kill switch and risk gates
    "KILL_SWITCH":        ("Kill switch",         "All order paths blocked by the kill switch"),
    "KILL_SWITCH_ACTIVE": ("Kill switch",         "All order paths blocked by the kill switch"),
    "MARKET_CLOSED":      ("Market closed",        "Outside trading hours for this symbol"),
    "POSITION_LIMIT":     ("Position limit",       "Max open positions reached"),
    "DAILY_LOSS_LIMIT":   ("Daily loss cap",       "Daily loss limit reached — trading halted"),
    "ORDER_LIMIT":        ("Order limit",           "Max orders per day reached"),
    "CONSEC_LOSSES":      ("Consec. losses",       "Too many consecutive losses — size reduced"),
    # Regime gates
    "NEWS_RISK":          ("News risk",             "NEWS_RISK regime — new entries blocked"),
    "CHOPPY_REGIME":      ("Choppy regime",         "CHOPPY market — reduced entry criteria"),
    "HIGH_VOL":           ("High volatility",       "HIGH_VOL regime — tighter management"),
    "BEAR_REGIME":        ("Bear regime",           "BEAR_OPEN — only short-biased strategies"),
    # Signal quality gates
    "NO_SETUP":           ("No setup",              "No qualifying entry conditions found"),
    "LOW_CONFIDENCE":     ("Low confidence",        "Signal confidence below minimum threshold"),
    "WEAK_RVOL":          ("Low rel-vol",           "Pre-market relative volume too low"),
    "WIDE_SPREAD":        ("Wide spread",           "Bid-ask spread too wide — slippage risk"),
    "BAD_RR":             ("Poor R:R",              "Risk:reward below minimum"),
    "STALE_SIGNAL":       ("Stale signal",          "Signal too old — entry window closed"),
    # Execution gates
    "NOT_CONFIGURED":     ("Not configured",        "Broker credentials missing"),
    "POLICY_BLOCKED":     ("Policy blocked",        "Symbol policy prevents live trading"),
    "CORRELATION_LIMIT":  ("Correlation cap",       "Too many correlated positions open"),
    # Size reduction (not hard block)
    "SIZE_REDUCED":       ("Reduced size",          "Risk governor throttling position size"),
}


def blocker_label(reason_key: str, *, detail: bool = False) -> str:
    """Return a human-readable label for a blocker reason key.

    Args:
        reason_key: Internal reason string (case-insensitive, underscores or spaces).
        detail:     If True, return the longer detail string instead of the short label.

    Returns plain text (not HTML). Wrap in eligibility_chip or pill as needed.
    """
    key = reason_key.upper().replace(" ", "_").replace("-", "_")
    label_short, label_detail = _BLOCKER_LABELS.get(key, (reason_key.replace("_", " ").title(), ""))
    return label_detail if detail and label_detail else label_short


def blocker_chip(reason_key: str) -> str:
    """Return a blocked eligibility_chip labelled with the standardised reason.

    Usage::
        st.markdown(blocker_chip("KILL_SWITCH"), unsafe_allow_html=True)
        # renders: ✕ BLOCKED [title="All order paths blocked by the kill switch"]
    """
    short  = blocker_label(reason_key)
    detail = blocker_label(reason_key, detail=True)
    # Show the short reason inline and full detail on hover
    text, cls = "✕ " + short.upper(), "blocked"
    title_attr = f' title="{detail}"' if detail else ""
    return f"<span class='tx-chip {cls}'{title_attr}>{text}</span>"


# ── Scanner candidate eligibility ─────────────────────────────────────────────


def candidate_state(
    score: int,
    direction: str,
    auto_traded: bool,
    is_recommended_match: bool,
) -> tuple[str, str]:
    """Derive eligibility state and reason for a scanner candidate row.

    Returns ``(state, reason)`` for use with ``eligibility_chip()``.

    Priority:
      ACTIVE   — symbol is currently auto-traded
      READY    — score ≥ 60 and matches the historically recommended strategy
      READY    — score ≥ 60 with strong agreement
      WATCH    — score 35–59 or weak agreement
      IDLE     — score < 35 (surfaced but weak)
    """
    if auto_traded:
        return "active", "Symbol is currently auto-traded"
    if score >= 60 and is_recommended_match:
        return "ready", f"Score {score}/100 · matches recommended strategy"
    if score >= 60:
        return "ready", f"Score {score}/100 · strong signal agreement"
    if score >= 35:
        return "watch", f"Score {score}/100 · marginal signal strength"
    return "idle", f"Score {score}/100 · weak signal — monitor only"


# ── Risk gauge ────────────────────────────────────────────────────────────────


def risk_gauge_html(label: str, used: float, maximum: float, *, prefix: str = "$") -> str:
    """Return HTML for a labelled risk gauge row (use above st.progress())."""
    pct = used / max(maximum, 1)
    cls = "danger" if pct >= 0.85 else ("warn" if pct >= 0.60 else "ok")
    val_str = f"{prefix}{abs(used):,.0f} / {prefix}{abs(maximum):,.0f}"
    return (
        f"<div class='tx-gauge-row'>"
        f"<span>{label}</span>"
        f"<span class='tx-gauge-val {cls}'>{val_str}</span>"
        f"</div>"
    )
