"""
PDTTracker — Pattern Day Trader (PDT) rule awareness for live margin accounts.

Background
----------
FINRA's PDT rule applies to **margin** accounts that execute **4 or more day
trades within 5 rolling business days**. A "day trade" is buying and selling (or
selling short and buying to cover) the *same security* on the *same trading day*.
An account flagged as a Pattern Day Trader must maintain ≥ $25,000 equity; below
that, the broker restricts further day-trading (and a violation can freeze the
account for 90 days).

The rule does NOT cap the *number* of trades for accounts that are exempt:
  - **Cash accounts** are not subject to PDT (settlement/good-faith rules apply
    instead — out of scope here).
  - Accounts with **≥ $25,000** equity are not restricted.
  - **Paper / simulated** accounts have no regulatory exposure.

So this tracker is a *targeted* guard: it is a no-op unless ALL of the following
hold — account_type == "margin", equity < $25,000, and the broker is real
(not paper). When active, it counts day trades over the rolling window and blocks
a *new* day-trade entry once 3 have been used (the 4th would trigger the flag).

Design
------
Stateless, rebuilt each call from the completed round-trip log — the same pattern
``RiskGovernor`` uses. Each round-trip is a dict with at least ``symbol``,
``buy_at`` and ``sell_at`` (datetimes or ISO strings). A round-trip counts as a
day trade when its buy and sell land on the same trading day. This intentionally
mirrors how brokers count: it is the *closing* leg that "uses" a day trade, so we
bucket day trades by the sell date.

The window is the last 5 **business days** (Mon–Fri) including today — a simple,
slightly conservative approximation of "5 rolling business days" that does not
need a market-holiday calendar. Being conservative here is the safe direction for
a compliance guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

# Regulatory constants
PDT_EQUITY_FLOOR: float = 25_000.0   # accounts ≥ this are exempt
PDT_MAX_DAY_TRADES: int = 3          # a 4th in the window triggers the flag
PDT_WINDOW_BUSINESS_DAYS: int = 5    # rolling 5 business days


@dataclass
class PDTStatus:
    """Snapshot of PDT standing for one account."""
    guard_active: bool                 # False = exempt (cash / ≥25k / paper)
    exempt_reason: str                 # why the guard is off (empty when active)
    day_trades_used: int               # day trades inside the rolling window
    day_trades_remaining: int          # max(0, PDT_MAX_DAY_TRADES - used)
    window_start: date | None          # first day of the rolling window
    equity: float
    account_type: str                  # "cash" | "margin"
    is_paper: bool
    # Per-day breakdown for the UI ({iso_date: count})
    per_day: dict[str, int] = field(default_factory=dict)

    @property
    def at_limit(self) -> bool:
        """True when the guard is active AND no day trades remain."""
        return self.guard_active and self.day_trades_remaining <= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard_active": self.guard_active,
            "exempt_reason": self.exempt_reason,
            "day_trades_used": self.day_trades_used,
            "day_trades_remaining": self.day_trades_remaining,
            "max_day_trades": PDT_MAX_DAY_TRADES,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_business_days": PDT_WINDOW_BUSINESS_DAYS,
            "equity": round(self.equity, 2),
            "equity_floor": PDT_EQUITY_FLOOR,
            "equity_to_floor": round(max(0.0, PDT_EQUITY_FLOOR - self.equity), 2),
            "account_type": self.account_type,
            "is_paper": self.is_paper,
            "at_limit": self.at_limit,
            "per_day": self.per_day,
        }


@dataclass
class PDTDecision:
    """Result of asking whether a new day-trade entry is allowed."""
    allowed: bool
    reason: str
    status: PDTStatus


class PDTTracker:
    """Stateless PDT evaluator. Build a status from the round-trip log, then ask
    ``check_can_open_day_trade`` before arming a new intraday entry.

    Parameters
    ----------
    account_type : "cash" | "margin". Cash accounts are never restricted here.
    equity : current account equity (used for the $25k exemption).
    is_paper : True for paper / simulated brokers (never restricted).
    """

    def __init__(
        self,
        account_type: str = "cash",
        equity: float = 0.0,
        is_paper: bool = True,
    ) -> None:
        self.account_type = (account_type or "cash").lower()
        self.equity = float(equity)
        self.is_paper = bool(is_paper)

    # ── Exemption logic ────────────────────────────────────────────────────────

    def _exempt_reason(self) -> str:
        """Return why the guard is OFF, or "" when it must be enforced."""
        if self.is_paper:
            return "paper/simulated account — PDT does not apply"
        if self.account_type != "margin":
            return f"{self.account_type} account — PDT applies to margin accounts only"
        if self.equity >= PDT_EQUITY_FLOOR:
            return f"equity ${self.equity:,.0f} >= ${PDT_EQUITY_FLOOR:,.0f} - exempt"
        return ""

    # ── Build status ───────────────────────────────────────────────────────────

    def build_status(
        self,
        round_trips: Iterable[dict[str, Any]],
        as_of: date | None = None,
    ) -> PDTStatus:
        """Count day trades in the rolling 5-business-day window ending today.

        round_trips: dicts with ``symbol``, ``buy_at`` and ``sell_at`` (datetime
        or ISO string). Only same-trading-day round-trips count as day trades;
        they are bucketed by the *sell* date (the closing leg).
        """
        today = as_of or date.today()
        window_days = _last_n_business_days(today, PDT_WINDOW_BUSINESS_DAYS)
        window_set = set(window_days)
        window_start = window_days[0] if window_days else None

        per_day: dict[str, int] = {}
        for rt in round_trips:
            buy_d = _coerce_date(rt.get("buy_at"))
            sell_d = _coerce_date(rt.get("sell_at"))
            if buy_d is None or sell_d is None:
                continue
            # A day trade closes on the same trading day it opened.
            if buy_d != sell_d:
                continue
            if sell_d not in window_set:
                continue
            key = sell_d.isoformat()
            per_day[key] = per_day.get(key, 0) + 1

        used = sum(per_day.values())
        exempt = self._exempt_reason()
        guard_active = exempt == ""

        return PDTStatus(
            guard_active=guard_active,
            exempt_reason=exempt,
            day_trades_used=used,
            day_trades_remaining=max(0, PDT_MAX_DAY_TRADES - used),
            window_start=window_start,
            equity=self.equity,
            account_type=self.account_type,
            is_paper=self.is_paper,
            per_day=per_day,
        )

    # ── Decision ───────────────────────────────────────────────────────────────

    def check_can_open_day_trade(
        self,
        round_trips: Iterable[dict[str, Any]],
        *,
        symbol: str | None = None,
        as_of: date | None = None,
    ) -> PDTDecision:
        """Decide whether opening a *new* position that could become a day trade
        is allowed under PDT. Exempt accounts always pass.

        We treat any new intraday entry as a *potential* day trade (it usually is
        for this autotrader, which flattens by EOD), so the guard blocks once the
        rolling count has used all 3 day trades. This is the conservative reading
        that keeps a sub-$25k margin account from tripping the flag.
        """
        status = self.build_status(round_trips, as_of=as_of)

        if not status.guard_active:
            return PDTDecision(allowed=True, reason=status.exempt_reason, status=status)

        if status.day_trades_remaining > 0:
            return PDTDecision(
                allowed=True,
                reason=(
                    f"PDT ok: {status.day_trades_used}/{PDT_MAX_DAY_TRADES} day "
                    f"trades used in the rolling {PDT_WINDOW_BUSINESS_DAYS}-day window."
                ),
                status=status,
            )

        sym = f" for {symbol}" if symbol else ""
        return PDTDecision(
            allowed=False,
            reason=(
                f"PDT BLOCK{sym}: {status.day_trades_used}/{PDT_MAX_DAY_TRADES} day "
                f"trades already used in the rolling {PDT_WINDOW_BUSINESS_DAYS}-day "
                f"window on a sub-${PDT_EQUITY_FLOOR:,.0f} margin account. A 4th day "
                f"trade would flag the account as a Pattern Day Trader. "
                f"Add ${max(0.0, PDT_EQUITY_FLOOR - status.equity):,.0f} equity to lift this."
            ),
            status=status,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _last_n_business_days(end: date, n: int) -> list[date]:
    """Return the last ``n`` business days (Mon–Fri) ending on/at ``end``.

    Walks backward from ``end`` (inclusive) skipping Sat/Sun. Does not account
    for market holidays — intentionally conservative for a compliance guard.
    """
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:   # 0=Mon .. 4=Fri
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def _coerce_date(value: Any) -> date | None:
    """Coerce a datetime / date / ISO string into a date, or None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
    return None
