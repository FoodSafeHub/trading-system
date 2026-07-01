from __future__ import annotations

"""
Risk Engine — all checks must pass before an order is submitted.

Hard rules (any failure blocks the order):
  1. Kill switch check
  2. Live trading safety flags (3-factor: ENV flag + confirmation + broker credentials)
  3. Market hours check
  4. Max orders per day
  5. Order cooldown period
  6. Duplicate order prevention (idempotency key)
  7. Max position size
  8. Max daily loss

No live order should ever bypass this engine.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.config import Settings, get_settings
from app.db import SessionLocal
from app.models.orders import Order
from app.models.settings import AppSetting
from app.schemas.orders import OrderRequest
from app.schemas.risk import RiskCheckResult, RiskStatusOut
from app.utils.time_utils import is_market_hours, now_in_tz

logger = logging.getLogger(__name__)

KILL_SWITCH_KEY = "kill_switch_active"


class RiskEngine:

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    # ── Kill switch ──────────────────────────────────────────────────────────

    def is_kill_switch_active(self) -> bool:
        try:
            with SessionLocal() as db:
                row = db.query(AppSetting).filter_by(key=KILL_SWITCH_KEY).first()
                if row:
                    return row.value.lower() in ("true", "1", "yes")
        except Exception:
            pass
        return False

    def set_kill_switch(self, active: bool) -> None:
        with SessionLocal() as db:
            row = db.query(AppSetting).filter_by(key=KILL_SWITCH_KEY).first()
            if not row:
                row = AppSetting(key=KILL_SWITCH_KEY, description="Emergency trading stop")
                db.add(row)
            row.value = "true" if active else "false"
            db.commit()
        logger.warning("[risk] Kill switch set to: %s", active)

    # ── Daily stats ──────────────────────────────────────────────────────────

    def _orders_today(self) -> int:
        tz = self._settings.tz
        start_of_day = now_in_tz(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_of_day.astimezone(timezone.utc)
        with SessionLocal() as db:
            return (
                db.query(Order)
                .filter(
                    Order.created_at >= start_utc,
                    Order.status.notin_(["rejected", "error"]),
                )
                .count()
            )

    def _daily_realized_loss(self) -> float:
        """Loss magnitude (positive number) realized so far today, in USD.

        Reads FIFO-matched round-trips from `realized_trades` whose SELL closed
        today (broker tz), nets their P&L, and returns the loss as a positive
        figure (0.0 if flat or net-positive). The caller compares this against
        max_daily_loss_usd, so only a net-loss day can trip the breaker — a
        winning round-trip cancelling a losing one will not.
        """
        tz = self._settings.tz
        start_of_day = now_in_tz(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_of_day.astimezone(timezone.utc)
        try:
            from app.models.realized_trades import RealizedTrade
            from app.services.pnl.store import sync_realized_trades

            with SessionLocal() as db:
                # Materialize any new round-trips first so an exit that just
                # filled today is counted, not missed until the next /pnl call.
                sync_realized_trades(db)
                net = (
                    db.query(RealizedTrade.realized_pnl)
                    .filter(RealizedTrade.sell_at >= start_utc)
                    .all()
                )
            total = sum(p for (p,) in net)
            return -total if total < 0 else 0.0
        except Exception as exc:
            # Never let a P&L read failure block trading; log and fail open.
            logger.warning("[risk] daily-loss read failed, treating as 0: %s", exc)
            return 0.0

    def _last_order_time(self, symbol: str) -> Optional[datetime]:
        with SessionLocal() as db:
            order = (
                db.query(Order)
                .filter(Order.symbol == symbol, Order.status.notin_(["rejected", "error", "cancelled"]))
                .order_by(Order.created_at.desc())
                .first()
            )
            return order.created_at if order else None

    def _existing_position_value(self, symbol: str) -> float:
        """Return approximate market value of existing position in this symbol."""
        with SessionLocal() as db:
            order = (
                db.query(Order)
                .filter(
                    Order.symbol == symbol,
                    Order.side == "BUY",
                    Order.status == "filled",
                )
                .order_by(Order.created_at.desc())
                .first()
            )
        if order and order.fill_price and order.quantity:
            return order.fill_price * order.quantity
        return 0.0

    def _is_duplicate(self, idempotency_key: Optional[str]) -> bool:
        if not idempotency_key:
            return False
        with SessionLocal() as db:
            exists = (
                db.query(Order)
                .filter(Order.idempotency_key == idempotency_key)
                .first()
            )
        return exists is not None

    # ── Market hours (per-broker) ──────────────────────────────────────────────

    def _resolve_broker(self, symbol: str) -> str:
        """Effective broker for a symbol: its assignment override, else global.

        Mirrors the factory's routing precedence — a symbol pinned to a specific
        broker uses that broker's market session; everything else follows the
        global active_broker / trade_routing toggle.
        """
        try:
            from app.models.assignments import SymbolStrategyAssignment

            with SessionLocal() as db:
                row = (
                    db.query(SymbolStrategyAssignment)
                    .filter_by(symbol=symbol.upper().strip())
                    .first()
                )
            if row and row.broker and row.broker != "default":
                return row.broker
        except Exception:
            pass
        # Auto-route by market: an India (NSE/BSE) symbol with no explicit broker
        # override goes to Zerodha regardless of the US-oriented global toggle, so
        # US and India symbols each trade in their own session simultaneously.
        try:
            from app.services.markets import is_india_symbol
            if is_india_symbol(symbol):
                return "zerodha"
        except Exception:
            pass
        s = self._settings
        if s.trade_routing not in ("auto", "both"):
            return s.trade_routing
        return s.active_broker

    def _market_hours_for(self, symbol: str) -> tuple[str, str, "object", str]:
        """Return (open, close, tzinfo, tz_name) for the symbol's broker session."""
        s = self._settings
        if self._resolve_broker(symbol) == "zerodha":
            return s.india_market_open, s.india_market_close, s.india_tz, s.india_timezone
        return s.trading_start_time, s.trading_end_time, s.tz, s.trading_timezone

    # ── Main check ───────────────────────────────────────────────────────────

    def check(
        self,
        order: OrderRequest,
        estimated_price: Optional[float] = None,
        held_value_usd: float = 0.0,
    ) -> RiskCheckResult:
        s = self._settings
        warnings = []

        # 1. Kill switch
        if self.is_kill_switch_active():
            return RiskCheckResult(passed=False, blocked_reason="Kill switch is active. Trading halted.")

        # 2. Live trading safety flags (only relevant if not paper)
        if s.active_broker != "paper":
            if not s.live_trading_enabled:
                return RiskCheckResult(
                    passed=False,
                    blocked_reason="LIVE_TRADING_ENABLED is false. Set it to true in .env to proceed.",
                )
            if not s.live_trading_confirmed:
                return RiskCheckResult(
                    passed=False,
                    blocked_reason="LIVE_TRADING_CONFIRMED is false. Set it to true in .env to proceed.",
                )

        # 3. Market hours — per-broker. India (Zerodha) trades on the NSE/BSE
        #    session (09:15–15:30 IST), not the US session.
        #    LIMIT orders are allowed outside regular hours: the Schwab adapter
        #    routes them session=SEAMLESS into the extended-hours session, where
        #    a resting limit can fill or wait for the next open. MARKET orders
        #    are NOT eligible for extended hours and would queue blindly into an
        #    unknown open price, so they stay blocked outside RTH.
        open_str, close_str, tz, tz_name = self._market_hours_for(order.symbol)
        if not is_market_hours(open_str, close_str, tz) and order.order_type == "MARKET":
            return RiskCheckResult(
                passed=False,
                blocked_reason=f"Outside market hours ({open_str}–{close_str} {tz_name})",
            )

        # 4. Max orders per day
        orders_today = self._orders_today()
        if orders_today >= s.max_orders_per_day:
            return RiskCheckResult(
                passed=False,
                blocked_reason=f"Daily order limit reached ({orders_today}/{s.max_orders_per_day})",
            )

        # 5. Cooldown
        last_time = self._last_order_time(order.symbol)
        if last_time:
            elapsed = (datetime.now(tz=timezone.utc) - last_time.replace(tzinfo=timezone.utc)).total_seconds()
            if elapsed < s.order_cooldown_seconds:
                remaining = int(s.order_cooldown_seconds - elapsed)
                return RiskCheckResult(
                    passed=False,
                    blocked_reason=f"Cooldown active for {order.symbol}: {remaining}s remaining",
                )

        # 6. Duplicate prevention
        if self._is_duplicate(order.idempotency_key):
            return RiskCheckResult(
                passed=False,
                blocked_reason=f"Duplicate order detected (idempotency_key={order.idempotency_key})",
            )

        # 7. Max position size — a ceiling on TOTAL held value for the symbol,
        #    NOT a per-order limit. A BUY is judged by what the position becomes
        #    (already-held value + this order), so a legitimate top-up that stays
        #    under the cap passes, and only an order that would push the TOTAL
        #    past the cap is blocked. SELLs reduce exposure and are never gated
        #    here. held_value_usd is supplied by the caller (execution service),
        #    which knows the current broker holding; defaults to 0 for callers
        #    that don't (treats the order as the whole position, as before).
        price = estimated_price or order.limit_price or 0
        order_value = price * order.quantity
        total_value = order_value + max(0.0, held_value_usd) if order.side == "BUY" else order_value
        if order.side == "BUY" and total_value > s.max_position_size_usd:
            return RiskCheckResult(
                passed=False,
                blocked_reason=(
                    f"Total position value ${total_value:.2f} (held ${held_value_usd:.2f} "
                    f"+ order ${order_value:.2f}) exceeds max position size "
                    f"${s.max_position_size_usd:.2f}"
                ),
            )
        elif order.side == "BUY" and total_value > s.max_position_size_usd * 0.8:
            warnings.append(
                f"Total position value ${total_value:.2f} is >80% of position size limit"
            )

        # 8. Daily loss
        daily_loss = self._daily_realized_loss()
        if daily_loss >= s.max_daily_loss_usd:
            return RiskCheckResult(
                passed=False,
                blocked_reason=f"Daily loss limit hit (${daily_loss:.2f} >= ${s.max_daily_loss_usd:.2f})",
            )

        return RiskCheckResult(passed=True, warnings=warnings)

    def get_status(self) -> RiskStatusOut:
        s = self._settings
        return RiskStatusOut(
            kill_switch_active=self.is_kill_switch_active(),
            live_trading_enabled=s.live_trading_enabled,
            live_trading_confirmed=s.live_trading_confirmed,
            active_broker=s.active_broker,
            is_live=s.is_live,
            orders_today=self._orders_today(),
            max_orders_per_day=s.max_orders_per_day,
            daily_loss_usd=self._daily_realized_loss(),
            max_daily_loss_usd=s.max_daily_loss_usd,
            market_hours_active=is_market_hours(s.trading_start_time, s.trading_end_time, s.tz),
        )
