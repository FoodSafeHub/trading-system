from __future__ import annotations

"""
Execution Service — orchestrates the full order lifecycle:
  Signal → Risk Check → Preview → Submit → Track → Persist

Every step is logged and persisted to SQLite.
No live order is submitted without passing ALL risk checks.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings
from app.db import SessionLocal
from app.models.executions import Execution
from app.models.orders import Order, OrderPreview
from app.schemas.orders import OrderRequest, OrderStatusResponse
from app.services.audit.service import AuditService
from app.services.brokers.base import BrokerBase
from app.services.risk.engine import RiskEngine

logger = logging.getLogger(__name__)
_audit = AuditService()
_risk = RiskEngine()


class ExecutionService:

    def __init__(self, broker: BrokerBase) -> None:
        self.broker = broker

    async def execute(
        self,
        order_req: OrderRequest,
        account_id: str,
        signal_id: Optional[int] = None,
        estimated_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Full execution pipeline. Returns the persisted Order on success, None if blocked.
        """
        # ── Idempotency key ─────────────────────────────────────────────────
        if not order_req.idempotency_key:
            order_req.idempotency_key = str(uuid.uuid4())
        if signal_id:
            order_req.signal_id = signal_id

        logger.info(
            f"[exec] Starting execution: {order_req.side} {order_req.symbol} "
            f"x{order_req.quantity:.2f} (key={order_req.idempotency_key})"
        )

        # ── Step 1: Risk checks ─────────────────────────────────────────────
        risk_result = _risk.check(order_req, estimated_price=estimated_price)
        if not risk_result.passed:
            logger.warning("[exec] Risk BLOCKED: %s", risk_result.blocked_reason)
            _audit.log(
                event_type="RISK_BLOCKED",
                description=f"{order_req.side} {order_req.symbol} blocked: {risk_result.blocked_reason}",
                metadata={"order": order_req.model_dump(mode="json")},
            )
            return None

        if risk_result.warnings:
            for w in risk_result.warnings:
                logger.warning("[exec] Risk warning: %s", w)

        # ── Step 1b: Buying-power preflight ─────────────────────────────────
        # Hits the broker once per BUY to confirm we actually have capital,
        # because the static max_position_size_usd in risk/engine.py doesn't
        # know what's currently committed at the broker. Disabled via config
        # or when the broker can't answer (we err toward letting the order
        # through and let the broker reject it — the dashboard surfaces that).
        bp_block = await self._buying_power_preflight(order_req, account_id, estimated_price)
        if bp_block is not None:
            logger.warning("[exec] Buying-power BLOCKED: %s", bp_block)
            _audit.log(
                event_type="BUYING_POWER_BLOCKED",
                description=f"{order_req.side} {order_req.symbol} blocked: {bp_block}",
                metadata={"order": order_req.model_dump(mode="json")},
            )
            return None

        # ── Step 2: Persist order record (pending) ──────────────────────────
        db_order = self._persist_order(order_req, signal_id, status="pending")

        # ── Step 3: Preview (if broker supports it) ─────────────────────────
        try:
            preview = await self.broker.preview_order(order_req, account_id)
            self._persist_preview(db_order.id, preview)
            self._update_order_status(db_order.id, "previewed", preview_json=json.dumps(preview.model_dump()))
            logger.info(
                f"[exec] Preview: cost={preview.estimated_cost or 0:.2f} "
                f"commission={preview.estimated_commission or 0:.2f}"
            )
        except NotImplementedError:
            logger.debug("[exec] Broker does not support preview — skipping")
        except Exception as exc:
            logger.warning("[exec] Preview failed (non-fatal): %s", exc)

        # ── Step 4: Submit order ─────────────────────────────────────────────
        try:
            status_resp = await self.broker.place_order(order_req, account_id)
            self._update_order_status(
                db_order.id,
                status="submitted",
                broker_order_id=status_resp.broker_order_id,
                submitted_at=datetime.now(tz=timezone.utc),
            )
            _audit.log(
                event_type="ORDER_SUBMITTED",
                entity_type="order",
                entity_id=db_order.id,
                description=f"{order_req.side} {order_req.symbol} x{order_req.quantity} → broker_id={status_resp.broker_order_id}",
            )
            logger.info(
                "[exec] Order submitted: broker_id=%s status=%s",
                status_resp.broker_order_id, status_resp.status,
            )
        except Exception as exc:
            logger.error("[exec] Order submission failed: %s", exc)
            self._update_order_status(db_order.id, status="error", error_message=str(exc))
            self._persist_execution_event(db_order.id, "error", error=str(exc))
            _audit.log(
                event_type="ORDER_ERROR",
                entity_type="order",
                entity_id=db_order.id,
                description=str(exc),
            )
            return db_order

        # ── Step 5: Confirm actual broker state ─────────────────────────────
        # Schwab returns 201 from place_order before deciding to accept/reject the
        # order. Without a follow-up GET, our DB reports "submitted" for orders the
        # broker immediately rejected. Poll once to capture the real status.
        confirmed = await self._confirm_broker_status(
            status_resp.broker_order_id, account_id, db_order.id
        )
        final_status = confirmed.status if confirmed else status_resp.status

        # ── Step 6: Handle immediate fill ────────────────────────────────────
        if final_status in ("filled", "partial"):
            self._handle_fill(db_order.id, confirmed or status_resp)
            # ── Step 6b: Protective stop on a filled BUY ─────────────────────
            # Place a resting SELL STOP at the broker so the position survives a
            # software outage. Default OFF; skipped for paper (the paper broker
            # fills a STOP at market immediately, which would self-close the
            # position we just opened). Best-effort — a stop failure must not
            # unwind the (already-filled) entry.
            if order_req.side == "BUY":
                try:
                    await self._submit_protective_stop(
                        order_req, account_id, confirmed or status_resp
                    )
                except Exception as exc:
                    logger.error("[exec] Protective stop submission failed (non-fatal): %s", exc)

        return db_order

    def _volatility_trail_pct(self, symbol: str, price: float) -> float:
        """Volatility-aware trail %: ATR% (ATR/price × 100) scaled by a
        multiplier that adjusts for how volatile the symbol is.

        ATR% buckets (based on 22-day ATR as % of price):
            < 1.5%  → low-vol  (KO, XOM, SO)    → 2.0× ATR trail (~2.5–3%)
            1.5–3%  → mid-vol  (NVDA, BAC, COP) → 2.5× ATR trail (~4–7%)
            3–4.5%  → high-vol (NVDA, TSLA)     → 2.0× ATR trail (~7–9%)
            > 4.5%  → very-high (TOST, PLTR)    → 1.5× ATR trail (~7–9%)

        Low-vol names get a tighter trail (less slippage on exit); high-vol
        names get breathing room (normal pullbacks won't shake them out).
        Floored at 1.5% (never choke) and capped at 10% (never give back more
        than a solid swing move). Falls back to settings.trail_stop_pct when
        OHLCV/ATR is unavailable.
        """
        settings = get_settings()
        trail_pct = settings.trail_stop_pct  # fallback
        try:
            from app.services.market_data.provider import get_ohlcv
            from app.services.strategy.rules import _atr_raw
            df = get_ohlcv(symbol, period="3mo")
            if not df.empty and len(df) >= 22:
                atr = float(_atr_raw(df, 22).iloc[-1])
                if price > 0 and atr > 0:
                    atr_pct = (atr / price) * 100
                    if atr_pct < 1.5:
                        mult = 2.0
                    elif atr_pct < 3.0:
                        mult = 2.5
                    elif atr_pct < 4.5:
                        mult = 2.0
                    else:
                        mult = 1.5
                    trail_pct = max(1.5, min(10.0, round(atr_pct * mult, 2)))
                    logger.info(
                        "[exec] Volatility trail %s: ATR=%.2f (%.1f%% of price) "
                        "× %.1f → trail=%.2f%%",
                        symbol, atr, atr_pct, mult, trail_pct,
                    )
        except Exception as exc:
            logger.debug("[exec] ATR trail compute failed, using pct fallback: %s", exc)
        return trail_pct

    async def _submit_protective_stop(
        self,
        buy_req: OrderRequest,
        account_id: str,
        fill: OrderStatusResponse,
    ) -> None:
        """Place a broker-native trailing stop after a BUY fill.

        When settings.trailing_stop_enabled=True (default), places a
        TRAILING_STOP at the broker using chandelier ATR trail distance when
        OHLCV is available, otherwise falling back to trail_stop_pct%.

        When trailing_stop_enabled=False, falls back to the legacy fixed STOP
        at fill_price * (1 - protective_stop_pct/100).

        No-op unless auto_protective_stop_enabled=True.
        Skipped for paper broker (STOP fills at market on submit).
        """
        settings = get_settings()
        if not settings.auto_protective_stop_enabled:
            return

        broker_name = getattr(self.broker, "name", "") or ""
        if "paper" in broker_name and "schwab" not in broker_name \
                and "webull" not in broker_name and "zerodha" not in broker_name:
            logger.debug("[exec] Trailing stop skipped — paper broker (%s)", broker_name)
            return

        filled_qty = fill.filled_quantity or buy_req.quantity
        if not filled_qty or filled_qty <= 0:
            return
        fill_price = fill.fill_price or 0.0

        # Brokers without a native trailing-stop order type (e.g. Zerodha/Kite)
        # cannot accept a TRAILING_STOP — placing one would either be rejected
        # or silently become a MARKET sell. For those, fall back to a static
        # STOP (the Chandelier job ratchets it every 15 min during market hrs).
        native_trail_ok = getattr(self.broker, "supports_native_trailing_stop", False)

        if settings.trailing_stop_enabled:
            # Volatility-aware trail %: ATR% (ATR/price × 100) scaled by a
            # multiplier that adjusts for how volatile the symbol is. Computed
            # the same way regardless of broker; how we PLACE it differs below.
            trail_pct = self._volatility_trail_pct(buy_req.symbol, fill_price)

            if native_trail_ok:
                stop_req = OrderRequest(
                    symbol=buy_req.symbol,
                    side="SELL",
                    order_type="TRAILING_STOP",
                    quantity=filled_qty,
                    trail_type="PERCENT",
                    trail_value=round(trail_pct, 2),
                    time_in_force="GTC",
                    source=buy_req.source,
                    idempotency_key=f"trailstop-{buy_req.idempotency_key}",
                )
                event_desc = (
                    f"SELL TRAILING_STOP {buy_req.symbol} x{filled_qty} "
                    f"trail={trail_pct:.1f}% (chandelier ATR, protects BUY {buy_req.idempotency_key})"
                )
                log_msg = "[exec] Trailing stop placed: %s x%.4f trail=%.1f%% broker_id=%s"
                log_args = (buy_req.symbol, filled_qty, trail_pct)
            else:
                # No native trailing (Zerodha): place a static STOP at the same
                # ATR-derived distance below fill. The Chandelier job ratchets
                # it up on later highs, approximating a trailing stop.
                if fill_price <= 0:
                    logger.warning(
                        "[exec] Protective stop skipped — no fill price for %s "
                        "(broker has no native trailing stop)", buy_req.symbol,
                    )
                    return
                stop_price = round(fill_price * (1 - trail_pct / 100.0), 2)
                stop_req = OrderRequest(
                    symbol=buy_req.symbol,
                    side="SELL",
                    order_type="STOP",
                    quantity=filled_qty,
                    stop_price=stop_price,
                    time_in_force="GTC",
                    source=buy_req.source,
                    idempotency_key=f"trailstop-{buy_req.idempotency_key}",
                )
                event_desc = (
                    f"SELL STOP {buy_req.symbol} x{filled_qty} @ {stop_price} "
                    f"(static ATR trail={trail_pct:.1f}%, no native trailing on "
                    f"{getattr(self.broker, 'name', '?')}; Chandelier ratchets — "
                    f"protects BUY {buy_req.idempotency_key})"
                )
                log_msg = "[exec] Static ATR stop placed: SELL STOP %s x%.4f @ %.2f broker_id=%s"
                log_args = (buy_req.symbol, filled_qty, stop_price)  # type: ignore[assignment]
        else:
            # Legacy fixed STOP fallback
            fill_price_safe = fill_price or 0.0
            stop_price = buy_req.stop_price
            if not stop_price or stop_price <= 0:
                if fill_price_safe <= 0:
                    logger.warning("[exec] Protective stop skipped — no stop_price and no fill_price")
                    return
                stop_price = round(fill_price_safe * (1 - settings.protective_stop_pct / 100.0), 2)
            if fill_price_safe > 0 and stop_price >= fill_price_safe:
                logger.warning(
                    "[exec] Protective stop %.2f not below fill %.2f — skipping",
                    stop_price, fill_price_safe,
                )
                return
            stop_req = OrderRequest(
                symbol=buy_req.symbol,
                side="SELL",
                order_type="STOP",
                quantity=filled_qty,
                stop_price=stop_price,
                time_in_force="GTC",
                source=buy_req.source,
                idempotency_key=f"protstop-{buy_req.idempotency_key}",
            )
            event_desc = (
                f"SELL STOP {buy_req.symbol} x{filled_qty} @ {stop_price} "
                f"(protects BUY {buy_req.idempotency_key})"
            )
            log_msg = "[exec] Protective stop placed: SELL STOP %s x%.4f @ %.2f broker_id=%s"
            log_args = (buy_req.symbol, filled_qty, stop_price)  # type: ignore[assignment]

        status_resp = await self.broker.place_order(stop_req, account_id)
        stop_order = self._persist_order(stop_req, None, status="submitted")
        self._update_order_status(
            stop_order.id,
            status="submitted",
            broker_order_id=status_resp.broker_order_id,
            submitted_at=datetime.now(tz=timezone.utc),
        )
        _audit.log(
            event_type="TRAILING_STOP_PLACED" if settings.trailing_stop_enabled else "PROTECTIVE_STOP_PLACED",
            entity_type="order",
            entity_id=stop_order.id,
            description=event_desc,
        )
        logger.info(log_msg, *log_args, status_resp.broker_order_id)

    async def tighten_trail_on_sell(
        self,
        symbol: str,
        quantity: float,
        account_id: str,
        signal_price: float,
        *,
        trail_pct: float = 2.0,
        floor_buffer_pct: float = 0.25,
        source: str = "scheduler",
        idempotency_suffix: str = "",
        signal_id: Optional[int] = None,
        force_replace: bool = False,
    ) -> bool:
        """Cancel any resting STOP/TRAILING_STOP for a symbol, then place a
        tight trailing stop in its place.

        Called ONLY when the ASSIGNED strategy for this symbol fires a SELL
        signal. Other strategies signalling SELL on the same symbol are ignored
        — the caller (scheduler._run_cycle) is already filtered to assigned-
        strategy signals only via the signals_to_act list.

        Idempotent across cycles: the scheduler re-evaluates every ~30s and a
        SELL condition typically persists for many cycles. If a healthy resting
        SELL STOP/TRAILING_STOP already protects this symbol we RETURN EARLY
        (treated as success) WITHOUT cancel-and-replace. Re-placing every cycle
        would reset a broker-native trail's ratchet back to the current price
        (so it could never climb) and churn cancel/replace orders on brokers
        without a native trail (Zerodha), opening a brief unprotected window
        each cycle. The existing trail keeps ratcheting on its own — native
        (Schwab/Webull) or via the Chandelier job (Zerodha). Pass
        force_replace=True to re-arm regardless (e.g. trail params changed).

        trail_pct defaults to 2.0% — wider than 1% to absorb normal intraday
        noise before exiting, while still being tight enough to capture most of
        the post-signal upside. 1% was too tight for mid/high-vol names (KO,
        NVDA) and got shaken out by normal daily wicks.

        floor_buffer_pct: retained for signature compatibility; no longer used.
        Approach C arms a TRAILING stop that rides price up, so there is no
        static floor — the trail itself follows the high and only exits on a
        genuine trail_pct% pullback.

        signal_id: the Signal row id for this SELL signal. Linked to the trail
        order so PnL audit can show signal_price vs actual exit price.

        Returns True if the protective order was placed OR a healthy trail was
        already resting; False on failure (no fallback market sell — see below).
        """
        # Step 0: idempotency guard. Fetch working orders ONCE and reuse the
        # list for both the "already protected?" check and the cancel sweep.
        try:
            open_orders = await self.broker.list_orders(account_id, status="working")
        except Exception as exc:
            logger.warning(
                "[exec] tighten_trail %s: could not list working orders (%s) "
                "— proceeding to place a fresh trail", symbol, exc,
            )
            open_orders = []

        resting_sell_stops = [
            o for o in open_orders
            if getattr(o, "symbol", "").upper() == symbol.upper()
            and getattr(o, "side", "").upper() == "SELL"
            and getattr(o, "order_type", "").upper() in ("STOP", "TRAILING_STOP")
            and getattr(o, "broker_order_id", None)
        ]

        if resting_sell_stops and not force_replace:
            logger.info(
                "[exec] tighten_trail %s: already protected by a resting %s — "
                "skipping re-arm (existing trail keeps ratcheting). "
                "Use force_replace=True to override.",
                symbol, getattr(resting_sell_stops[0], "order_type", "stop"),
            )
            return True

        # Step 1: cancel any resting sell-side stops (we're replacing them).
        # Reached only when force_replace=True or no stop was resting.
        for o in resting_sell_stops:
            try:
                await self.broker.cancel_order(o.broker_order_id, account_id)
                logger.info(
                    "[exec] tighten_trail %s: cancelled resting %s",
                    symbol, getattr(o, "order_type", "stop"),
                )
            except Exception as exc:
                logger.warning(
                    "[exec] tighten_trail %s: could not cancel resting %s (%s) "
                    "— placing tight trail anyway",
                    symbol, getattr(o, "order_type", "stop"), exc,
                )

        # Step 2: place the protective SELL order. APPROACH C — after the assigned
        # strategy's SELL signal, arm a TRAILING stop so the position keeps riding
        # the momentum upward and only exits when price actually pulls back by
        # trail_pct%. The trailing stop maximizes gains: the SELL signal tends to
        # fire BEFORE the momentum peak, so we let the trail follow price up.
        #
        # Native-trailing brokers (Schwab/Webull): place a broker-native
        # TRAILING_STOP. The broker ratchets the trigger up automatically as price
        # makes new highs and fills only on a genuine trail_pct% pullback. This is
        # the primary, preferred path — it never converts to a market order before
        # the pullback and rides the full run-up.
        #
        # Non-native brokers (Zerodha): no TRAILING_STOP order type exists, so we
        # place a static STOP just below the current price and the Chandelier job
        # ratchets it up on later highs (a software-side trailing stop).
        current_price = 0.0
        try:
            quotes = await self.broker.get_quotes([symbol])
            q = quotes.get(symbol.upper()) or quotes.get(symbol)
            if q is not None:
                current_price = float(q.last or q.bid or q.ask or 0.0)
        except Exception as exc:
            logger.warning(
                "[exec] tighten_trail %s: quote fetch failed (%s)", symbol, exc,
            )

        if signal_price <= 0 and current_price > 0:
            signal_price = current_price

        suffix = idempotency_suffix or str(int((signal_price or current_price) * 100))
        native_trail_ok = getattr(self.broker, "supports_native_trailing_stop", False)

        if native_trail_ok:
            # APPROACH C: native broker trailing stop, trail_pct% from the high.
            trail_req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="TRAILING_STOP",
                quantity=quantity,
                trail_type="PERCENT",
                trail_value=round(trail_pct, 2),
                time_in_force="GTC",
                source=source,  # type: ignore[arg-type]
                idempotency_key=f"sell-trail-{symbol}-{suffix}",
            )
            logger.info(
                "[exec] tighten_trail %s: native %.1f%% TRAILING_STOP armed "
                "(signal $%.2f, current $%.2f). Broker ratchets up with new highs; "
                "fills only on a %.1f%% pullback — rides the momentum.",
                symbol, trail_pct, signal_price, current_price, trail_pct,
            )
        else:
            # Zerodha / no native trailing: static STOP ~trail_pct% below current,
            # ratcheted up by the Chandelier job. Stop must rest below market.
            stop_level = round(current_price * (1.0 - trail_pct / 100.0), 2) if current_price > 0 else 0.0
            if stop_level <= 0:
                logger.error(
                    "[exec] tighten_trail %s: no live price — cannot place a "
                    "software trailing STOP. Investigate manually.", symbol,
                )
                return False
            trail_req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="STOP",
                quantity=quantity,
                stop_price=stop_level,
                time_in_force="GTC",
                source=source,  # type: ignore[arg-type]
                idempotency_key=f"sell-trail-{symbol}-{suffix}",
            )
            logger.info(
                "[exec] tighten_trail %s: software trailing STOP @ $%.2f "
                "(current $%.2f − %.1f%%). Chandelier job ratchets it up on new highs.",
                symbol, stop_level, current_price, trail_pct,
            )
        try:
            resp = await self.broker.place_order(trail_req, account_id)
            # Persist with signal_id so PnL audit can join signal_price → exit_price
            stop_order = self._persist_order(trail_req, signal_id, status="submitted")
            self._update_order_status(
                stop_order.id,
                status="submitted",
                broker_order_id=resp.broker_order_id,
                submitted_at=datetime.now(tz=timezone.utc),
            )
            _detail = (
                f"trail={trail_pct}%"
                if trail_req.order_type == "TRAILING_STOP"
                else f"stop=${trail_req.stop_price:.2f}"
            )
            _audit.log(
                event_type="TIGHT_TRAIL_PLACED",
                entity_type="order",
                entity_id=stop_order.id,
                description=(
                    f"SELL {trail_req.order_type} {symbol} x{quantity} {_detail} "
                    f"(assigned SELL @ ${signal_price:.2f}, signal_id={signal_id})"
                ),
            )
            logger.info(
                "[exec] tighten_trail %s: placed %s (%s, signal $%.2f) "
                "signal_id=%s broker_id=%s",
                symbol, trail_req.order_type, _detail, signal_price,
                signal_id, resp.broker_order_id,
            )
            return True
        except Exception as exc:
            # Primary trail placement failed (broker rejection, network blip,
            # instrument not eligible for TRAILING_STOP, etc.). Fall back to a
            # static STOP ~trail_pct% below the current price so the position is
            # still protected even if the native trail couldn't be placed.
            logger.error(
                "[exec] tighten_trail %s: failed to place tight trail (%s) "
                "— attempting fallback static STOP.",
                symbol, exc,
            )
            if current_price > 0:
                fallback_stop = round(current_price * (1.0 - trail_pct / 100.0), 2)
            else:
                logger.error(
                    "[exec] tighten_trail %s: fallback STOP needs a live price "
                    "but none available — POSITION LEFT UNPROTECTED.", symbol,
                )
                return False
            fallback_req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="STOP",
                quantity=quantity,
                stop_price=fallback_stop,
                time_in_force="GTC",
                source=source,  # type: ignore[arg-type]
                idempotency_key=f"sell-trail-fallback-{symbol}-{suffix}",
            )
            try:
                resp2 = await self.broker.place_order(fallback_req, account_id)
                fb_order = self._persist_order(fallback_req, signal_id, status="submitted")
                self._update_order_status(
                    fb_order.id,
                    status="submitted",
                    broker_order_id=resp2.broker_order_id,
                    submitted_at=datetime.now(tz=timezone.utc),
                )
                _audit.log(
                    event_type="TIGHT_TRAIL_FALLBACK",
                    entity_type="order",
                    entity_id=fb_order.id,
                    description=(
                        f"Fallback STOP {symbol} x{quantity} @ ${fallback_stop:.2f} "
                        f"(primary trail failed: {exc}; signal ${signal_price:.2f})"
                    ),
                )
                logger.warning(
                    "[exec] tighten_trail %s: fallback STOP placed @ $%.2f "
                    "(current − %.1f%%). Position protected. broker_id=%s",
                    symbol, fallback_stop, trail_pct, resp2.broker_order_id,
                )
                return True
            except Exception as exc2:
                logger.error(
                    "[exec] tighten_trail %s: fallback STOP also failed (%s) "
                    "— POSITION LEFT UNPROTECTED. Investigate manually.",
                    symbol, exc2,
                )
                return False

    async def _buying_power_preflight(
        self,
        order_req: OrderRequest,
        account_id: str,
        estimated_price: Optional[float],
    ) -> Optional[str]:
        """Return a blocking reason string if the broker doesn't have enough
        buying power to cover this BUY; return None to allow.

        SELLs are skipped (they free capital). Disabled when
        settings.buying_power_check_enabled is False, or when the broker
        can't return a usable buying_power figure — in those cases we fall
        through and let the broker itself reject the order.
        """
        settings = get_settings()
        if not settings.buying_power_check_enabled:
            return None
        if order_req.side != "BUY":
            return None

        # Need a price to compute order value. Use limit_price for LIMIT,
        # estimated_price (passed by scheduler/autotrader) otherwise.
        price = order_req.limit_price or estimated_price
        if not price or price <= 0:
            logger.debug("[exec] Buying-power preflight skipped — no usable price")
            return None
        order_value = float(price) * float(order_req.quantity)
        buffer = float(settings.buying_power_min_buffer_usd or 0.0)
        needed = order_value + buffer

        try:
            accounts = await self.broker.get_accounts()
        except Exception as exc:
            logger.warning("[exec] Buying-power preflight: get_accounts failed (%s) — allowing order through", exc)
            return None

        # Pick the matching account by id when supplied; otherwise sum all
        # accounts returned. MultiBroker fans out get_accounts, so for a
        # "both" routing the available pool is the sum of Schwab + Webull
        # buying power. That matches the fact that place_order also fans
        # out — each leg consumes its own broker's capital.
        if account_id:
            target = [a for a in accounts if a.account_id == account_id]
            if not target:
                target = accounts  # fallback: id didn't match (multi-broker case)
        else:
            target = accounts
        bp_total = 0.0
        any_known = False
        for a in target:
            if a.buying_power is None:
                continue
            any_known = True
            bp_total += float(a.buying_power)
        if not any_known:
            logger.debug("[exec] Buying-power preflight skipped — broker returned no figure")
            return None

        if bp_total < needed:
            return (
                f"insufficient buying power: need ${needed:,.2f} "
                f"(order ${order_value:,.2f} + buffer ${buffer:,.2f}) "
                f"but have ${bp_total:,.2f}"
            )
        return None

    async def _confirm_broker_status(
        self, broker_order_id: Optional[str], account_id: str, order_id: int
    ) -> Optional[OrderStatusResponse]:
        """One-shot status poll after place_order. Updates DB to the real broker state."""
        if not broker_order_id:
            return None
        try:
            confirmed = await self.broker.get_order(broker_order_id, account_id)
        except Exception as exc:
            logger.warning("[exec] Status confirm failed for %s: %s", broker_order_id, exc)
            return None

        # Map broker status into our lifecycle. Anything that isn't a working state
        # (queued/working/pending_activation) overrides "submitted".
        broker_status = (confirmed.status or "").lower()
        terminal = {"filled", "partial", "rejected", "cancelled", "canceled", "expired", "replaced"}
        # Webull names a partial fill "partial_filled"; normalize to our "partial".
        if broker_status in ("partial", "partial_filled", "partially_filled"):
            broker_status = "partial"
        if broker_status in terminal:
            local_status = broker_status
            if local_status == "canceled":
                local_status = "cancelled"
            reason = ""
            if isinstance(confirmed.raw, dict):
                reason = confirmed.raw.get("statusDescription") or ""
            # Persist the fill price/qty and broker order id too — without these
            # an order could flip to "filled" yet show no price (the Webull bug
            # we hit on the SO order). filled_at is best-effort: use now() since
            # the broker timestamp format varies by venue.
            filled_at = (
                datetime.now(tz=timezone.utc)
                if local_status in ("filled", "partial") else None
            )
            self._update_order_status(
                order_id,
                status=local_status,
                broker_order_id=confirmed.broker_order_id or None,
                fill_price=confirmed.fill_price,
                filled_at=filled_at,
                error_message=reason or None,
            )
            _audit.log(
                event_type=f"ORDER_{local_status.upper()}",
                entity_type="order",
                entity_id=order_id,
                description=f"broker confirmed status={local_status} reason={reason or 'n/a'}",
            )
            logger.info(
                "[exec] Broker confirmed status=%s for order_id=%s (%s)",
                local_status, order_id, reason or "no reason",
            )
        return confirmed

    # ── Persistence helpers ──────────────────────────────────────────────────

    def _persist_order(self, req: OrderRequest, signal_id: Optional[int], status: str) -> Order:
        with SessionLocal() as db:
            order = Order(
                broker=self.broker.name,
                symbol=req.symbol,
                side=req.side,
                order_type=req.order_type,
                quantity=req.quantity,
                limit_price=req.limit_price,
                stop_price=req.stop_price,
                trail_type=getattr(req, "trail_type", None) if req.order_type == "TRAILING_STOP" else None,
                trail_value=getattr(req, "trail_value", None) if req.order_type == "TRAILING_STOP" else None,
                status=status,
                is_paper=(self.broker.name == "paper"),
                signal_id=signal_id,
                idempotency_key=req.idempotency_key,
                source=getattr(req, "source", "manual") or "manual",
                created_at=datetime.now(tz=timezone.utc),
            )
            db.add(order)
            db.commit()
            db.refresh(order)
            return order

    def _update_order_status(
        self,
        order_id: int,
        status: str,
        broker_order_id: Optional[str] = None,
        submitted_at: Optional[datetime] = None,
        filled_at: Optional[datetime] = None,
        fill_price: Optional[float] = None,
        error_message: Optional[str] = None,
        preview_json: Optional[str] = None,
    ) -> None:
        with SessionLocal() as db:
            order = db.query(Order).filter_by(id=order_id).first()
            if not order:
                return
            order.status = status
            if broker_order_id:
                order.broker_order_id = broker_order_id
            if submitted_at:
                order.submitted_at = submitted_at
            if filled_at:
                order.filled_at = filled_at
            if fill_price is not None:
                order.fill_price = fill_price
            if error_message:
                order.error_message = error_message
            if preview_json:
                order.preview_json = preview_json
            db.commit()

    def _persist_preview(self, order_id: int, preview) -> None:
        with SessionLocal() as db:
            prev = OrderPreview(
                order_id=order_id,
                broker=self.broker.name,
                estimated_cost=preview.estimated_cost,
                estimated_commission=preview.estimated_commission,
                buying_power_effect=preview.buying_power_effect,
                raw_response_json=json.dumps(preview.raw),
            )
            db.add(prev)
            db.commit()

    def _persist_execution_event(
        self,
        order_id: int,
        event_type: str,
        fill_price: Optional[float] = None,
        qty: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        with SessionLocal() as db:
            exec_rec = Execution(
                order_id=order_id,
                broker=self.broker.name,
                event_type=event_type,
                fill_price=fill_price,
                quantity_filled=qty,
                occurred_at=datetime.now(tz=timezone.utc),
                raw_response_json=json.dumps({"error": error}) if error else None,
            )
            db.add(exec_rec)
            db.commit()

    def _handle_fill(self, order_id: int, status: OrderStatusResponse) -> None:
        fill_status = "filled" if status.status == "filled" else "partial"
        self._update_order_status(
            order_id,
            status=fill_status,
            fill_price=status.fill_price,
            filled_at=datetime.now(tz=timezone.utc),
        )
        self._persist_execution_event(
            order_id,
            event_type="fill" if fill_status == "filled" else "partial_fill",
            fill_price=status.fill_price,
            qty=status.filled_quantity,
        )
        logger.info(f"[exec] Fill recorded: order_id={order_id} price={status.fill_price or 0:.4f}")
