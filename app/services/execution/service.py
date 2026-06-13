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

        if settings.trailing_stop_enabled:
            # Volatility-aware trail: use ATR% (ATR/price × 100) scaled by a
            # multiplier that adjusts for how volatile the symbol is.
            #
            # ATR% buckets (based on 22-day ATR as % of price):
            #   < 1.5%  → low-vol  (KO, XOM, SO)   → 2× ATR trail (~2.5–3%)
            #   1.5–3%  → mid-vol  (NVDA, BAC, COP) → 2.5× ATR trail (~4–7%)
            #   > 3%    → high-vol (TSLA, PLTR)      → 3× ATR trail (~9–12%)
            #
            # This means low-vol stocks get a tighter trail (less slippage on
            # the exit) and high-vol stocks get breathing room (normal pullbacks
            # won't shake them out).
            trail_pct = settings.trail_stop_pct  # fallback
            try:
                from app.services.market_data.provider import get_ohlcv
                from app.services.strategy.rules import _atr_raw
                df = get_ohlcv(buy_req.symbol, period="3mo")
                if not df.empty and len(df) >= 22:
                    atr = float(_atr_raw(df, 22).iloc[-1])
                    if fill_price > 0 and atr > 0:
                        atr_pct = (atr / fill_price) * 100
                        if atr_pct < 1.5:
                            mult = 2.0     # low-vol:  KO, SO        → ~3%
                        elif atr_pct < 3.0:
                            mult = 2.5     # mid-vol:  XOM, BAC, COP → ~4–7%
                        elif atr_pct < 4.5:
                            mult = 2.0     # high-vol: NVDA, TSLA    → ~7–9%
                        else:
                            mult = 1.5     # very-high: TOST, PLTR   → ~7–9%
                        trail_pct = round(atr_pct * mult, 2)
                        # Floor 1.5% (never choke), ceiling 10% (never give
                        # back more than a solid swing move).
                        trail_pct = max(1.5, min(10.0, trail_pct))
                        logger.info(
                            "[exec] Volatility trail %s: ATR=%.2f (%.1f%% of price) "
                            "× %.1f → trail=%.2f%%",
                            buy_req.symbol, atr, atr_pct, mult, trail_pct,
                        )
            except Exception as exc:
                logger.debug("[exec] ATR trail compute failed, using pct fallback: %s", exc)

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
    ) -> bool:
        """Cancel any resting STOP/TRAILING_STOP for a symbol, then place a
        tight trailing stop in its place.

        Called ONLY when the ASSIGNED strategy for this symbol fires a SELL
        signal. Other strategies signalling SELL on the same symbol are ignored
        — the caller (scheduler._run_cycle) is already filtered to assigned-
        strategy signals only via the signals_to_act list.

        trail_pct defaults to 2.0% — wider than 1% to absorb normal intraday
        noise before exiting, while still being tight enough to capture most of
        the post-signal upside. 1% was too tight for mid/high-vol names (KO,
        NVDA) and got shaken out by normal daily wicks.

        floor_buffer_pct (default 0.25%): the protective floor is set a touch
        ABOVE the signal price (signal × (1 + buffer)) so that a floor-triggered
        exit clears commission/slippage and is net-positive versus the signal,
        not merely break-even. The trailing stop is NEVER allowed to sit below
        this floor — so we can ride further upside but can never close the
        position below (signal price + buffer).

        signal_id: the Signal row id for this SELL signal. Linked to the trail
        order so PnL audit can show signal_price vs actual exit price.

        Returns True if the protective order was placed, False on failure
        (no fallback market sell — see below).
        """
        # Step 1: cancel any resting sell-side stops
        try:
            open_orders = await self.broker.list_orders(account_id, status="working")
            for o in open_orders:
                if (getattr(o, "symbol", "").upper() == symbol.upper()
                        and getattr(o, "side", "").upper() == "SELL"
                        and getattr(o, "order_type", "").upper() in ("STOP", "TRAILING_STOP")
                        and getattr(o, "broker_order_id", None)):
                    await self.broker.cancel_order(o.broker_order_id, account_id)
                    logger.info(
                        "[exec] tighten_trail %s: cancelled resting %s",
                        symbol, o.order_type,
                    )
        except Exception as exc:
            logger.warning(
                "[exec] tighten_trail %s: could not cancel resting stops (%s) "
                "— placing tight trail anyway", symbol, exc,
            )

        # Step 2: place the protective SELL order, anchored to the PEAK since the
        # SELL signal (not the current price) so it captures any run-up that
        # already happened, with a hard floor at the signal price.
        #
        # Three reference levels:
        #   floor       = signal × (1 + buffer)        — never exit below this
        #   peak_trail  = peak_since_signal × (1 - trail%)  — trail off the high
        #   native_trail= current × (1 - trail%)       — what a fresh native
        #                                                 trail would start at
        #
        # A broker-native TRAILING_STOP trails from the CURRENT price forward, so
        # it CANNOT retroactively capture a peak that occurred before arming.
        # Therefore:
        #   - If the native trail (from current) is already >= peak_trail and
        #     >= floor, use a native TRAILING_STOP (broker ratchets it up — best).
        #   - Otherwise place a STOP at max(floor, peak_trail) to lock in the
        #     run-up; the chandelier job ratchets it up on later highs.
        floor_price = round(signal_price * (1.0 + floor_buffer_pct / 100.0), 2)

        current_price = 0.0
        try:
            quotes = await self.broker.get_quotes([symbol])
            q = quotes.get(symbol.upper()) or quotes.get(symbol)
            if q is not None:
                current_price = float(q.last or q.bid or q.ask or 0.0)
        except Exception as exc:
            logger.warning(
                "[exec] tighten_trail %s: quote fetch failed (%s) — "
                "defaulting to floor STOP for safety", symbol, exc,
            )

        # Peak (high-water mark) since the SELL signal fired.
        peak_price = 0.0
        try:
            from app.services.market_data.provider import get_ohlcv
            _df = get_ohlcv(symbol, period="3mo")
            if _df is not None and not _df.empty:
                _col = "High" if "High" in _df.columns else "Close"
                # Best-effort: use the recent window's high; the scheduler passes
                # signal-anchored data when available, but a 3mo high is a safe
                # upper bound that never trails below where price has been.
                peak_price = float(_df[_col].tail(60).max())
        except Exception:
            peak_price = 0.0
        peak_price = max(peak_price, current_price, signal_price)

        native_trail  = current_price * (1.0 - trail_pct / 100.0) if current_price > 0 else 0.0
        peak_trail     = peak_price * (1.0 - trail_pct / 100.0)

        # The ideal protective level is the peak-based trail, floored at signal.
        ideal_stop = max(floor_price, peak_trail)

        # A SELL STOP must sit BELOW the current price or the broker rejects it
        # (a stop at/above market would trigger immediately). If the ideal level
        # is at/above current price, price has already pulled back THROUGH the
        # trail — cap the stop just below current so it's a valid resting order.
        # (Had a trail been resting earlier, it would already have exited here.)
        if current_price > 0:
            cap = round(current_price * 0.999, 2)   # ~0.1% below market
            target_stop = round(min(ideal_stop, cap), 2)
            # Never below the hard floor, even if that means no valid stop.
            target_stop = max(target_stop, floor_price) if cap >= floor_price else None
        else:
            target_stop = round(ideal_stop, 2)

        # Native trailing only captures forward movement from the current price.
        # Prefer it when the native start already meets the peak-based ideal AND
        # is below current (so it rests cleanly); otherwise a static STOP just
        # below market at target_stop locks in what's recoverable.
        use_native_trail = (
            native_trail > 0.0
            and native_trail >= ideal_stop
            and native_trail < current_price
        )
        use_floor = not use_native_trail

        # If even the floor sits at/above current price, price has fallen below
        # the protective floor entirely — no valid resting STOP exists. We do
        # NOT market-sell (same no-fallback safety as elsewhere); surface it.
        if use_floor and target_stop is None:
            logger.error(
                "[exec] tighten_trail %s: current $%.2f is at/below the signal "
                "floor $%.2f — cannot place a valid protective stop. Position "
                "left as-is (no fallback market sell). Investigate manually.",
                symbol, current_price, floor_price,
            )
            return False

        suffix = idempotency_suffix or str(int(signal_price * 100))
        if use_floor:
            # Static STOP just below market, anchored to the PEAK-based trail
            # (captures the run-up) and floored at signal. Chandelier ratchets up.
            trail_req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="STOP",
                quantity=quantity,
                stop_price=target_stop,
                time_in_force="GTC",
                source=source,  # type: ignore[arg-type]
                idempotency_key=f"sell-trail-{symbol}-{suffix}",
            )
            logger.info(
                "[exec] tighten_trail %s: STOP @ $%.2f = max(floor $%.2f, "
                "peak $%.2f − %.1f%% = $%.2f). Current $%.2f; native trail would "
                "start @ $%.2f (loses the run-up), so locking the peak-based level.",
                symbol, target_stop, floor_price, peak_price, trail_pct,
                peak_trail, current_price, native_trail,
            )
        else:
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
                "[exec] tighten_trail %s: native %.1f%% TRAILING_STOP — starts @ "
                "$%.2f (>= peak-based target $%.2f and floor $%.2f), broker ratchets up.",
                symbol, trail_pct, native_trail, target_stop, floor_price,
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
            _audit.log(
                event_type="TIGHT_TRAIL_PLACED",
                entity_type="order",
                entity_id=stop_order.id,
                description=(
                    f"SELL {trail_req.order_type} {symbol} x{quantity} "
                    f"{'trail=' + str(trail_pct) + '%' if trail_req.order_type == 'TRAILING_STOP' else 'stop=$' + format(target_stop, '.2f')} "
                    f"(assigned SELL @ ${signal_price:.2f}, floor ${floor_price:.2f}, "
                    f"peak ${peak_price:.2f}, signal_id={signal_id})"
                ),
            )
            logger.info(
                "[exec] tighten_trail %s: placed %s (signal $%.2f, floor $%.2f, "
                "peak $%.2f) signal_id=%s broker_id=%s",
                symbol, trail_req.order_type, signal_price, floor_price,
                peak_price, signal_id, resp.broker_order_id,
            )
            return True
        except Exception as exc:
            # SAFETY: if the trailing stop can't be placed (broker rejection,
            # network blip, instrument not eligible for TRAILING_STOP, etc.)
            # we do NOT fall back to a MARKET sell. A failed trail leaves
            # the position alone -- the user can intervene; a silent market
            # sell hides the failure AND closes a position the strategy may
            # never have intended to exit immediately.
            #
            # The previous fallback was the second escape hatch that allowed
            # SELL signals to become MARKET sells (the first being scanner
            # auto-trade; both are now closed). If the trail fails repeatedly,
            # the GTC fill-sync job will surface the absent stop and the user
            # will see the position un-protected in the dashboard.
            logger.error(
                "[exec] tighten_trail %s: failed to place tight trail (%s) "
                "-- POSITION LEFT UNPROTECTED, no fallback MARKET sell. "
                "Investigate the broker rejection and re-place manually if needed.",
                symbol, exc,
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
