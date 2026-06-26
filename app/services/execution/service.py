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
from types import SimpleNamespace
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

    def _peak_since_signal(self, symbol: str, signal_at) -> float:
        """Highest traded price for `symbol` since the SELL signal fired.

        A trailing stop ratchets off the high-water mark since arming, never the
        current price — so the stop must be measured from this peak. When the
        signal is from today we pull intraday bars (so an intraday spike-and-
        fade is captured); otherwise daily highs since the signal date. Returns
        0.0 on any failure (caller falls back to current price / resting stop).
        """
        try:
            from datetime import datetime as _dt, timezone as _tz
            from app.services.market_data.provider import get_ohlcv

            sig_day = None
            if signal_at is not None:
                try:
                    sig_day = str(signal_at)[:10]
                except Exception:
                    sig_day = None

            # Intraday peak when the signal is recent (today/yesterday): a daily
            # bar would miss a same-session run-up-and-pullback like AMAL's.
            is_recent = False
            if signal_at is not None:
                try:
                    sat = signal_at
                    if getattr(sat, "tzinfo", None) is None:
                        sat = sat.replace(tzinfo=_tz.utc)
                    is_recent = (_dt.now(_tz.utc) - sat).total_seconds() <= 2 * 86400
                except Exception:
                    is_recent = False

            if is_recent:
                try:
                    intraday = get_ohlcv(symbol, period="5d", interval="5m")
                    if intraday is not None and not intraday.empty:
                        col = "High" if "High" in intraday.columns else "Close"
                        if sig_day is not None:
                            try:
                                sliced = intraday.loc[sig_day:]
                                if not sliced.empty:
                                    return float(sliced[col].max())
                            except Exception:
                                pass
                        return float(intraday[col].max())
                except Exception:
                    pass  # interval unsupported / provider error → daily below

            df = get_ohlcv(symbol, period="3mo")
            if df is None or df.empty:
                return 0.0
            col = "High" if "High" in df.columns else "Close"
            if sig_day is not None:
                try:
                    sliced = df.loc[sig_day:]
                    if not sliced.empty:
                        return float(sliced[col].max())
                except Exception:
                    pass
            return float(df[col].max())
        except Exception:
            return 0.0

    def _stored_trail_peak(self, symbol: str, signal_id: Optional[int]) -> float:
        """Read-only durable high-water mark for an armed trail (0.0 if none).

        Used by the arm gate to tell whether the trail has EVER cleared the floor
        (i.e. already armed). Matches the same anchor key as _advance_trail_peak:
        the specific signal_id when present, else the latest signal-less row.
        """
        from app.models.trail_peaks import TrailPeak

        with SessionLocal() as db:
            q = db.query(TrailPeak).filter(TrailPeak.symbol == symbol.upper())
            if signal_id is not None:
                q = q.filter(TrailPeak.signal_id == signal_id)
            else:
                q = q.filter(TrailPeak.signal_id.is_(None))
            row = q.order_by(TrailPeak.id.desc()).first()
            return float(row.peak_price or 0.0) if row is not None else 0.0

    def _advance_trail_peak(
        self, symbol: str, signal_id: Optional[int], signal_price: float,
        *, candidate_peak: float,
    ) -> float:
        """Read + advance the durable high-water mark for this armed trail.

        Returns max(stored_peak, candidate_peak) and persists it when the
        candidate is higher, so the stored peak is monotonic non-decreasing for
        the life of the trail (keyed by signal_id). Logs every advance. Falls
        back to candidate_peak on any DB error (never lowers the trail).
        """
        from app.models.trail_peaks import TrailPeak

        try:
            with SessionLocal() as db:
                q = db.query(TrailPeak).filter(TrailPeak.symbol == symbol.upper())
                # Match the specific anchor signal when we have one; else the
                # latest row for the symbol (a trail with no signal_id link).
                if signal_id is not None:
                    q = q.filter(TrailPeak.signal_id == signal_id)
                else:
                    q = q.filter(TrailPeak.signal_id.is_(None))
                row = q.order_by(TrailPeak.id.desc()).first()

                now = datetime.now(tz=timezone.utc)
                if row is None:
                    row = TrailPeak(
                        symbol=symbol.upper(),
                        signal_id=signal_id,
                        signal_price=signal_price or None,
                        peak_price=round(candidate_peak, 4),
                        peak_at=now,
                    )
                    db.add(row)
                    db.commit()
                    logger.info(
                        "[exec] trail peak %s: initialised high-water mark $%.2f "
                        "(signal_id=%s, signal $%.2f).",
                        symbol, candidate_peak, signal_id, signal_price or 0.0,
                    )
                    return candidate_peak

                stored = float(row.peak_price or 0.0)
                if candidate_peak > stored + 1e-9:
                    row.peak_price = round(candidate_peak, 4)
                    row.peak_at = now
                    db.commit()
                    logger.info(
                        "[exec] trail peak %s: advanced high-water mark $%.2f → $%.2f "
                        "(signal_id=%s).",
                        symbol, stored, candidate_peak, signal_id,
                    )
                    return candidate_peak
                # Stored peak wins — today's high is below the prior peak.
                logger.debug(
                    "[exec] trail peak %s: stored peak $%.2f ≥ candidate $%.2f — "
                    "trail holds at the prior high (no drop with price).",
                    symbol, stored, candidate_peak,
                )
                return stored
        except Exception as exc:
            logger.warning(
                "[exec] trail peak %s: persistence failed (%s) — using candidate "
                "$%.2f for this cycle.", symbol, exc, candidate_peak,
            )
            return candidate_peak

    async def tighten_trail_on_sell(
        self,
        symbol: str,
        quantity: float,
        account_id: str,
        signal_price: float,
        *,
        signal_at=None,
        trail_pct: float = 2.0,
        floor_buffer_pct: float = 0.25,
        source: str = "scheduler",
        idempotency_suffix: str = "",
        signal_id: Optional[int] = None,
        force_replace: bool = False,
    ) -> bool:
        """Approach C — arm a bot-managed, floored trailing stop after a SELL.

        Called ONLY when the ASSIGNED strategy for this symbol fires a SELL
        signal. Other strategies signalling SELL on the same symbol are ignored
        — the caller (scheduler._run_cycle) is already filtered to assigned-
        strategy signals only via the signals_to_act list.

        The user's rule (do NOT market-sell on the SELL signal — the stock
        usually rides further before reversing) has two parts the BOT enforces
        itself, because no broker offers both natively:

          1. ARM GATE — don't place any trailing protection until price has
             crossed signal_price × (1 + floor_buffer_pct/100). Below that the
             SELL is "pending arm": we return True (the position keeps whatever
             protective stop it already had) and re-check next cycle.

          2. HARD FLOOR — once armed, the protective stop is
                 max(current_price × (1 − trail_pct/100), floor)
             where floor = signal_price × (1 + floor_buffer_pct/100). The trail
             follows price up, but the stop can NEVER drop below the floor, so a
             reversal still exits at least floor_buffer_pct% ABOVE the signal.

        We place a STATIC STOP (not a broker-native TRAILING_STOP) on ALL
        brokers and re-compute/ratchet it every cycle, because a native % trail
        computes its trigger as high−trail% and would happily sit below the
        floor — defeating part 2. This is the bot-computed management the user
        asked for ("if it is not possible at the broker level, you as a bot
        compute them and place the orders").

        Idempotency / ratchet: a resting stop is only cancel/replaced when the
        freshly-computed target is materially HIGHER (> 0.1%) than the resting
        one — so a persisting SELL condition across many cycles doesn't churn
        cancel/replace orders, and the stop only ever moves up. force_replace=True
        re-arms regardless (e.g. trail params changed).

        trail_pct defaults to 2.0% — wide enough to absorb normal intraday noise
        before exiting while still capturing most of the post-signal upside.
        floor_buffer_pct defaults to 0.25% — the arm threshold AND the floor.

        signal_id: the Signal row id for this SELL signal. Linked to the stop
        order so PnL audit can show signal_price vs actual exit price.

        Returns True if a protective order is resting/placed OR the SELL is
        still pending the arm gate; False only on a placement failure that
        leaves the position unprotected.
        """
        # ── Live price first — needed for BOTH the arm gate and the floor. ──
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

        # floor = signal × (1 + floor_buffer_pct/100): the lowest the protective
        # stop may ever sit (locks in floor_buffer_pct% over the signal). Also the
        # ARM threshold — price must clear the floor before we place anything.
        floor = round(signal_price * (1.0 + floor_buffer_pct / 100.0), 2) if signal_price > 0 else 0.0

        if current_price <= 0:
            logger.error(
                "[exec] tighten_trail %s: no live price — cannot evaluate arm gate "
                "or place a floored stop. Investigate manually.", symbol,
            )
            return False

        # Has this trail EVER armed? The arm gate (below) only governs the FIRST
        # arming — once price has cleared the floor at any point, the trail is
        # live and must keep protecting even after price falls back below the
        # floor. We read the durable high-water mark (trail_peaks) and treat the
        # trail as armed if that stored peak ever reached the floor. Without this
        # check a momentum SELL that ran up (peak $96) then reversed below the
        # floor (e.g. IRMD: signal $95, peak $96, now $93.87) wrongly fell back
        # into "pending arm" and rode the position ALL THE WAY DOWN, unprotected,
        # never exiting — the exact bug this guards against.
        already_armed = False
        try:
            stored_peak = self._stored_trail_peak(symbol, signal_id)
            if floor > 0 and stored_peak >= floor:
                already_armed = True
        except Exception as exc:
            logger.debug("[exec] tighten_trail %s: stored-peak read failed: %s", symbol, exc)

        # ── ARM GATE ────────────────────────────────────────────────────────
        # Hold off until price has crossed the floor (signal + floor_buffer_pct%).
        # Below it, the SELL is "pending arm": ride the momentum, re-check next
        # cycle. The position keeps whatever protective stop it already had, so
        # it is not naked. Return True so the caller doesn't treat this as failure.
        # SKIPPED once the trail has already armed (stored peak cleared the floor):
        # from then on we always evaluate the trail-hit / ratchet logic below, so a
        # reversal exits instead of riding the position down.
        if floor > 0 and current_price < floor and not already_armed:
            logger.info(
                "[exec] tighten_trail %s: SELL pending arm — current $%.2f < floor "
                "$%.2f (signal $%.2f + %.2f%%). Riding momentum; re-check next cycle.",
                symbol, current_price, floor, signal_price, floor_buffer_pct,
            )
            return True
        if already_armed and current_price < floor:
            logger.info(
                "[exec] tighten_trail %s: trail already armed (stored peak ≥ floor "
                "$%.2f) but price $%.2f fell back below floor — evaluating trail-hit "
                "exit (do NOT re-enter pending-arm).",
                symbol, floor, current_price,
            )

        suffix = idempotency_suffix or str(int((signal_price or current_price) * 100))

        # Step 0: idempotency / ratchet. Fetch resting orders ONCE and reuse the
        # list for peak reconstruction, the "already protected?" check and the
        # cancel sweep. A resting STOP does NOT sit in "working" on every broker —
        # Schwab parks it in AWAITING_STOP_CONDITION / PENDING_ACTIVATION until the
        # trigger is hit. Querying only "working" missed it, so the guard re-placed
        # the SAME stop every cycle → broker accepted a fresh order each time and
        # the deterministic idempotency_key collided in the DB. Sweep all the
        # pending statuses (same set the scheduler reconcile uses) and dedup.
        open_orders: list = []
        seen_boids: set = set()
        for _status in (
            "working", "awaiting_stop_condition", "pending_activation",
            "submitted", "queued", "accepted", "pending_acknowledgement",
        ):
            try:
                for o in (await self.broker.list_orders(account_id, status=_status)) or []:
                    boid = getattr(o, "broker_order_id", None)
                    if boid and boid in seen_boids:
                        continue
                    if boid:
                        seen_boids.add(boid)
                    open_orders.append(o)
            except Exception as exc:
                logger.debug(
                    "[exec] tighten_trail %s: list_orders(status=%s) failed: %s",
                    symbol, _status, exc,
                )

        resting_sell_stops = [
            o for o in open_orders
            if getattr(o, "symbol", "").upper() == symbol.upper()
            and getattr(o, "side", "").upper() == "SELL"
            and getattr(o, "order_type", "").upper() in ("STOP", "TRAILING_STOP")
            and getattr(o, "broker_order_id", None)
        ]

        # ── DB supplement: some brokers' working-order list can't be relied on to
        # surface an existing rest. Webull's list_orders hits /trade/orders/list-
        # today (TODAY ONLY), so a tight trail armed on a prior day is invisible to
        # the broker sweep above — which made the native-trail "leave it alone"
        # guard miss, so every cycle RE-PLACED the native TRAILING_STOP and reset
        # its ratchet to the current price (it could never climb → no Approach C
        # profit capture). We persisted that order with its broker_order_id, so
        # merge our own still-resting STOP/TRAILING_STOP rows for this symbol that
        # the broker sweep didn't already return. Broker remains source of truth
        # for level; this only ensures we SEE the rest exists.
        try:
            known_boids = {
                getattr(o, "broker_order_id", None) for o in resting_sell_stops
            }
            with SessionLocal() as _db:
                db_rows = (
                    _db.query(Order)
                    .filter(
                        Order.symbol == symbol.upper(),
                        Order.side == "SELL",
                        Order.order_type.in_(["STOP", "TRAILING_STOP"]),
                        Order.status.in_(["submitted", "working"]),
                        Order.broker_order_id.isnot(None),
                    )
                    .all()
                )
            for r in db_rows:
                if r.broker_order_id in known_boids:
                    continue
                known_boids.add(r.broker_order_id)
                # Shape it like an OrderStatusResponse for the consumers below.
                resting_sell_stops.append(
                    SimpleNamespace(
                        broker_order_id=r.broker_order_id,
                        symbol=r.symbol,
                        side="SELL",
                        order_type=r.order_type,
                        stop_price=r.stop_price,
                        trail_value=r.trail_value,
                        status=r.status,
                        raw={},
                    )
                )
        except Exception as exc:
            logger.debug(
                "[exec] tighten_trail %s: DB resting-stop supplement failed: %s",
                symbol, exc,
            )

        # Highest resting static-STOP level (broker = source of truth).
        resting_level = 0.0
        for o in resting_sell_stops:
            if getattr(o, "order_type", "").upper() == "TRAILING_STOP":
                continue
            lvl = getattr(o, "stop_price", None)
            if lvl is None:
                try:
                    lvl = float(getattr(o, "raw", {}).get("stopPrice", 0) or 0)
                except Exception:
                    lvl = 0.0
            resting_level = max(resting_level, float(lvl or 0.0))

        # ── TARGET STOP: ratchet off the PEAK SINCE THE SIGNAL, not the current
        # price. A trailing stop must follow the high-water mark — if price ran to
        # $45 a few days ago then faded to $42, the stop stays at 45×(1−trail%),
        # it does NOT drop to 42×(1−trail%).
        #
        # The peak is a DURABLE high-water mark persisted in trail_peaks (keyed by
        # signal_id), so it survives provider gaps and intraday-vs-daily quirks and
        # is the authoritative value. We seed/advance it each cycle from:
        #   • the stored peak                        (authoritative; today's $45)
        #   • current_price                          (live high)
        #   • peak traded since the signal (OHLCV)    (seeds the first write;
        #                                              daily High since signal date
        #                                              captures an older peak)
        #   • the resting STOP's implied peak = stop / (1 − trail%)  (last cycle's
        #                                              locked-in peak; lag-proof)
        # The max is then written back, so the stored peak is monotonic non-
        # decreasing for the life of the trail.
        peak = current_price
        try:
            hist_peak = self._peak_since_signal(symbol, signal_at)
            if hist_peak > peak:
                peak = hist_peak
        except Exception as exc:
            logger.debug("[exec] tighten_trail %s: peak lookup failed: %s", symbol, exc)
        if resting_level > 0 and trail_pct < 100.0:
            implied_peak = resting_level / (1.0 - trail_pct / 100.0)
            if implied_peak > peak:
                peak = implied_peak

        # Merge with (and write back) the durable stored peak. This both reads the
        # authoritative high-water mark and advances it to today's high.
        peak = self._advance_trail_peak(
            symbol, signal_id, signal_price, candidate_peak=peak,
        )

        # trail follows the PEAK up, floored at `floor`.
        trail_level = round(peak * (1.0 - trail_pct / 100.0), 2)
        target_stop = max(trail_level, floor)
        logger.info(
            "[exec] tighten_trail %s: peak $%.2f → trail level $%.2f, floor $%.2f, "
            "target stop $%.2f (current $%.2f, signal $%.2f).",
            symbol, peak, trail_level, floor, target_stop, current_price, signal_price,
        )

        # ── TRAIL ALREADY HIT → EXIT NOW (not an invalid stop) ───────────────
        # A SELL STOP must rest BELOW the market. If the computed stop is at/above
        # the current price, price has already pulled back THROUGH the trail (or the
        # floor sits above market) — the stop's trigger condition is already met. A
        # broker rejects a SELL STOP placed above market, which is exactly the
        # AMAL reject/idempotency-collision loop (stop $44.33 vs current $44.00).
        # The correct action is to SELL NOW at market: the trail did its job (rode
        # the peak, price reversed past the trail distance). We cancel any resting
        # protective stops first so we don't double up.
        if current_price <= target_stop:
            logger.info(
                "[exec] tighten_trail %s: trail HIT — target stop $%.2f ≥ current "
                "$%.2f. Exiting at market now (trail rode peak $%.2f, price reversed).",
                symbol, target_stop, current_price, peak,
            )
            for o in resting_sell_stops:
                try:
                    await self.broker.cancel_order(o.broker_order_id, account_id)
                except Exception as exc:
                    logger.warning(
                        "[exec] tighten_trail %s: could not cancel resting %s before "
                        "market exit (%s)", symbol, getattr(o, "order_type", "stop"), exc,
                    )
            market_req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                quantity=quantity,
                time_in_force="DAY",
                source=source,  # type: ignore[arg-type]
                idempotency_key=f"sell-trail-exit-{symbol}-{suffix}-{int(current_price * 100)}",
            )
            try:
                resp = await self.broker.place_order(market_req, account_id)
                ex_order = self._persist_order(market_req, signal_id, status="submitted")
                self._update_order_status(
                    ex_order.id, status="submitted",
                    broker_order_id=resp.broker_order_id,
                    submitted_at=datetime.now(tz=timezone.utc),
                )
                _audit.log(
                    event_type="TIGHT_TRAIL_EXIT",
                    entity_type="order",
                    entity_id=ex_order.id,
                    description=(
                        f"SELL MARKET {symbol} x{quantity} — trail hit (stop "
                        f"${target_stop:.2f} ≥ current ${current_price:.2f}, peak "
                        f"${peak:.2f}, signal ${signal_price:.2f}, signal_id={signal_id})"
                    ),
                )
                logger.info(
                    "[exec] tighten_trail %s: market exit placed (trail hit) "
                    "broker_id=%s", symbol, resp.broker_order_id,
                )
                return True
            except Exception as exc:
                logger.error(
                    "[exec] tighten_trail %s: trail-hit market exit FAILED (%s) — "
                    "POSITION STILL OPEN. Investigate manually.", symbol, exc,
                )
                return False

        # NATIVE HANDOFF DECISION. Hand off to a broker-native TRAILING_STOP only
        # when BOTH hold:
        #   (a) the trail level is at/above the floor — so native (which anchors
        #       to the current price) can't sit below the floor, AND
        #   (b) current price is AT the peak (within a hair) — a native trail
        #       anchors to the CURRENT price, so handing off during a pullback
        #       would re-anchor LOWER than where we've already ratcheted the
        #       bot STOP (the AMAL bug). On a pullback we keep the peak-based
        #       static STOP; we only hand off when price makes a fresh high.
        # When handed off, native then follows tick-by-tick with zero poll lag.
        native_ok = bool(getattr(self.broker, "supports_native_trailing_stop", False))
        at_peak = current_price >= peak * 0.999
        use_native = native_ok and trail_level >= floor and at_peak

        # Classify resting orders. A HEALTHY native trail we already handed off to
        # must be left alone — re-placing it would reset its ratchet to the
        # current price every cycle (so it could never climb). The resting static
        # STOP level (resting_level, computed above) is compared to the target so
        # we only ratchet UP.
        resting_native = [
            o for o in resting_sell_stops
            if getattr(o, "order_type", "").upper() == "TRAILING_STOP"
        ]

        # Leave a healthy native trail in place (it self-ratchets) — BUT only if
        # it is already at least as TIGHT as the trail we now want. The protective
        # stop placed on the BUY fill uses a wide ATR-based trail (~2.5–9%); once
        # the ASSIGNED strategy fires its SELL, Approach C wants the tighter
        # assignment trail (often 1–2%) so we exit closer to the peak. If the
        # resting native trail is LOOSER (larger %) than the new tight trail_pct,
        # leaving it would give back more profit than intended — so we fall
        # through and replace it. We only "leave it" when its recorded trail % is
        # ≤ the target (already tight enough); an unknown/None trail value is
        # treated as looser so the tighter trail wins.
        if resting_native and use_native and not force_replace:
            # Replace ONLY when we have a KNOWN, materially-looser resting trail %
            # (the wide ATR protective-stop placed on the BUY fill, recorded in our
            # DB with its trail_value). An unknown/None trail value means a broker
            # order we can't compare — leave it alone (re-placing would reset its
            # ratchet to the current price, the AMAL bug). max() = loosest resting.
            known_pcts = [
                float(getattr(o, "trail_value", None) or 0.0)
                for o in resting_native
            ]
            loosest_known = max((p for p in known_pcts if p > 0), default=0.0)
            resting_is_looser = loosest_known > trail_pct + 1e-9
            if not resting_is_looser:
                logger.info(
                    "[exec] tighten_trail %s: native TRAILING_STOP already resting "
                    "(%.2f%%%s ≤ target %.2f%%, trail level $%.2f ≥ floor $%.2f) — "
                    "broker self-ratchets, leaving it.",
                    symbol, loosest_known,
                    "" if loosest_known > 0 else "/unknown",
                    trail_pct, trail_level, floor,
                )
                return True
            logger.info(
                "[exec] tighten_trail %s: resting native trail %.2f%% is LOOSER than "
                "target %.2f%% — replacing with the tighter Approach-C trail so we "
                "exit closer to the peak.",
                symbol, loosest_known, trail_pct,
            )

        # Static STOP already at/above target (and we still want a static STOP) →
        # no ratchet needed. Only ever move UP, ignore sub-0.1% noise, so a
        # persisting SELL condition doesn't churn cancel/replace every cycle.
        if (
            resting_sell_stops and not resting_native
            and not use_native and not force_replace
            and resting_level >= target_stop * 0.999
        ):
            logger.info(
                "[exec] tighten_trail %s: resting STOP $%.2f already >= target "
                "$%.2f (floor $%.2f) — no ratchet needed.",
                symbol, resting_level, target_stop, floor,
            )
            return True

        # Step 1: build the replacement order. We PLACE it BEFORE cancelling the
        # old resting stop (place-then-cancel), so a rejected/failed replacement
        # never leaves the position naked — the old stop stays put until the new
        # one is confirmed. (The old cancel-then-place order left positions
        # unprotected whenever the new stop was rejected, e.g. a stop above
        # market.)
        if use_native:
            # Native trailing stop — broker follows tick-by-tick from here. Safe
            # because trail_level ≥ floor: even native's initial trigger
            # (current − trail%) sits at/above the floor.
            trail_req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="TRAILING_STOP",
                quantity=quantity,
                trail_type="PERCENT",
                trail_value=round(trail_pct, 2),
                time_in_force="GTC",
                source=source,  # type: ignore[arg-type]
                idempotency_key=f"sell-trail-{symbol}-{suffix}-native",
            )
            logger.info(
                "[exec] tighten_trail %s: HANDOFF to native %.1f%% TRAILING_STOP "
                "(trail level $%.2f ≥ floor $%.2f, current $%.2f). Broker now "
                "ratchets tick-by-tick — captures fast run-ups with no poll lag.",
                symbol, trail_pct, trail_level, floor, current_price,
            )
        else:
            # Bot-managed static STOP, floored. Used until the trail level clears
            # the floor (then we hand off to native above), and ALWAYS for brokers
            # with no native trail (Zerodha). The fast trail job ratchets this.
            trail_req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="STOP",
                quantity=quantity,
                stop_price=target_stop,
                time_in_force="GTC",
                source=source,  # type: ignore[arg-type]
                idempotency_key=f"sell-trail-{symbol}-{suffix}-{int(target_stop * 100)}",
            )
            logger.info(
                "[exec] tighten_trail %s: floored STOP @ $%.2f = max(trail $%.2f, "
                "floor $%.2f) — current $%.2f, signal $%.2f, trail %.1f%%%s.",
                symbol, target_stop, trail_level, floor, current_price, signal_price,
                trail_pct, " [floor active]" if target_stop == floor else "",
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
            # New stop confirmed — NOW cancel the old resting stops it replaces.
            # (place-then-cancel: never naked if the new order is rejected.)
            for o in resting_sell_stops:
                try:
                    await self.broker.cancel_order(o.broker_order_id, account_id)
                    logger.info(
                        "[exec] tighten_trail %s: cancelled superseded resting %s",
                        symbol, getattr(o, "order_type", "stop"),
                    )
                except Exception as exc:
                    logger.warning(
                        "[exec] tighten_trail %s: could not cancel superseded %s (%s) "
                        "— new stop is live; old may linger until next reconcile",
                        symbol, getattr(o, "order_type", "stop"), exc,
                    )
            _detail = (
                f"native TRAILING_STOP trail {trail_pct}%"
                if use_native
                else f"STOP @ ${target_stop:.2f} (trail {trail_pct}%, floor ${floor:.2f})"
            )
            _audit.log(
                event_type="TIGHT_TRAIL_PLACED",
                entity_type="order",
                entity_id=stop_order.id,
                description=(
                    f"SELL {_detail} {symbol} x{quantity} "
                    f"(assigned SELL @ ${signal_price:.2f}, signal_id={signal_id})"
                ),
            )
            logger.info(
                "[exec] tighten_trail %s: placed %s (signal $%.2f) "
                "signal_id=%s broker_id=%s",
                symbol, trail_req.order_type, signal_price, signal_id,
                resp.broker_order_id,
            )
            return True
        except Exception as exc:
            # Replacement failed — the OLD resting stop was NOT cancelled (we
            # place before cancelling), so the position keeps its prior protection
            # rather than going naked.
            still_protected = bool(resting_sell_stops)
            logger.error(
                "[exec] tighten_trail %s: failed to place %s (%s). %s",
                symbol, trail_req.order_type, exc,
                "Prior stop still resting — position remains protected."
                if still_protected
                else "POSITION LEFT UNPROTECTED. Investigate manually.",
            )
            # Return True when the prior stop still protects us, so the caller
            # doesn't flag the signal for an immediate retry loop.
            return still_protected

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
        from sqlalchemy.exc import IntegrityError

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
            try:
                db.commit()
            except IntegrityError:
                # Duplicate idempotency_key — a row for this logical order already
                # exists (e.g. a prior cycle placed it). This is NOT a failure: the
                # broker order is live. Return the existing row so the caller treats
                # it as success rather than logging "POSITION LEFT UNPROTECTED".
                db.rollback()
                existing = (
                    db.query(Order)
                    .filter(Order.idempotency_key == req.idempotency_key)
                    .first()
                )
                if existing is not None:
                    logger.info(
                        "[exec] _persist_order: idempotency_key %r already persisted "
                        "(order id=%s) — reusing existing row.",
                        req.idempotency_key, existing.id,
                    )
                    return existing
                raise
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
