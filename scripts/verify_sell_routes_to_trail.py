"""
Diagnostic: simulate a SELL signal hitting the scheduler and verify that the
scheduler routes it through tighten_trail_on_sell rather than a market sell.

This stubs the broker AND the ExecutionService so no real orders fire -- it
just observes WHICH path the scheduler takes. Prints a clear pass/fail line.

Run after restarting the backend on the new code:
    .venv/Scripts/python.exe scripts/verify_sell_routes_to_trail.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.execution import service as exec_service_module
from app.services.strategy import scheduler as sched


# ── Stub broker + execution service so no real orders fire ───────────────────

class _StubBroker:
    """Records every method call instead of making real broker requests."""
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def authenticate(self): self.calls.append(("authenticate", {}))
    async def get_accounts(self):
        self.calls.append(("get_accounts", {}))
        acct = MagicMock(); acct.account_id = "TEST-ACCT"
        return [acct]
    async def get_positions(self, account_id):
        self.calls.append(("get_positions", {"account_id": account_id}))
        pos = MagicMock(); pos.symbol = "BNY"; pos.quantity = 5.0
        return [pos]
    async def get_quotes(self, symbols):
        self.calls.append(("get_quotes", {"symbols": symbols}))
        out = {}
        for s in symbols:
            q = MagicMock(); q.last = 142.50; q.ask = 142.55; q.bid = 142.45
            out[s] = q
        return out
    async def list_orders(self, account_id, status=None):
        self.calls.append(("list_orders", {"status": status}))
        return []
    async def cancel_order(self, oid, acct):
        self.calls.append(("cancel_order", {"oid": oid}))
    async def place_order(self, req, acct):
        self.calls.append(("place_order", {
            "side": req.side, "order_type": req.order_type,
            "qty": req.quantity, "symbol": req.symbol,
        }))
        r = MagicMock(); r.broker_order_id = "BROKER-1"; r.status = "submitted"
        return r
    async def get_order(self, oid, acct):
        r = MagicMock(); r.status = "submitted"; return r


def _mk_signal(symbol="BNY", strategy="BNY_RSI2_Mean_Reversion", direction="SELL"):
    """Synthetic SELL signal in the StrategySignal shape evaluate_strategy uses."""
    from app.services.strategy.models import StrategySignal
    return StrategySignal(
        symbol=symbol, direction=direction, strength=0.8,
        price_at_signal=142.50,
        indicators={"rsi": 75, "reason": "diagnostic"},
        strategy_name=strategy,
    )


def main() -> int:
    print("=" * 70)
    print("SELL -> tight-trail routing diagnostic")
    print("=" * 70)

    broker = _StubBroker()

    # Patch get_broker so the scheduler uses our stub instead of Schwab.
    # `from app.services.brokers.factory import get_broker` at the top of
    # scheduler.py creates a module-level binding; patching factory.get_broker
    # alone is not enough -- patch the scheduler's own resolved symbol.
    from app.services.brokers import factory as broker_factory
    orig_get_broker = broker_factory.get_broker
    broker_factory.get_broker = lambda *a, **k: broker  # type: ignore
    orig_sched_get_broker = sched.get_broker
    sched.get_broker = lambda *a, **k: broker  # type: ignore

    # Patch the ExecutionService's tighten_trail_on_sell to record + short-circuit.
    trail_calls: list[dict] = []

    async def _trail_spy(self, *, symbol, quantity, account_id, signal_price,
                        trail_pct=2.0, source="scheduler",
                        idempotency_suffix="", signal_id=None):
        trail_calls.append({
            "symbol": symbol, "quantity": quantity, "trail_pct": trail_pct,
            "source": source, "signal_id": signal_id,
        })
        return True
    orig_trail = exec_service_module.ExecutionService.tighten_trail_on_sell
    exec_service_module.ExecutionService.tighten_trail_on_sell = _trail_spy  # type: ignore

    # Patch ExecutionService.execute too -- if any SELL reaches THIS path, the
    # fix is broken. We record but don't actually call the broker.
    execute_calls: list[dict] = []

    async def _execute_spy(self, order_req, account_id=None, signal_id=None,
                           estimated_price=None):
        execute_calls.append({
            "side": order_req.side, "order_type": order_req.order_type,
            "qty": order_req.quantity, "symbol": order_req.symbol,
        })
        r = MagicMock(); r.status = "submitted"; return r
    orig_execute = exec_service_module.ExecutionService.execute
    exec_service_module.ExecutionService.execute = _execute_spy  # type: ignore

    # Patch the SELL signal source. The scheduler's scanner-assigned path
    # uses _make_generic_configs_full + evaluate_strategy, so we stub both:
    #  - _make_generic_configs_full returns a fake config matching the assignment
    #  - evaluate_strategy returns our synthetic SELL signal
    from app.services.scanner import scanner_service as scan_svc
    from app.services.strategy import rules as strat_rules
    from app.services.strategy.models import StrategyConfig

    fake_config = StrategyConfig(
        name="BNY_RSI2_Mean_Reversion", type="rsi2_mean_reversion",
        symbol="BNY", params={}, enabled=True,
    )
    orig_make = scan_svc._make_generic_configs_full
    orig_eval = strat_rules.evaluate_strategy

    scan_svc._make_generic_configs_full = lambda sym: [fake_config]  # type: ignore

    def _eval_stub(stype, symbol, prices, params, *, ohlcv=None, position=None):
        return _mk_signal()
    strat_rules.evaluate_strategy = _eval_stub  # type: ignore
    # The scheduler imports evaluate_strategy locally inside the function;
    # patch the resolved binding via the rules module before the function
    # imports it. Already done above.

    # Patch DB session to return a single BNY assignment.
    from app.db import SessionLocal as _orig_session
    from app.models.assignments import SymbolStrategyAssignment

    class _StubAssignmentRow:
        symbol = "BNY"
        # Match BNY's actual prod assignment: scanner-system RSI2.
        system = "scanner"
        strategy_name = "BNY_RSI2_Mean_Reversion"
        max_capital_usd = 1000.0
        max_shares = None
        broker = "default"
        tight_trail_pct = 2.5  # explicit so we can verify it propagates
        enabled = True

    # No need to patch the session -- instead, intercept the SymbolStrategyAssignment
    # query inside _run_cycle. Easiest path: monkeypatch the model's __getattr__
    # is too invasive. Use a session subclass instead.
    class _StubQuery:
        def __init__(self, rows): self.rows = rows
        def filter_by(self, **kw):
            if kw == {"enabled": True}:
                return self
            return _StubQuery([])
        def all(self): return self.rows
        def first(self): return self.rows[0] if self.rows else None
        def filter(self, *a, **kw): return self
        def order_by(self, *a, **kw): return self
        def limit(self, n): return self

    class _StubSession:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def query(self, model):
            if model is SymbolStrategyAssignment:
                return _StubQuery([_StubAssignmentRow()])
            # For Signal / other models the scheduler may also query, return empty.
            return _StubQuery([])
        def add(self, *a, **kw): pass
        def commit(self): pass
        def refresh(self, obj): pass
        def close(self): pass

    sched.SessionLocal = lambda: _StubSession()  # type: ignore

    # Patch market_hours so it doesn't reject us.
    sched.is_market_hours = lambda *a, **kw: True  # type: ignore
    # Patch get_ohlcv to return enough bars for the strategy to be queryable
    # (we already stubbed run_perplexity_signal so this only needs to not blow up).
    import pandas as pd
    idx = pd.bdate_range("2024-01-01", periods=400)
    fake_bars = pd.DataFrame({
        "Open": 140.0, "High": 143.0, "Low": 139.0,
        "Close": 142.5, "Volume": 1_000_000,
    }, index=idx)
    from app.services.market_data import provider as mdp
    orig_get_ohlcv = mdp.get_ohlcv
    mdp.get_ohlcv = lambda *a, **k: fake_bars  # type: ignore

    # Now run a cycle. dry_run=False so the SELL routing actually executes;
    # all broker I/O is stubbed.
    print("Running scheduler._run_cycle(force=True, dry_run=False)...")
    import logging
    sched_logger = logging.getLogger("app.services.strategy.scheduler")
    sched_logger.setLevel(logging.DEBUG)
    log_capture: list[str] = []
    class _Handler(logging.Handler):
        def emit(self, record):
            try:
                log_capture.append(self.format(record))
            except Exception:
                pass
    h = _Handler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    sched_logger.addHandler(h)
    try:
        sched._run_cycle(force=True, dry_run=False)
    except Exception as exc:
        print(f"[ERROR] _run_cycle raised: {exc}")
        return 2
    finally:
        sched_logger.removeHandler(h)
        print("--- scheduler log (last 30) ---")
        for line in log_capture[-30:]:
            print("  " + line.encode("ascii", "replace").decode("ascii"))
        print("-------------------------------")
        # Restore everything we monkeypatched.
        broker_factory.get_broker = orig_get_broker  # type: ignore
        sched.get_broker = orig_sched_get_broker  # type: ignore
        exec_service_module.ExecutionService.tighten_trail_on_sell = orig_trail  # type: ignore
        exec_service_module.ExecutionService.execute = orig_execute  # type: ignore
        scan_svc._make_generic_configs_full = orig_make  # type: ignore
        strat_rules.evaluate_strategy = orig_eval  # type: ignore
        mdp.get_ohlcv = orig_get_ohlcv  # type: ignore
        sched.SessionLocal = _orig_session  # type: ignore

    print()
    print(f"tighten_trail_on_sell calls: {len(trail_calls)}")
    for c in trail_calls:
        print(f"  -> {c}")
    print(f"svc.execute calls: {len(execute_calls)}")
    for c in execute_calls:
        print(f"  -> {c}")
    print()

    # Verdict
    if not trail_calls:
        print("FAIL: FAIL: SELL signal did NOT call tighten_trail_on_sell.")
        return 1
    sell_executes = [c for c in execute_calls if c["side"] == "SELL"]
    if sell_executes:
        print("FAIL: FAIL: A SELL signal ALSO reached svc.execute(...) -- "
              "would have placed a MARKET sell:")
        for c in sell_executes:
            print(f"   {c}")
        return 1

    # Check the trail % matches the assignment's tight_trail_pct
    if not any(abs(c["trail_pct"] - 2.5) < 0.001 for c in trail_calls):
        print(f"WARN: WARNING: assignment had tight_trail_pct=2.5 but trail calls "
              f"used: {[c['trail_pct'] for c in trail_calls]}")
        return 1

    print("PASS: PASS: SELL routed through tighten_trail_on_sell with the "
          "assignment's tight_trail_pct (2.5%). No market-sell escape hatch hit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
