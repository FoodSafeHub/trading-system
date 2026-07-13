"""Tape-health gate (knife-entry veto) unit tests."""
import numpy as np
import pandas as pd
import pytest

from app.services.strategy.tape_health import check_tape_health


def _ohlcv(closes):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Open": closes.shift(1).fillna(closes.iloc[0]),
        "High": closes * 1.005,
        "Low": closes * 0.995,
        "Close": closes,
        "Volume": 1_000_000,
    })


def _steady(n=60, px=100.0):
    """Flat-ish healthy tape."""
    rng = np.random.default_rng(7)
    return list(px + np.cumsum(rng.normal(0.02, 0.15, n)))


def test_healthy_tape_passes():
    th = check_tape_health("TEST", "TEST_Pullback_EMA50", ohlcv=_ohlcv(_steady()))
    assert th.ok
    assert th.reasons == []


def test_fast_crash_blocked():
    # 15% straight-line drop over the last 5 sessions.
    closes = _steady(55) + [100, 97, 94, 91, 88, 85]
    th = check_tape_health("TEST", "TEST_Fib_Pullback", ohlcv=_ohlcv(closes))
    assert not th.ok
    assert any("5d return" in r for r in th.reasons)


def test_red_streak_alone_no_longer_blocks():
    # 4 mildly red closes on an otherwise healthy tape (the AAPL/KO drift) —
    # the 2026-07-02 study showed streak-alone blocked winners (+1.8% avg).
    closes = _steady(56, 100) + [100.0, 99.7, 99.4, 99.1, 98.9]
    th = check_tape_health("TEST", "TEST_Pullback_EMA50", ohlcv=_ohlcv(closes))
    assert th.ok
    assert th.metrics["red_streak"] >= 4  # still reported, just not gating


def test_rally_crash_roundtrip_blocked():
    # The AMAT 2026-07-06 blind spot: a parabolic run-up, then a spike-and-crash
    # INSIDE the 5-session window. Close-to-close 5d is mild (−4.8%: the window's
    # left edge was already low) but price sits −17.5% under the 723 close printed
    # mid-window. The live gate PASSED this entry; it gapped −9% overnight.
    runup = list(np.linspace(400, 630, 50))
    closes = runup + [626.84, 694.64, 723.00, 650.91, 603.04, 596.62]
    th = check_tape_health("AMAT", "Legacy_AMAT_Fib_Pullback", ohlcv=_ohlcv(closes))
    assert th.metrics["ret_5d_pct"] > -6          # old rule alone would pass it
    assert th.metrics["off_5d_high_pct"] < -15
    assert not th.ok
    assert any("off 5d high" in r for r in th.reasons)


def test_rally_crash_panic_strategy_exempt():
    # RSI2 / VIX-spike panic buyers keep their velocity exemption for the new
    # off-5d-high rule too — buying the fast dip is their edge.
    runup = list(np.linspace(400, 630, 50))
    closes = runup + [626.84, 694.64, 723.00, 650.91, 603.04, 596.62]
    th = check_tape_health("AMAT", "AMAT_RSI2_Mean_Reversion", ohlcv=_ohlcv(closes))
    assert th.ok


def test_structural_rules_must_trip_together():
    # Deep off the 20d high but still near EMA20 (long slow slide that has
    # already based out) — one structural rule alone must NOT block.
    closes = _steady(30, 100) + list(pd.Series(
        [100, 97, 94, 91, 88, 86, 85, 85.5, 85.2, 85.4, 85.1, 85.3,
         85.0, 85.2, 85.4, 85.3, 85.5, 85.4, 85.6, 85.5]))
    th = check_tape_health("TEST", "TEST_Pullback_EMA50", ohlcv=_ohlcv(closes))
    # off 20d high is deep, but price sits ON its EMA20 → allowed.
    assert th.metrics["off_20d_high_pct"] < -12
    assert th.metrics["vs_ema20_pct"] > -5
    assert th.ok


def test_deep_drawdown_blocked_even_on_slow_decline():
    # Grinds down slowly (no 5d velocity trip) but sits 15% off the 20d high.
    closes = _steady(40, 100) + [100, 99.2, 99.5, 98.6, 98.9, 98.0, 97.6, 96.9,
                                 96.2, 95.6, 96.0, 94.8, 94.1, 93.3, 92.8, 92.2,
                                 91.5, 90.8, 90.2, 85.0]
    th = check_tape_health("TEST", "TEST_Pullback_EMA50", ohlcv=_ohlcv(closes))
    assert not th.ok
    assert any("20d high" in r for r in th.reasons)


def test_panic_strategy_exempt_from_velocity_but_not_structure():
    # Sharp 4-day dip (~-8% in 5 sessions), still near the 20d high: RSI2 may
    # buy the panic; a trend-following dip buyer may not.
    closes = _steady(55, 100) + [101, 99.0, 97.0, 94.5, 92.0]
    named = check_tape_health("TEST", "TEST_RSI2_Mean_Reversion", ohlcv=_ohlcv(closes))
    plain = check_tape_health("TEST", "TEST_Pullback_EMA50", ohlcv=_ohlcv(closes))
    assert named.ok            # panic buyer allowed into the fast dip
    assert not plain.ok        # trend-following dip buyer is not

    # But a BROKEN tape (deep off the high) blocks even the panic strategy.
    broken = _steady(40, 100) + list(np.linspace(100, 80, 20))
    th = check_tape_health("TEST", "TEST_RSI2_Mean_Reversion", ohlcv=_ohlcv(broken))
    assert not th.ok


def test_insufficient_history_fails_open():
    th = check_tape_health("TEST", "X", ohlcv=_ohlcv([100.0] * 10))
    assert th.ok


def test_garbage_input_fails_open():
    th = check_tape_health("TEST", "X", ohlcv=pd.DataFrame({"bogus": [1, 2, 3]}))
    assert th.ok


def test_annotate_backtest_trades_buckets_by_verdict():
    from app.services.strategy.tape_health import annotate_backtest_trades
    # 60 healthy bars, then a crash into bar 65 — a BUY there is a knife.
    closes = _steady(60, 100) + [100, 95, 90, 86, 82, 84, 86, 88, 90, 92]
    df = _ohlcv(closes)
    dates = [str(i) for i in range(len(closes))]
    df.index = pd.to_datetime("2026-01-01") + pd.to_timedelta(range(len(closes)), unit="D")
    d = lambda i: str(df.index[i])[:10]

    trades = [
        {"date": d(30), "side": "BUY",  "value": 1000.0},   # healthy tape
        {"date": d(40), "side": "SELL", "value": 1100.0},   # +10%
        {"date": d(64), "side": "BUY",  "value": 1000.0},   # mid-crash knife
        {"date": d(69), "side": "SELL", "value": 950.0},    # -5%
    ]
    summary = annotate_backtest_trades("TEST", trades, df, "TEST_Fib_Pullback")

    assert trades[0]["tape_gate"] == "pass"
    assert trades[2]["tape_gate"] == "block"
    assert trades[2]["tape_gate_reason"]
    assert summary["blocked_buys"] == 1
    assert summary["pass"]["round_trips"] == 1
    assert summary["block"]["round_trips"] == 1
    assert summary["pass"]["avg_return_pct"] > 0 > summary["block"]["avg_return_pct"]


def test_annotate_fails_open_on_garbage():
    from app.services.strategy.tape_health import annotate_backtest_trades
    assert annotate_backtest_trades("TEST", [{"bad": 1}], None) == {}


def test_scheduler_wiring_present():
    """Structural: both scheduler BUY paths must consult the gate + cooldown
    BEFORE spending cash, and the module-level helpers must exist."""
    import ast, inspect
    from app.services.strategy import scheduler

    src = inspect.getsource(scheduler)
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_tape_gate_blocks" in names
    assert "_in_reentry_cooldown" in names
    # _time_stop_pass was removed 2026-07-06 (user decision: losing swings get
    # time to recover) — assert it stays gone so it isn't reintroduced silently.
    assert "_time_stop_pass" not in names
    # The assigned path is the ONLY scheduler BUY path since 2026-07-06
    # (consensus BUYs are review-only notifications, no orders) — the gate and
    # cooldown must still guard it. The tape gate keeps a 2nd call site in the
    # consensus branch for reviewer context (verdict text on the proposal).
    assert src.count("_tape_gate_blocks(") >= 3   # def + assigned + consensus-info
    assert src.count("_in_reentry_cooldown(") >= 2  # def + assigned BUY path
    assert "notify_consensus_proposal" in src     # consensus BUYs → review feed
    # Vetoes surface as notifications, not silent skips.
    assert 'reason="tape_health"' in src
    assert 'reason="reentry_cooldown"' in src
    assert 'reason="time_stop"' not in src
