"""Unit tests for the strategy alias map + scanner_profiles fallback (Phase 0)."""
import app.services.backtest.scanner_profiles as sp
from app.services.backtest.scanner_profiles import ScannerParamProfile, get_param_overrides
from app.services.strategy.strategy_aliases import resolve_alias, canonical


def test_old_types_resolve_to_self_only():
    for old in ("rsi2_mean_reversion", "pullback_ema50", "vix_spike_reversal",
                "bb_squeeze_breakout", "ema_macd_crossover"):
        assert resolve_alias(old) == [old]


def test_new_types_resolve_self_first_then_predecessors():
    assert resolve_alias("trend_pullback") == ["trend_pullback", "pullback_ema50", "fib_pullback"]
    assert resolve_alias("rsi2_reversion") == ["rsi2_reversion", "rsi2_mean_reversion"]
    assert resolve_alias("rs_rotation") == ["rs_rotation"]


def test_canonical_reverse_maps():
    assert canonical("pullback_ema50") == "trend_pullback"
    assert canonical("rsi2_mean_reversion") == "rsi2_reversion"
    assert canonical("trend_pullback") == "trend_pullback"   # already canonical
    assert canonical("unknown_x") == "unknown_x"             # pass through


def test_get_param_overrides_self_first_and_alias_fallback(monkeypatch):
    fake = {
        ("pullback_ema50", "FOO"): ScannerParamProfile(
            symbol="FOO", strategy_type="pullback_ema50", calibrated_at="2026-01-01",
            n_trades=20, n_wins=12, win_rate_pct=60.0,
            param_overrides={"rsi_min": 40, "price_ema_proximity_pct": 2.0}),
    }
    monkeypatch.setattr(sp, "load_profile", lambda st, sym: fake.get((st, sym.upper())))

    # Live type with nothing saved -> empty (unchanged behaviour).
    assert get_param_overrides("rsi2_mean_reversion", "FOO") == {}
    # Live type reads its own key only.
    assert get_param_overrides("pullback_ema50", "FOO") == {"rsi_min": 40, "price_ema_proximity_pct": 2.0}
    # New type inherits predecessor calibration via the alias fallback.
    assert get_param_overrides("trend_pullback", "FOO") == {"rsi_min": 40, "price_ema_proximity_pct": 2.0}
    # New type with no predecessor profile -> empty.
    assert get_param_overrides("squeeze_breakout", "FOO") == {}
