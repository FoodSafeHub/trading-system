"""Phase 1: unified compare-all config path is opt-in; default stays legacy."""
import inspect

from app.api.routes.backtest import backtest_custom_compare_all
from app.services.scanner.scanner_service import (
    _make_unified_configs, _make_generic_configs_full, _make_generic_configs,
)


def test_compare_all_default_is_legacy():
    # Route default must keep the legacy strategy set so existing output is unchanged.
    sig = inspect.signature(backtest_custom_compare_all)
    assert sig.parameters["strategy_set"].default == "legacy"


def test_legacy_set_still_seven_configs():
    cfgs = _make_generic_configs_full("TEST")
    assert len(cfgs) == 7
    # Live scanner factory unchanged: still the 5 regime-aware types.
    assert len(_make_generic_configs("TEST")) == 5


def test_unified_set_is_six_new_types_with_exit_policy():
    cfgs = _make_unified_configs("TEST")
    assert [c.type for c in cfgs] == [
        "rsi2_reversion", "trend_pullback", "squeeze_breakout",
        "momentum_breakout", "panic_reversal", "trend_follow",
    ]
    assert all("exit_policy" in c.params for c in cfgs)


def test_unified_inherits_predecessor_calibration(monkeypatch):
    import app.services.backtest.scanner_profiles as sp
    from app.services.backtest.scanner_profiles import ScannerParamProfile
    fake = {("pullback_ema50", "TEST"): ScannerParamProfile(
        symbol="TEST", strategy_type="pullback_ema50", calibrated_at="2026-01-01",
        n_trades=20, n_wins=12, win_rate_pct=60.0, param_overrides={"rsi_max": 50})}
    monkeypatch.setattr(sp, "load_profile", lambda st, sym: fake.get((st, sym.upper())))
    cfgs = {c.type: c for c in _make_unified_configs("TEST")}
    # trend_pullback should inherit pullback_ema50's calibrated rsi_max via the alias map.
    assert cfgs["trend_pullback"].params["rsi_max"] == 50
