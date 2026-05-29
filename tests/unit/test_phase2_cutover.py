"""Phase 2 controlled calibration cutover.

Covers: new names calibratable, old names still calibratable, the calibration
route's config lookup finds unified configs, coordinate-descent tolerates the
exit_policy dict, alias-based profile fallback, compare-all legacy vs unified on
the SAME fixture, and non-destructive migration.
"""
import app.services.backtest.scanner_profiles as sp
from app.services.backtest.scanner_profiles import (
    grid_for, coordinate_descent_search, migrate_aliased_profiles,
    get_param_overrides, ScannerParamProfile,
)
from app.api.routes.backtest import _SCANNER_CALIBRATABLE, backtest_custom_compare_all
from app.services.scanner.scanner_service import _make_generic_configs, _make_unified_configs

NEW = ["rsi2_reversion", "trend_pullback", "squeeze_breakout",
       "momentum_breakout", "panic_reversal", "trend_follow"]
OLD = ["rsi2_mean_reversion", "ema_macd_crossover", "bb_squeeze_breakout",
       "pullback_ema50", "vix_spike_reversal"]


# ── 1. Calibratable surface ─────────────────────────────────────────────────────

def test_new_names_are_calibratable():
    for t in NEW:
        assert t in _SCANNER_CALIBRATABLE
        assert grid_for(t), f"{t} needs a tunable grid"


def test_old_names_still_calibratable():
    for t in OLD:
        assert t in _SCANNER_CALIBRATABLE
        assert grid_for(t)


def test_calibration_route_can_locate_unified_config():
    # Mirrors the route's lookup: union of generic + unified factories.
    all_cfgs = list(_make_generic_configs("AAPL")) + list(_make_unified_configs("AAPL"))
    types = {c.type for c in all_cfgs}
    for t in NEW + OLD:
        assert t in types


# ── 2. Coordinate descent tolerates non-scalar (exit_policy) params ─────────────

def test_coordinate_descent_handles_exit_policy_dict():
    base = {"atr_proximity_mult": 1.2, "rsi_min": 35, "rsi_max": 55,
            "wick_ratio_min": 0.4, "exit_extension_pct": 3.0,
            "exit_policy": {"trail": "chandelier", "atr_mult": 3.0}}
    # Reward a higher wick_ratio_min so a distinct override is produced.
    overrides, score, m, n = coordinate_descent_search(
        "trend_pullback", base, lambda p: (p.get("wick_ratio_min", 0) * 100.0, {}), rounds=2)
    assert "exit_policy" not in overrides       # never tune the policy dict
    assert n > 1


# ── 3. Alias-based profile fallback (existing resolution unchanged) ─────────────

def test_alias_fallback_and_existing_resolution(monkeypatch):
    fake = {("pullback_ema50", "AAPL"): ScannerParamProfile(
        symbol="AAPL", strategy_type="pullback_ema50", calibrated_at="2026-01-01",
        n_trades=20, n_wins=12, win_rate_pct=60.0, param_overrides={"rsi_max": 50})}
    monkeypatch.setattr(sp, "load_profile", lambda st, sym: fake.get((st, sym.upper())))
    # New name inherits predecessor calibration.
    assert get_param_overrides("trend_pullback", "AAPL") == {"rsi_max": 50}
    # Old name resolves exactly as before (self key only).
    assert get_param_overrides("pullback_ema50", "AAPL") == {"rsi_max": 50}
    # Unrelated old name with no profile -> empty (unchanged).
    assert get_param_overrides("rsi2_mean_reversion", "AAPL") == {}


# ── 4. Migration is non-destructive and reversible ──────────────────────────────

def test_migration_dry_run_reports_without_writing(monkeypatch):
    store = {"pullback_ema50:AAPL": {
        "symbol": "AAPL", "strategy_type": "pullback_ema50", "calibrated_at": "2026-01-01",
        "n_trades": 20, "n_wins": 12, "win_rate_pct": 60.0, "param_overrides": {"rsi_max": 50},
        "survival_rate_pct": 80.0, "improved_by": ["expectancy"], "notes": ""}}
    saved = {}
    monkeypatch.setattr(sp, "_load_all", lambda: dict(store))
    monkeypatch.setattr(sp, "_save_all", lambda d: saved.update({"written": d}))
    rep = migrate_aliased_profiles(dry_run=True)
    assert any(r["new_type"] == "trend_pullback" and r["source"] == "pullback_ema50" for r in rep)
    assert "written" not in saved   # dry run never writes


def test_is_genuine_calibration_classifier():
    from app.services.backtest.scanner_profiles import _is_genuine_calibration
    # trades > 0 -> genuine
    assert _is_genuine_calibration({"n_trades": 12, "param_overrides": {"rsi_max": 50}}) is True
    # 0 trades but an entry override -> genuine
    assert _is_genuine_calibration({"n_trades": 0, "param_overrides": {"rsi_max": 50}}) is True
    # 0 trades, only trail scaffolding -> NOT genuine (live trail config)
    trail = {"trail_enabled": True, "trail_trigger_pct": 3.0, "atr_trail_mult": 3.0, "atr_trail_period": 22}
    assert _is_genuine_calibration({"n_trades": 0, "param_overrides": trail}) is False


def test_migration_skips_trail_only_source(monkeypatch):
    trail = {"trail_enabled": True, "trail_trigger_pct": 3.0, "atr_trail_mult": 3.0, "atr_trail_period": 22}
    store = {
        # genuine calibration -> should copy
        "rsi2_mean_reversion:NVDA": {"symbol": "NVDA", "strategy_type": "rsi2_mean_reversion",
            "calibrated_at": "2026-05-27", "n_trades": 31, "n_wins": 28, "win_rate_pct": 90.3,
            "param_overrides": {"rsi_entry_threshold": 8}, "survival_rate_pct": 96.9,
            "improved_by": [], "notes": ""},
        # 0-trade trail-only live scaffolding -> must skip
        "rsi2_mean_reversion:TOST": {"symbol": "TOST", "strategy_type": "rsi2_mean_reversion",
            "calibrated_at": "2026-05-28", "n_trades": 0, "n_wins": 0, "win_rate_pct": 0.0,
            "param_overrides": dict(trail), "survival_rate_pct": 0.0, "improved_by": [], "notes": ""},
    }
    written = {}
    monkeypatch.setattr(sp, "_load_all", lambda: dict(store))
    monkeypatch.setattr(sp, "_save_all", lambda d: written.update({"data": d}))
    rep = migrate_aliased_profiles(dry_run=True)
    copies = {(r["source"], r["symbol"]) for r in rep if r["action"] == "copy"}
    skips = {(r["source"], r["symbol"]) for r in rep if r["action"].startswith("skip")}
    assert ("rsi2_mean_reversion", "NVDA") in copies
    assert ("rsi2_mean_reversion", "TOST") in skips
    assert "data" not in written  # dry run writes nothing


def test_migration_write_keeps_old_profile(monkeypatch):
    store = {"pullback_ema50:AAPL": {
        "symbol": "AAPL", "strategy_type": "pullback_ema50", "calibrated_at": "2026-01-01",
        "n_trades": 20, "n_wins": 12, "win_rate_pct": 60.0, "param_overrides": {"rsi_max": 50},
        "survival_rate_pct": 80.0, "improved_by": ["expectancy"], "notes": ""}}
    captured = {}
    monkeypatch.setattr(sp, "_load_all", lambda: dict(store))
    monkeypatch.setattr(sp, "_save_all", lambda d: captured.update(d))
    migrate_aliased_profiles(dry_run=False)
    assert "pullback_ema50:AAPL" in captured            # OLD preserved
    assert "trend_pullback:AAPL" in captured            # NEW materialized
    assert captured["trend_pullback:AAPL"]["param_overrides"] == {"rsi_max": 50}


# ── 5. Compare-all: legacy default vs unified opt-in on the SAME fixture ─────────

def test_compare_all_legacy_vs_unified(monkeypatch):
    from tests.golden.synth import make_ohlcv
    import app.services.backtest.engine as engine
    monkeypatch.setattr(engine, "get_ohlcv", lambda symbol, period="1y", **k: make_ohlcv())

    legacy = backtest_custom_compare_all("TEST", period="5y", strategy_set="legacy")
    unified = backtest_custom_compare_all("TEST", period="5y", strategy_set="unified")

    assert len(legacy) == 7 and len(unified) == 6
    # Different strategy sets -> different row names.
    assert {r["strategy_name"] for r in legacy} != {r["strategy_name"] for r in unified}
    # Both well-formed: ranking-relevant metric keys present on every row.
    for rows in (legacy, unified):
        for r in rows:
            assert {"expectancy_pct", "profit_factor", "total_return_pct",
                    "max_drawdown_pct", "win_rate_pct"} <= set(r)
    # Legacy path is deterministic (cutover safety).
    legacy2 = backtest_custom_compare_all("TEST", period="5y", strategy_set="legacy")
    assert [r["strategy_name"] for r in legacy] == [r["strategy_name"] for r in legacy2]
    assert [r["total_return_pct"] for r in legacy] == [r["total_return_pct"] for r in legacy2]
