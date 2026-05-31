"""
Tests for risk_templates — ExitPlan factories and risk utility functions.

Covers:
- All 6 exit plan factories (US + NSE buckets)
- Trigger price arithmetic
- Scale level pct_to_close sums
- session_risk_adjustment threshold boundaries
- can_enter concurrency and sector caps
- get_symbol_bucket classification
"""
from __future__ import annotations

import pytest

from app.services.strategy.daytrading.risk_templates import (
    ExitPlan,
    ScaleLevel,
    can_enter,
    ema_exit_plan,
    gap_fade_exit_plan,
    get_symbol_bucket,
    nr_squeeze_exit_plan,
    orb_exit_plan,
    session_risk_adjustment,
    supertrend_exit_plan,
    vwap_exit_plan,
    DAILY_CAPS,
    RISK_PER_TRADE_PCT,
)


# ── get_symbol_bucket ─────────────────────────────────────────────────────────

class TestGetSymbolBucket:
    def test_known_etfs(self):
        assert get_symbol_bucket("SPY", "US") == "US_ETF"
        assert get_symbol_bucket("QQQ", "US") == "US_ETF"
        assert get_symbol_bucket("IWM", "US") == "US_ETF"

    def test_known_large_caps(self):
        assert get_symbol_bucket("AAPL", "US") == "US_LARGE_CAP"
        assert get_symbol_bucket("MSFT", "US") == "US_LARGE_CAP"
        assert get_symbol_bucket("NVDA", "US") == "US_LARGE_CAP"

    def test_unknown_us_falls_back_to_mid_small(self):
        assert get_symbol_bucket("XYZW", "US") == "US_MID_SMALL"

    def test_nse_large_caps(self):
        assert get_symbol_bucket("RELIANCE", "NSE") == "NSE_LARGE_CAP"
        assert get_symbol_bucket("BHARTIARTL", "NSE") == "NSE_LARGE_CAP"
        assert get_symbol_bucket("INFY", "NSE") == "NSE_LARGE_CAP"

    def test_nse_unknown_falls_back_to_mid_cap(self):
        assert get_symbol_bucket("SOMESTOCK", "NSE") == "NSE_MID_CAP"

    def test_strip_suffix(self):
        # "RELIANCE.NS" should resolve same as "RELIANCE"
        assert get_symbol_bucket("RELIANCE.NS", "NSE") == "NSE_LARGE_CAP"


# ── ORB exit plan ─────────────────────────────────────────────────────────────

class TestOrbExitPlan:
    def _plan(self, bucket, orb_range=2.0, entry=450.0, stop=448.5):
        return orb_exit_plan(bucket, entry=entry, stop=stop, orb_range=orb_range)

    def test_us_etf_has_two_scales(self):
        ep = self._plan("US_ETF")
        assert len(ep.scale_levels) == 2

    def test_us_etf_scale1_trigger_price(self):
        ep = self._plan("US_ETF", orb_range=2.0, entry=450.0)
        # scale1 = entry + ORB×1.2 = 450 + 2.4 = 452.4
        assert ep.scale_levels[0].trigger_price == pytest.approx(452.4, abs=0.01)

    def test_us_etf_scale2_trigger_price(self):
        ep = self._plan("US_ETF", orb_range=2.0, entry=450.0)
        # scale2 = entry + ORB×2.0 = 454.0
        assert ep.scale_levels[1].trigger_price == pytest.approx(454.0, abs=0.01)

    def test_us_etf_pcts_sum_to_at_most_100(self):
        ep = self._plan("US_ETF")
        total = sum(s.pct_to_close for s in ep.scale_levels)
        assert total <= 1.001   # sum of scale tiers + runner = 1.0

    def test_us_etf_has_ema9_trail(self):
        ep = self._plan("US_ETF")
        assert ep.trail_type == "ema9_5m"

    def test_nse_large_cap_no_runner(self):
        ep = self._plan("NSE_LARGE_CAP")
        assert ep.trail_type == "none"

    def test_nse_large_cap_two_scales(self):
        ep = self._plan("NSE_LARGE_CAP")
        assert len(ep.scale_levels) == 2

    def test_nse_large_cap_earlier_time_exit(self):
        ep = self._plan("NSE_LARGE_CAP")
        assert ep.hard_exit_time_ist == "11:45"

    def test_us_mid_small_wider_targets(self):
        ep_etf = self._plan("US_ETF", orb_range=2.0, entry=450.0)
        ep_mid = self._plan("US_MID_SMALL", orb_range=2.0, entry=450.0)
        # Mid-small scale2 = ORB×2.5 > ETF scale2 = ORB×2.0
        assert ep_mid.scale_levels[1].trigger_price > ep_etf.scale_levels[1].trigger_price

    def test_stop_price_stored(self):
        ep = self._plan("US_ETF", entry=450.0, stop=448.5)
        assert ep.initial_stop_price == pytest.approx(448.5)

    def test_trigger_r_positive(self):
        ep = self._plan("US_ETF")
        for s in ep.scale_levels:
            assert s.trigger_r > 0


# ── VWAP exit plan ────────────────────────────────────────────────────────────

class TestVwapExitPlan:
    def _long(self, bucket, entry=450.0, stop=449.2, vwap=451.0, atr=0.8):
        return vwap_exit_plan(bucket, entry=entry, stop=stop, vwap=vwap,
                              atr=atr, direction="BUY")

    def _short(self, bucket, entry=451.0, stop=451.8, vwap=450.0, atr=0.8):
        return vwap_exit_plan(bucket, entry=entry, stop=stop, vwap=vwap,
                              atr=atr, direction="SELL")

    def test_us_etf_long_scale1_at_vwap(self):
        ep = self._long("US_ETF", vwap=451.0)
        # scale1 trigger price = VWAP = 451.0
        assert ep.scale_levels[0].trigger_price == pytest.approx(451.0)

    def test_us_large_cap_has_small_runner(self):
        ep = self._long("US_LARGE_CAP")
        assert ep.trail_type == "ema9_5m"

    def test_nse_no_runner(self):
        ep = self._long("NSE_LARGE_CAP")
        assert ep.trail_type == "none"

    def test_nse_two_scales(self):
        ep = self._long("NSE_LARGE_CAP")
        assert len(ep.scale_levels) == 2

    def test_short_scale1_at_vwap_below(self):
        # For SELL, vwap is the target level (price moves DOWN to vwap)
        ep = self._short("US_ETF", entry=451.0, vwap=450.0)
        assert ep.scale_levels[0].trigger_price == pytest.approx(450.0)


# ── EMA exit plan ─────────────────────────────────────────────────────────────

class TestEmaExitPlan:
    def test_us_has_three_tranches(self):
        ep = ema_exit_plan("US_LARGE_CAP", entry=450.0, stop=449.1,
                           atr=0.9, direction="BUY")
        assert len(ep.scale_levels) == 2   # scale1 + scale2; runner implicit

    def test_us_stop_type_is_bar_low(self):
        ep = ema_exit_plan("US_ETF", entry=450.0, stop=449.4, atr=0.6, direction="BUY")
        assert ep.stop_type == "bar_low_atr"

    def test_nse_no_runner(self):
        ep = ema_exit_plan("NSE_LARGE_CAP", entry=2480.0, stop=2477.65,
                           atr=2.35, direction="BUY")
        assert ep.trail_type == "none"

    def test_nse_earlier_time_exit(self):
        ep = ema_exit_plan("NSE_LARGE_CAP", entry=2480.0, stop=2477.65,
                           atr=2.35, direction="BUY")
        assert ep.hard_exit_time_ist == "11:45"

    def test_scale1_trigger_price_1_5_atr(self):
        # US: scale1 = entry + 1.5×ATR
        atr = 1.0
        ep = ema_exit_plan("US_LARGE_CAP", entry=100.0, stop=98.7,
                           atr=atr, direction="BUY")
        assert ep.scale_levels[0].trigger_price == pytest.approx(101.5)


# ── Gap fade exit plan ────────────────────────────────────────────────────────

class TestGapFadeExitPlan:
    def test_us_scale1_at_60pct_fill(self):
        # Gap from 450 (prior close) to 455 (gap up entry)
        # 60% fill = 455 - 0.6×(455-450) = 455 - 3 = 452
        ep = gap_fade_exit_plan(
            "US_LARGE_CAP", entry=455.0, stop=455.78,
            gap_close=450.0, gap_pct=1.1, direction="SELL"
        )
        assert ep.scale_levels[0].trigger_price == pytest.approx(452.0, abs=0.01)

    def test_us_has_atr_trail_runner(self):
        ep = gap_fade_exit_plan(
            "US_LARGE_CAP", entry=455.0, stop=455.78,
            gap_close=450.0, gap_pct=1.1, direction="SELL"
        )
        assert ep.trail_type == "atr_fixed"

    def test_nse_scale1_at_50pct_fill(self):
        ep = gap_fade_exit_plan(
            "NSE_LARGE_CAP", entry=2500.0, stop=2509.0,
            gap_close=2475.0, gap_pct=1.0, direction="SELL"
        )
        # 50% fill = 2500 - 0.5×25 = 2487.5
        assert ep.scale_levels[0].trigger_price == pytest.approx(2487.5, abs=0.1)

    def test_nse_no_runner(self):
        ep = gap_fade_exit_plan(
            "NSE_LARGE_CAP", entry=2500.0, stop=2509.0,
            gap_close=2475.0, gap_pct=1.0, direction="SELL"
        )
        assert ep.trail_type == "none"

    def test_us_time_exit_before_midday(self):
        ep = gap_fade_exit_plan(
            "US_ETF", entry=455.0, stop=455.78,
            gap_close=450.0, gap_pct=1.1, direction="SELL"
        )
        h = int(ep.hard_exit_time_et.split(":")[0])
        assert h <= 12   # gap fades close early


# ── Supertrend exit plan ──────────────────────────────────────────────────────

class TestSupertrendExitPlan:
    def test_us_has_st_trail(self):
        ep = supertrend_exit_plan("US_LARGE_CAP", entry=450.0,
                                  stop=448.8, direction="BUY")
        assert ep.trail_type == "supertrend_5m"

    def test_us_scale1_at_2r(self):
        entry, stop = 450.0, 448.8
        risk = entry - stop
        ep = supertrend_exit_plan("US_LARGE_CAP", entry=entry,
                                  stop=stop, direction="BUY")
        assert ep.scale_levels[0].trigger_r == pytest.approx(2.0)
        assert ep.scale_levels[0].trigger_price == pytest.approx(entry + 2.0 * risk, abs=0.01)

    def test_us_scale1_exits_30pct(self):
        ep = supertrend_exit_plan("US_ETF", entry=450.0, stop=448.8, direction="BUY")
        assert ep.scale_levels[0].pct_to_close == pytest.approx(0.30)

    def test_nse_has_st_trail(self):
        ep = supertrend_exit_plan("NSE_LARGE_CAP", entry=2480.0,
                                  stop=2475.2, direction="BUY")
        assert ep.trail_type == "supertrend_5m"

    def test_nse_earlier_scale1(self):
        ep_us  = supertrend_exit_plan("US_LARGE_CAP", entry=450.0, stop=448.0, direction="BUY")
        ep_nse = supertrend_exit_plan("NSE_LARGE_CAP", entry=450.0, stop=448.0, direction="BUY")
        # NSE scale1_r = 1.5 < US scale1_r = 2.0
        assert ep_nse.scale_levels[0].trigger_r < ep_us.scale_levels[0].trigger_r

    def test_short_scale1_trigger_below_entry(self):
        entry, stop = 450.0, 451.2
        ep = supertrend_exit_plan("US_LARGE_CAP", entry=entry,
                                  stop=stop, direction="SELL")
        assert ep.scale_levels[0].trigger_price < entry


# ── NR/Squeeze exit plan ──────────────────────────────────────────────────────

class TestNRSqueezeExitPlan:
    def test_us_etf_only_one_scale_no_nse(self):
        ep = nr_squeeze_exit_plan("US_ETF", entry=450.0, stop=448.65, direction="BUY")
        assert len(ep.scale_levels) == 1
        assert ep.trail_type == "ema9_5m"

    def test_us_large_two_scales_structure_trail(self):
        ep = nr_squeeze_exit_plan("US_LARGE_CAP", entry=450.0, stop=448.3, direction="BUY")
        assert len(ep.scale_levels) == 2
        assert ep.trail_type == "prior_bar_low_5m"

    def test_us_scale2_at_3_5r(self):
        entry, stop = 450.0, 448.3
        ep = nr_squeeze_exit_plan("US_LARGE_CAP", entry=entry, stop=stop, direction="BUY")
        assert ep.scale_levels[1].trigger_r == pytest.approx(3.5)

    def test_nse_two_fixed_tiers_no_runner(self):
        ep = nr_squeeze_exit_plan("NSE_LARGE_CAP", entry=2480.0,
                                  stop=2475.84, direction="BUY")
        assert len(ep.scale_levels) == 2
        assert ep.trail_type == "none"

    def test_stop_type_is_nr_bar(self):
        ep = nr_squeeze_exit_plan("US_LARGE_CAP", entry=450.0, stop=448.3, direction="BUY")
        assert ep.stop_type == "nr_bar_low"


# ── session_risk_adjustment ───────────────────────────────────────────────────

class TestSessionRiskAdjustment:
    def test_normal_no_losses(self):
        r = session_risk_adjustment(current_pnl_r=0.0, consec_losses=0)
        assert r["status"] == "NORMAL"
        assert r["size_mult"] == 1.0

    def test_soft_limit_by_r(self):
        r = session_risk_adjustment(
            current_pnl_r=DAILY_CAPS["soft_limit_r"],
            consec_losses=0,
        )
        assert r["status"] == "REDUCE_SIZE"
        assert r["size_mult"] == pytest.approx(0.50)

    def test_soft_limit_by_pct(self):
        r = session_risk_adjustment(
            current_pnl_r=-1.0,
            consec_losses=0,
            current_pnl_pct=DAILY_CAPS["soft_limit_pct"],
        )
        assert r["status"] == "REDUCE_SIZE"

    def test_hard_stop_by_r(self):
        r = session_risk_adjustment(
            current_pnl_r=DAILY_CAPS["hard_stop_r"],
            consec_losses=0,
        )
        assert r["status"] == "HALT"
        assert r["size_mult"] == 0.0

    def test_hard_stop_by_pct(self):
        r = session_risk_adjustment(
            current_pnl_r=-1.0,
            consec_losses=0,
            current_pnl_pct=DAILY_CAPS["hard_stop_pct"],
        )
        assert r["status"] == "HALT"

    def test_consecutive_losses_pause(self):
        r = session_risk_adjustment(
            current_pnl_r=-2.0,
            consec_losses=DAILY_CAPS["consecutive_loss_pause"],
        )
        assert r["status"] == "PAUSED"
        assert r["size_mult"] == 0.0
        assert r["pause_minutes"] == 45

    def test_one_below_consec_pause_is_normal(self):
        r = session_risk_adjustment(
            current_pnl_r=-1.0,
            consec_losses=DAILY_CAPS["consecutive_loss_pause"] - 1,
        )
        assert r["status"] == "NORMAL"

    def test_halt_takes_priority_over_consec_losses(self):
        # Both hard_stop AND consecutive losses triggered → HALT wins
        r = session_risk_adjustment(
            current_pnl_r=DAILY_CAPS["hard_stop_r"],
            consec_losses=DAILY_CAPS["consecutive_loss_pause"],
        )
        assert r["status"] == "HALT"


# ── can_enter ─────────────────────────────────────────────────────────────────

class TestCanEnter:
    def _pos(self, symbol, market="US", sector="Technology"):
        return {"symbol": symbol, "market": market, "sector": sector}

    def test_empty_positions_allowed(self):
        r = can_enter("AAPL", "US", "Technology", [])
        assert r["allowed"] is True

    def test_same_symbol_blocked(self):
        r = can_enter("AAPL", "US", "Technology", [self._pos("AAPL")])
        assert r["allowed"] is False
        assert "AAPL" in r["reason"]

    def test_us_market_cap(self):
        positions = [self._pos(f"SYM{i}") for i in range(4)]   # 4 US positions
        r = can_enter("NEW", "US", "Healthcare", positions)
        assert r["allowed"] is False
        assert "4" in r["reason"]

    def test_under_us_cap_allowed(self):
        positions = [self._pos(f"SYM{i}") for i in range(3)]
        r = can_enter("NEW", "US", "Healthcare", positions)
        assert r["allowed"] is True

    def test_nse_market_cap(self):
        positions = [self._pos(f"NSE{i}", market="NSE", sector="Banking")
                     for i in range(3)]
        r = can_enter("NEW", "NSE", "Banking", positions)
        assert r["allowed"] is False

    def test_sector_cap_us(self):
        positions = [
            self._pos("AAPL", sector="Technology"),
            self._pos("MSFT", sector="Technology"),
        ]
        r = can_enter("GOOGL", "US", "Technology", positions)
        assert r["allowed"] is False
        assert "Technology" in r["reason"]

    def test_different_sector_allowed(self):
        positions = [
            self._pos("AAPL", sector="Technology"),
            self._pos("MSFT", sector="Technology"),
        ]
        r = can_enter("JPM", "US", "Financials", positions)
        assert r["allowed"] is True

    def test_empty_sector_skips_sector_check(self):
        # If sector is unknown (""), sector cap should not block entry
        positions = [
            self._pos("AAPL", sector="Technology"),
            self._pos("MSFT", sector="Technology"),
        ]
        r = can_enter("NEWCO", "US", "", positions)
        assert r["allowed"] is True


# ── ExitPlan.to_dict round-trip ───────────────────────────────────────────────

class TestExitPlanToDict:
    def test_orb_to_dict_has_required_keys(self):
        ep = orb_exit_plan("US_ETF", entry=450.0, stop=448.5, orb_range=2.0)
        d = ep.to_dict()
        for key in ("stop_type", "initial_stop_price", "scale_levels",
                    "trail_type", "trail_trigger_r", "max_hold_bars",
                    "hard_exit_time_et", "hard_exit_time_ist", "breakeven_r"):
            assert key in d, f"Missing key: {key}"

    def test_scale_levels_serializable(self):
        ep = orb_exit_plan("US_LARGE_CAP", entry=450.0, stop=448.5, orb_range=2.0)
        d = ep.to_dict()
        for sl in d["scale_levels"]:
            assert "trigger_r" in sl
            assert "pct_to_close" in sl
            assert "trigger_price" in sl
            assert sl["pct_to_close"] > 0
            assert sl["trigger_r"] > 0


# ── RISK_PER_TRADE_PCT sanity ─────────────────────────────────────────────────

class TestRiskPerTrade:
    def test_all_buckets_present(self):
        for bucket in ("US_ETF", "US_LARGE_CAP", "US_MID_SMALL",
                       "NSE_LARGE_CAP", "NSE_MID_CAP"):
            assert bucket in RISK_PER_TRADE_PCT

    def test_values_in_sane_range(self):
        for bucket, pct in RISK_PER_TRADE_PCT.items():
            assert 0.001 <= pct <= 0.02, f"{bucket}: {pct} outside expected range"
