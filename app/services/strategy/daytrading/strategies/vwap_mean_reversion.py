"""
VWAP Mean Reversion — Strategy 2

Concept:
    Price tends to revert to VWAP after extended deviations.
    Enter when price drops below VWAP with oversold RSI and reversal evidence.

Edge:
    Institutional algorithms use VWAP as a benchmark — they buy when price
    drops below it, creating the reversion force we trade.

Typical trades per day: 0.5–2 per symbol
Best conditions: BULL_OPEN, liquid names (SPY, QQQ, AAPL, MSFT)
Known weaknesses: Trending days where price never returns to VWAP

Reality calibration:
    SPY intraday moves 0.1–0.3% from VWAP on most days.
    A 0.6% VWAP deviation on SPY is rare (strong trend day).
    A 0.3% deviation with RSI < 38 is tradeable and occurs ~2x per week.
    Thresholds must scale to the symbol's typical intraday ATR%.
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import (
    compute_vwap, localize_for_symbol, market_session,
)
from app.services.strategy.daytrading.models import DayTradeSignal
from app.services.strategy.daytrading.risk_templates import (
    get_symbol_bucket, vwap_exit_plan,
)

# Entry window as offsets from the session open so it tracks the right market:
# skip the first 15 min of noise; no new entries after open+5h.
VWAP_MR_START_OFFSET_MIN = 15
VWAP_MR_LAST_ENTRY_OFFSET_MIN = 300


class VWAPMeanReversion:
    name = "VWAPMeanReversion"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        "vwap_atr_distance": 0.4,
        "rsi_period": 14,
        "rsi_oversold": 38,         # US default; NSE uses 35 (see _NSE_RSI below)
        "rsi_overbought": 62,       # for short side; NSE uses 65
        "wick_ratio_min": 0.40,
        "vol_ratio_min": 1.3,
        # Stop: US_ETF=0.8×ATR, US_LARGE=1.0×ATR, NSE=1.2×ATR (from risk_templates)
        "atr_stop_mult": 1.0,
        "atr_target_mult": 1.8,     # legacy single-target (superseded by ExitPlan)
        "max_hold_bars": 16,        # reduced from 20
        "min_rr": 1.5,
        # Symmetric VWAP slope gate: |slope| > this fraction per bar → skip
        # Replaces the old asymmetric max_vwap_drop_pct check.
        "vwap_slope_gate": 0.0002,  # 0.02% per bar
    }

    _STOP_ATR: dict[str, float] = {
        "US_ETF":        0.80,
        "US_LARGE_CAP":  1.00,
        "US_MID_SMALL":  1.00,
        "NSE_LARGE_CAP": 1.20,
        "NSE_MID_CAP":   1.20,
    }
    _NSE_RSI_OVERSOLD  = 35   # stricter for NSE (VWAP carries less mechanical force)
    _NSE_RSI_OVERBOUGHT = 65

    def generate_signals(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        symbol: str,
        config: dict | None = None,
        regime: str = "BULL_OPEN",
    ) -> list[DayTradeSignal]:
        cfg = {**self.default_config, **(config or {})}
        signals: list[DayTradeSignal] = []

        bucket = get_symbol_bucket(symbol, market="NSE" if "." in symbol else "US")
        is_nse = "NSE" in bucket

        # Long side: BULL_OPEN only.  Short side added for BEAR_OPEN.
        long_ok  = regime in ("BULL_OPEN", "TREND_UP")
        short_ok = regime in ("BEAR_OPEN", "TREND_DOWN") and not is_nse  # NSE short via separate flow
        if not long_ok and not short_ok:
            return signals

        rsi_oversold  = self._NSE_RSI_OVERSOLD  if is_nse else cfg["rsi_oversold"]
        rsi_overbought = self._NSE_RSI_OVERBOUGHT if is_nse else cfg["rsi_overbought"]
        stop_atr_mult = self._STOP_ATR.get(bucket, cfg["atr_stop_mult"])

        today = _today_bars(df_5m, symbol)
        # Need a full ATR/RSI window (14) — the `ta` library raises on fewer rows.
        # NaN guard in the loop still handles the rolling-20 warmup beyond that.
        if today.empty or len(today) < 15:
            return signals

        sess = market_session(symbol)
        mr_min_start = sess.after_open(VWAP_MR_START_OFFSET_MIN)
        mr_last_entry = sess.after_open(VWAP_MR_LAST_ENTRY_OFFSET_MIN)

        today = today.copy()
        today["vwap"] = compute_vwap(today)
        today["rsi"] = tam.RSIIndicator(today["Close"], window=cfg["rsi_period"]).rsi()
        atr_ind = tav.AverageTrueRange(today["High"], today["Low"], today["Close"], window=14)
        today["atr"] = atr_ind.average_true_range()
        vol_avg = today["Volume"].rolling(20).mean()

        open_signal_seen = False

        for i in range(3, len(today)):
            if open_signal_seen:
                break

            bar = today.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None

            if bar_time and (bar_time < mr_min_start or bar_time >= mr_last_entry):
                continue

            if any(pd.isna(bar[c]) for c in ["rsi", "vwap", "atr"]):
                continue
            if pd.isna(vol_avg.iloc[i]):
                continue

            close = float(bar["Close"])
            open_p = float(bar["Open"])
            high = float(bar["High"])
            low = float(bar["Low"])
            vwap = float(bar["vwap"])
            rsi_val = float(bar["rsi"])
            atr_val = float(bar["atr"])
            volume = float(bar["Volume"])
            avg_vol = float(vol_avg.iloc[i])

            if atr_val <= 0 or avg_vol <= 0:
                continue

            bar_range = high - low
            vol_ratio = volume / avg_vol
            prev_close = float(today.iloc[i - 1]["Close"])
            prev_vwap  = float(today.iloc[i - 1]["vwap"])

            # ── Symmetric VWAP slope gate ──────────────────────────────────
            # Replaces old one-sided freefall check.
            if i >= 4:
                vwap_3_ago = float(today.iloc[i - 4]["vwap"])
                if vwap_3_ago > 0:
                    vwap_slope = (vwap - vwap_3_ago) / vwap_3_ago
                    if long_ok  and vwap_slope < -cfg["vwap_slope_gate"]:
                        continue   # VWAP trending down — don't fade long
                    if short_ok and vwap_slope >  cfg["vwap_slope_gate"]:
                        continue   # VWAP trending up — don't fade short

            # ── LONG setup ────────────────────────────────────────────────
            if long_ok:
                vwap_dist_atr = (vwap - close) / atr_val
                if vwap_dist_atr < cfg["vwap_atr_distance"]:
                    continue

                vwap_dist_pct = (vwap - close) / vwap * 100
                lower_wick    = (close - low) / bar_range if bar_range > 0 else 0.0

                if prev_close >= prev_vwap:
                    continue  # prior bar was above VWAP — no context for reversion

                if close <= open_p:
                    continue  # bar must close green

                if not (
                    rsi_val < rsi_oversold
                    and lower_wick > cfg["wick_ratio_min"]
                    and vol_ratio  > cfg["vol_ratio_min"]
                ):
                    continue

                entry  = close
                stop   = entry - stop_atr_mult * atr_val
                # Legacy single target (used for R:R gate only; ExitPlan drives actual exits)
                target = vwap + 0.9 * atr_val

                risk = entry - stop
                if risk <= 0:
                    continue
                rr = (target - entry) / risk
                if rr < cfg["min_rr"]:
                    continue

                exit_plan = vwap_exit_plan(
                    bucket=bucket, entry=entry, stop=stop,
                    vwap=vwap, atr=atr_val, direction="BUY",
                )
                confidence = _score(rsi_val, vwap_dist_atr, vol_ratio, rr)
                signals.append(
                    DayTradeSignal(
                        symbol=symbol, strategy=self.name,
                        direction="BUY", timeframe=self.timeframe,
                        entry_price=round(entry, 4), stop_price=round(stop, 4),
                        target_price=round(target, 4), confidence=round(confidence, 2),
                        reason=(
                            f"Price {vwap_dist_pct:.2f}% ({vwap_dist_atr:.1f}×ATR) below VWAP {vwap:.2f}. "
                            f"RSI {rsi_val:.1f}. Wick {lower_wick:.0%}. Vol {vol_ratio:.1f}×. "
                            f"R:R {rr:.1f}. [{bucket}]"
                        ),
                        regime=regime,
                        indicators={
                            "vwap": round(vwap, 4),
                            "vwap_dist_pct": round(vwap_dist_pct, 3),
                            "vwap_dist_atr": round(vwap_dist_atr, 2),
                            "rsi": round(rsi_val, 2),
                            "wick_ratio": round(lower_wick, 3),
                            "vol_ratio": round(vol_ratio, 2),
                            "atr": round(atr_val, 4),
                            "r_r": round(rr, 2),
                            "bucket": bucket,
                            "exit_plan": exit_plan.to_dict(),
                        },
                        signal_time=bar.name.isoformat(),
                        exit_plan=exit_plan,
                    )
                )
                open_signal_seen = True

            # ── SHORT setup (BEAR_OPEN, US only) ─────────────────────────
            elif short_ok and not open_signal_seen:
                vwap_dist_atr = (close - vwap) / atr_val
                if vwap_dist_atr < cfg["vwap_atr_distance"]:
                    continue

                vwap_dist_pct = (close - vwap) / vwap * 100
                upper_wick    = (high - close) / bar_range if bar_range > 0 else 0.0

                if prev_close <= prev_vwap:
                    continue  # prior bar already below VWAP

                if close >= open_p:
                    continue  # bar must close red

                if not (
                    rsi_val > rsi_overbought
                    and upper_wick > cfg["wick_ratio_min"]
                    and vol_ratio  > cfg["vol_ratio_min"]
                ):
                    continue

                entry  = close
                stop   = entry + stop_atr_mult * atr_val
                target = vwap - 0.9 * atr_val

                risk = stop - entry
                if risk <= 0:
                    continue
                rr = (entry - target) / risk
                if rr < cfg["min_rr"]:
                    continue

                exit_plan = vwap_exit_plan(
                    bucket=bucket, entry=entry, stop=stop,
                    vwap=vwap, atr=atr_val, direction="SELL",
                )
                confidence = _score(rsi_val, vwap_dist_atr, vol_ratio, rr)
                signals.append(
                    DayTradeSignal(
                        symbol=symbol, strategy=self.name,
                        direction="SELL", timeframe=self.timeframe,
                        entry_price=round(entry, 4), stop_price=round(stop, 4),
                        target_price=round(target, 4), confidence=round(confidence, 2),
                        reason=(
                            f"Price {vwap_dist_pct:.2f}% ({vwap_dist_atr:.1f}×ATR) above VWAP {vwap:.2f}. "
                            f"RSI {rsi_val:.1f}. Upper wick {upper_wick:.0%}. Vol {vol_ratio:.1f}×. "
                            f"R:R {rr:.1f}. [{bucket}]"
                        ),
                        regime=regime,
                        indicators={
                            "vwap": round(vwap, 4),
                            "vwap_dist_pct": round(vwap_dist_pct, 3),
                            "vwap_dist_atr": round(vwap_dist_atr, 2),
                            "rsi": round(rsi_val, 2),
                            "wick_ratio": round(upper_wick, 3),
                            "vol_ratio": round(vol_ratio, 2),
                            "atr": round(atr_val, 4),
                            "r_r": round(rr, 2),
                            "bucket": bucket,
                            "exit_plan": exit_plan.to_dict(),
                        },
                        signal_time=bar.name.isoformat(),
                        exit_plan=exit_plan,
                    )
                )
                open_signal_seen = True

        return signals


def _today_bars(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    df = localize_for_symbol(df, symbol)
    idx = df.index
    today = idx[-1].date()
    return df[idx.date == today]


def _score(rsi: float, dist_atr: float, vol_ratio: float, rr: float) -> float:
    base = 0.50
    if rsi < 28:
        base += 0.15
    elif rsi < 33:
        base += 0.08
    if dist_atr > 0.8:
        base += 0.10
    elif dist_atr > 0.6:
        base += 0.05
    if vol_ratio > 2.0:
        base += 0.08
    if rr >= 2.5:
        base += 0.07
    return min(max(base, 0.0), 1.0)
