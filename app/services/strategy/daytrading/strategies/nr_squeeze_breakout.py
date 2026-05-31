"""
NR/Squeeze Breakout — merged replacement for BollingerMomentum + NarrowRangeBreakout.

Concept:
    A Bollinger Band squeeze (volatility contraction) followed by a price breakout
    above/below the band on expanding volume. For US single names and NSE, also
    require the prior bar to be the narrowest-range bar in the last N bars (NR5/NR7),
    confirming the contraction. ETFs use the squeeze alone (NR7 too rare on SPY).

Edge:
    Range compression → explosive directional expansion. The NR bar filters out
    random squeezes and keeps only the tightest coiling setups. Volume surge on the
    breakout bar confirms institutional participation.

Key changes vs predecessors:
    - Stop anchored to NR_bar_low − buffer×ATR (not the breakout bar's low).
      Gives room for the common first-bar retest of the breakout level.
    - ExitPlan with 2–3 scale levels and EMA9 / structure runner.
    - ETFs: Bollinger squeeze alone (no NR requirement).
    - NSE: NR5 instead of NR7 (NR7 too rare on NSE 5m).
    - NSE: wider breakout buffer (0.10% vs 0.05% US).

Retired strategies absorbed:
    - BollingerMomentum (squeeze logic migrated here)
    - NarrowRangeBreakout (NR logic migrated here)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import (
    compute_vwap, localize_for_symbol, market_session,
)
from app.services.strategy.daytrading.models import DayTradeSignal
from app.services.strategy.daytrading.risk_templates import (
    get_symbol_bucket, nr_squeeze_exit_plan,
)

# No new entries in last 30 min before close
NR_SQ_LAST_ENTRY_BEFORE_CLOSE_MIN = 30


class NRSqueezeBreakout:
    name = "NRSqueezeBreakout"
    timeframe = "5m"

    default_config: dict[str, Any] = {
        # Bollinger Band params
        "bb_length": 20,
        "bb_std": 2.0,
        "squeeze_lookback": 20,                 # bars to compute width percentile
        "squeeze_percentile": 0.40,             # squeeze = width in lowest 40%

        # NR bar params
        "nr_lookback_us_single": 7,             # NR7 for US single names
        "nr_lookback_nse": 5,                   # NR5 for NSE (NR7 too rare)
        "nr_lookback_etf": 0,                   # 0 = no NR requirement for ETFs

        # Breakout confirmation
        "breakout_buffer_pct_us": 0.05,         # close must exceed band by this %
        "breakout_buffer_pct_nse": 0.10,        # wider for NSE noise
        "vol_breakout_min": 1.50,               # volume must surge on breakout bar
        "close_in_upper_pct": 0.60,             # close in top 60% of bar (longs)

        # RSI
        "rsi_period": 14,
        "rsi_long_min": 52,
        "rsi_long_max": 82,
        "rsi_short_min": 18,
        "rsi_short_max": 48,
        "require_rsi_rising": True,

        # EMA trend filter
        "ema_fast": 9,
        "require_above_vwap": True,

        # Stop
        "stop_atr_buffer_normal": 0.50,
        "stop_atr_buffer_high_vol_bar": 0.80,   # wider if breakout bar > 1.8×avg range
        "high_vol_bar_ratio": 1.8,
        "stop_max_atr_mult": 1.5,               # never wider than this from entry

        # Risk / time
        "min_rr": 1.5,
        "max_hold_bars": 48,
    }

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
        is_etf = bucket == "US_ETF"

        # Regime gate: longs not in BEAR; shorts not in BULL
        allow_long  = regime not in ("BEAR_OPEN", "TREND_DOWN")
        allow_short = regime not in ("BULL_OPEN", "TREND_UP")
        if not allow_long and not allow_short:
            return signals

        today = _today_bars(df_5m, symbol)
        if today.empty or len(today) < cfg["bb_length"] + 5:
            return signals

        last_entry_time = market_session(symbol).before_close(NR_SQ_LAST_ENTRY_BEFORE_CLOSE_MIN)

        # ── Compute indicators ─────────────────────────────────────────────────
        df = today.copy()
        bb = tav.BollingerBands(df["Close"], window=cfg["bb_length"], window_dev=cfg["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        df["ema9"]  = tat.EMAIndicator(df["Close"], window=cfg["ema_fast"]).ema_indicator()
        df["rsi"]   = tam.RSIIndicator(df["Close"], window=cfg["rsi_period"]).rsi()
        atr_ind     = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
        df["atr"]   = atr_ind.average_true_range()
        df["vwap"]  = compute_vwap(df)
        vol_avg     = df["Volume"].rolling(20).mean()
        avg_range   = (df["High"] - df["Low"]).rolling(20).mean()

        # Determine NR lookback for this symbol type
        nr_lookback: int = (
            0 if is_etf
            else cfg["nr_lookback_nse"] if is_nse
            else cfg["nr_lookback_us_single"]
        )

        # Breakout buffer (%) for this market
        buf_pct = cfg["breakout_buffer_pct_nse"] if is_nse else cfg["breakout_buffer_pct_us"]

        signal_seen = False

        for i in range(max(nr_lookback, cfg["bb_length"]), len(df)):
            if signal_seen:
                break

            bar = df.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None
            if bar_time and bar_time >= last_entry_time:
                break

            # NaN guard
            required = ["bb_upper", "bb_lower", "bb_width", "ema9", "rsi", "atr", "vwap"]
            if any(pd.isna(bar[c]) for c in required):
                continue
            if pd.isna(vol_avg.iloc[i]) or pd.isna(avg_range.iloc[i]):
                continue

            close    = float(bar["Close"])
            open_    = float(bar["Open"])
            high_    = float(bar["High"])
            low_     = float(bar["Low"])
            bb_upper = float(bar["bb_upper"])
            bb_lower = float(bar["bb_lower"])
            bb_width = float(bar["bb_width"])
            ema9_val = float(bar["ema9"])
            rsi_val  = float(bar["rsi"])
            atr_val  = float(bar["atr"])
            vwap_val = float(bar["vwap"])
            volume   = float(bar["Volume"])
            avg_vol  = float(vol_avg.iloc[i])
            avg_rng  = float(avg_range.iloc[i])

            if atr_val <= 0 or avg_vol <= 0:
                continue

            bar_range = high_ - low_
            vol_ratio = volume / avg_vol

            # ── Bollinger squeeze check ────────────────────────────────────────
            recent_widths = df["bb_width"].iloc[max(0, i - cfg["squeeze_lookback"]):i]
            if recent_widths.empty:
                continue
            squeeze_threshold = float(recent_widths.quantile(cfg["squeeze_percentile"]))
            in_squeeze = bb_width <= squeeze_threshold

            if not in_squeeze:
                continue   # no squeeze → skip

            # ── NR bar check (prior bar is the NR bar we anchor stop to) ──────
            nr_bar_low  = low_    # default: current bar (fallback for ETFs)
            nr_bar_high = high_
            if nr_lookback > 0 and i >= nr_lookback:
                window_ranges = (
                    df["High"].iloc[i - nr_lookback: i]
                    - df["Low"].iloc[i - nr_lookback: i]
                )
                prior_bar = df.iloc[i - 1]
                prior_range = float(prior_bar["High"]) - float(prior_bar["Low"])
                # Prior bar is NR if its range is the smallest in the window
                if prior_range <= float(window_ranges.min()) * 1.01:
                    nr_bar_low  = float(prior_bar["Low"])
                    nr_bar_high = float(prior_bar["High"])
                else:
                    continue   # no NR bar → skip for single names / NSE

            # ── RSI slope (require rising for longs, falling for shorts) ──────
            rsi_rising = rsi_falling = False
            if cfg["require_rsi_rising"] and i >= 2:
                rsi_prev = df["rsi"].iloc[i - 1]
                if not pd.isna(rsi_prev):
                    rsi_rising  = rsi_val > float(rsi_prev)
                    rsi_falling = rsi_val < float(rsi_prev)

            # ── LONG: close above upper band ──────────────────────────────────
            if allow_long:
                band_exceed = (close - bb_upper) / close * 100
                close_pct_in_range = (close - low_) / bar_range if bar_range > 0 else 0.0

                if (
                    close > bb_upper                                # breakout above band
                    and band_exceed >= buf_pct                      # decisive break
                    and close_pct_in_range >= cfg["close_in_upper_pct"]
                    and ema9_val > 0 and close > ema9_val           # above EMA9
                    and (not cfg["require_above_vwap"] or close > vwap_val)
                    and cfg["rsi_long_min"] <= rsi_val <= cfg["rsi_long_max"]
                    and (not cfg["require_rsi_rising"] or rsi_rising)
                    and vol_ratio >= cfg["vol_breakout_min"]
                ):
                    # Stop below NR bar low − buffer×ATR
                    is_wide_bar = bar_range > cfg["high_vol_bar_ratio"] * avg_rng
                    stop_buf = cfg["stop_atr_buffer_high_vol_bar"] if is_wide_bar else cfg["stop_atr_buffer_normal"]
                    stop = nr_bar_low - stop_buf * atr_val
                    # Cap: never wider than stop_max_atr_mult×ATR from entry
                    stop = max(stop, close - cfg["stop_max_atr_mult"] * atr_val)

                    risk = close - stop
                    if risk <= 0:
                        continue
                    target = close + risk * 2.0   # legacy 2R gate; ExitPlan drives actual exits
                    rr = (target - close) / risk
                    if rr < cfg["min_rr"]:
                        continue

                    exit_plan = nr_squeeze_exit_plan(
                        bucket=bucket, entry=close, stop=stop, direction="BUY",
                    )
                    confidence = _score(
                        rsi_val, vol_ratio, regime, rr, in_squeeze, bb_width,
                        squeeze_threshold, rsi_rising, direction="LONG",
                    )
                    signals.append(DayTradeSignal(
                        symbol=symbol, strategy=self.name,
                        direction="BUY", timeframe=self.timeframe,
                        entry_price=round(close, 4), stop_price=round(stop, 4),
                        target_price=round(target, 4), confidence=round(confidence, 2),
                        reason=(
                            f"NR/Squeeze breakout LONG: close {close:.2f} > BB upper {bb_upper:.2f} "
                            f"(+{band_exceed:.2f}%). NR{nr_lookback or 'ETF'} bar low {nr_bar_low:.2f}. "
                            f"Squeeze {bb_width:.4f}<={squeeze_threshold:.4f}. "
                            f"Vol {vol_ratio:.1f}×. RSI {rsi_val:.1f}. R:R {rr:.1f}. [{bucket}]"
                        ),
                        regime=regime,
                        indicators={
                            "bb_upper": round(bb_upper, 4),
                            "bb_lower": round(bb_lower, 4),
                            "bb_width": round(bb_width, 4),
                            "squeeze_threshold": round(squeeze_threshold, 4),
                            "nr_bar_low": round(nr_bar_low, 4),
                            "ema9": round(ema9_val, 4),
                            "rsi": round(rsi_val, 2),
                            "vol_ratio": round(vol_ratio, 2),
                            "atr": round(atr_val, 4),
                            "r_r": round(rr, 2),
                            "bucket": bucket,
                            "exit_plan": exit_plan.to_dict(),
                        },
                        signal_time=bar.name.isoformat(),
                        exit_plan=exit_plan,
                    ))
                    signal_seen = True

            # ── SHORT: close below lower band ─────────────────────────────────
            elif allow_short and not signal_seen:
                band_exceed = (bb_lower - close) / close * 100
                close_pct_in_range = (high_ - close) / bar_range if bar_range > 0 else 0.0

                if (
                    close < bb_lower
                    and band_exceed >= buf_pct
                    and close_pct_in_range >= cfg["close_in_upper_pct"]   # close in lower 60% (i.e. top of bear bar)
                    and ema9_val > 0 and close < ema9_val
                    and (not cfg["require_above_vwap"] or close < vwap_val)
                    and cfg["rsi_short_min"] <= rsi_val <= cfg["rsi_short_max"]
                    and (not cfg["require_rsi_rising"] or rsi_falling)
                    and vol_ratio >= cfg["vol_breakout_min"]
                ):
                    is_wide_bar = bar_range > cfg["high_vol_bar_ratio"] * avg_rng
                    stop_buf = cfg["stop_atr_buffer_high_vol_bar"] if is_wide_bar else cfg["stop_atr_buffer_normal"]
                    stop = nr_bar_high + stop_buf * atr_val
                    stop = min(stop, close + cfg["stop_max_atr_mult"] * atr_val)

                    risk = stop - close
                    if risk <= 0:
                        continue
                    target = close - risk * 2.0
                    rr = (close - target) / risk
                    if rr < cfg["min_rr"]:
                        continue

                    exit_plan = nr_squeeze_exit_plan(
                        bucket=bucket, entry=close, stop=stop, direction="SELL",
                    )
                    confidence = _score(
                        rsi_val, vol_ratio, regime, rr, in_squeeze, bb_width,
                        squeeze_threshold, rsi_falling, direction="SHORT",
                    )
                    signals.append(DayTradeSignal(
                        symbol=symbol, strategy=self.name,
                        direction="SELL", timeframe=self.timeframe,
                        entry_price=round(close, 4), stop_price=round(stop, 4),
                        target_price=round(target, 4), confidence=round(confidence, 2),
                        reason=(
                            f"NR/Squeeze breakout SHORT: close {close:.2f} < BB lower {bb_lower:.2f} "
                            f"(-{band_exceed:.2f}%). NR{nr_lookback or 'ETF'} bar high {nr_bar_high:.2f}. "
                            f"Squeeze {bb_width:.4f}<={squeeze_threshold:.4f}. "
                            f"Vol {vol_ratio:.1f}×. RSI {rsi_val:.1f}. R:R {rr:.1f}. [{bucket}]"
                        ),
                        regime=regime,
                        indicators={
                            "bb_upper": round(bb_upper, 4),
                            "bb_lower": round(bb_lower, 4),
                            "bb_width": round(bb_width, 4),
                            "squeeze_threshold": round(squeeze_threshold, 4),
                            "nr_bar_high": round(nr_bar_high, 4),
                            "ema9": round(ema9_val, 4),
                            "rsi": round(rsi_val, 2),
                            "vol_ratio": round(vol_ratio, 2),
                            "atr": round(atr_val, 4),
                            "r_r": round(rr, 2),
                            "bucket": bucket,
                            "exit_plan": exit_plan.to_dict(),
                        },
                        signal_time=bar.name.isoformat(),
                        exit_plan=exit_plan,
                    ))
                    signal_seen = True

        return signals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _today_bars(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    df = localize_for_symbol(df, symbol)
    idx = df.index
    today = idx[-1].date()
    return df[idx.date == today]


def _score(
    rsi: float,
    vol_ratio: float,
    regime: str,
    rr: float,
    in_squeeze: bool,
    bb_width: float,
    squeeze_threshold: float,
    momentum_confirmed: bool,
    direction: str,
) -> float:
    base = 0.50
    # Volume quality
    if vol_ratio > 2.5:
        base += 0.12
    elif vol_ratio > 1.8:
        base += 0.06
    # Squeeze tightness (tighter = more coiled = better)
    if squeeze_threshold > 0:
        tightness = 1.0 - (bb_width / squeeze_threshold)
        base += min(0.10, max(0.0, tightness * 0.15))
    # RSI momentum
    if direction == "LONG":
        if 60 <= rsi <= 75:
            base += 0.07
        if regime in ("BULL_OPEN", "TREND_UP"):
            base += 0.08
        elif regime in ("BEAR_OPEN", "TREND_DOWN"):
            base -= 0.15
    else:
        if 25 <= rsi <= 40:
            base += 0.07
        if regime in ("BEAR_OPEN", "TREND_DOWN"):
            base += 0.08
        elif regime in ("BULL_OPEN", "TREND_UP"):
            base -= 0.15
    if momentum_confirmed:
        base += 0.05
    if rr >= 3.0:
        base += 0.08
    elif rr >= 2.0:
        base += 0.04
    return min(max(base, 0.0), 1.0)
