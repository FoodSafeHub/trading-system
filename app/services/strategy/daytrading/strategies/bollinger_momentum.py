"""
Bollinger Momentum Breakout — Strategy 6
Canonical spec: docs/strategies_spec.md § "Strategy 6 — BollingerMomentum"

Behavior audit (what the old code was doing wrong):
    The previous version was structurally a breakout strategy but behaved like a
    mean-reversion filter in practice because:
    - RSI was capped at 75, killing the strongest momentum candles (RSI 76-85 are
      common on a real BB squeeze breakout bar)
    - No RSI slope check — a sideways or declining RSI in-range still triggered
    - Stop used min(bar_low, entry-ATR), picking the tighter stop and getting
      stopped on normal noise on strong momentum bars
    - No breakout candle quality filter — a close 1 cent above the upper band
      qualified the same as a decisive 2×ATR breakout bar
    - No prior-bar compression check — squeeze was measured on lookback percentile
      but the bar immediately before the breakout was not required to be narrow
    - _score() penalised CHOPPY regime but docstring listed it as a best condition

New behavior (momentum breakout):
    - RSI confirms momentum direction but does NOT cap on the high side for longs
      (rsi_overextended_long = 82 is the only ceiling, handles blow-off only)
    - RSI slope required: RSI must be rising for longs, falling for shorts
    - Stop is placed at the breakout bar's structural low (for longs) buffered by
      ATR fraction, NOT the tighter ATR-only stop
    - Breakout quality: close must be in the upper portion of the bar range (long)
      and close-to-upper-band gap must exceed a minimum fraction of ATR
    - Prior bar compression: bar before the breakout must be narrower than avg,
      confirming the expansion is genuine
    - Regime awareness: TREND_UP gives a confidence bonus and loosens RSI min;
      CHOPPY tightens vol and squeeze requirements but still allows entries;
      BEAR_OPEN/TREND_DOWN flip to shorts only

Concept:
    Volatility contraction (BB squeeze) followed by a decisive RSI-confirmed
    close above the upper band signals an explosive directional move.
    RSI must be rising into the breakout, not just in a numeric range.

Edge:
    Low-volatility coiling periods concentrate energy. The combination of:
    (a) prior compression, (b) decisive band break, (c) rising RSI, (d) volume
    confirmation, and (e) close in the upper half of the breakout bar significantly
    reduces false-breakout entries versus a simple "close > upper band" check.

Typical trades per day: 0–1 per symbol
Best conditions: TREND_UP / BULL_OPEN (trending expansion days)
                 TREND_DOWN / BEAR_OPEN for shorts
                 CHOPPY only with tightened filters (config auto-adjusts)
Known weaknesses: Whipsaws on news spikes; fake expansions that re-enter band within 1-2 bars
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav

from app.services.strategy.daytrading.market_open import ET, compute_vwap, LAST_ENTRY_TIME
from app.services.strategy.daytrading.models import DayTradeSignal

BB_LAST_ENTRY = time(14, 30)


class BollingerMomentum:
    name = "BollingerMomentum"
    timeframe = "5m"
    default_config: dict[str, Any] = {
        # ── Band parameters ──────────────────────────────────────────────────
        "bb_length": 20,
        "bb_std": 2.0,
        # ── Squeeze / contraction detection ─────────────────────────────────
        "contraction_lookback": 20,      # bars of history for squeeze percentile
        "contraction_percentile": 0.25,  # band width must be in lowest 25%
        # ── RSI momentum filters ─────────────────────────────────────────────
        "rsi_period": 14,
        "rsi_long_min": 55,              # RSI must be at least this to confirm upside momentum
        "rsi_short_max": 45,             # RSI must be at most this to confirm downside momentum
        "rsi_overextended_long": 82,     # block only true blow-offs; was 75 (too strict)
        "rsi_overextended_short": 18,    # block only panic capitulation entries
        "require_rsi_rising_long": True, # RSI must have increased vs prior bar for longs
        "require_rsi_falling_short": True,
        "rsi_slope_lookback": 2,         # bars to look back for RSI slope (1 or 2)
        # ── EMA trend filter ─────────────────────────────────────────────────
        "ema_fast": 9,
        # ── Volume ───────────────────────────────────────────────────────────
        "vol_rel_min": 1.2,              # breakout bar volume vs 20-bar avg
        "vol_rolling_bars": 20,
        # ── Breakout quality ─────────────────────────────────────────────────
        "breakout_close_pct": 0.60,      # close must be in top 60% of bar range
        "breakout_min_atr_frac": 0.10,   # close must exceed band by at least 0.10×ATR
        # ── Prior bar compression ─────────────────────────────────────────────
        "require_prior_compression": True,  # prior bar range < 0.8 × avg range
        "compression_atr_frac": 0.80,
        # ── Stop / target ─────────────────────────────────────────────────────
        # Stop is placed at breakout bar structural low/high buffered by ATR fraction.
        # For momentum, this is WIDER than the old ATR-only stop on purpose —
        # we want to stay in through normal retests of the breakout level.
        "stop_atr_buffer": 0.25,        # buffer below bar low (long) or above bar high (short)
        "r_multiple_target": 2.0,
        "min_rr": 1.5,
        # ── Choppy-regime overrides (applied by ConfigAdjuster) ───────────────
        "max_hold_bars": 60,
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

        # Need enough bars for BB warmup + contraction lookback
        today_bars = _today_bars(df_5m)
        if today_bars.empty or len(today_bars) < cfg["bb_length"] + 4:
            return signals

        df = today_bars.copy()

        # ── Indicators ──────────────────────────────────────────────────────────
        bb = tav.BollingerBands(df["Close"], window=cfg["bb_length"], window_dev=cfg["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, float("nan"))

        df["ema_fast"] = tat.EMAIndicator(df["Close"], window=cfg["ema_fast"]).ema_indicator()
        df["rsi"]      = tam.RSIIndicator(df["Close"], window=cfg["rsi_period"]).rsi()
        df["atr"]      = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        df["bar_range"] = df["High"] - df["Low"]
        df["range_avg"] = df["bar_range"].rolling(cfg["vol_rolling_bars"]).mean()

        vol_avg    = df["Volume"].rolling(cfg["vol_rolling_bars"]).mean()
        vwap_ser   = compute_vwap(df)
        df["vwap"] = vwap_ser

        signal_seen = False

        for i in range(cfg["contraction_lookback"], len(df)):
            if signal_seen:
                break

            bar = df.iloc[i]
            bar_time = bar.name.time() if hasattr(bar.name, "time") else None
            if bar_time and bar_time >= BB_LAST_ENTRY:
                break

            req_cols = ["bb_upper", "bb_lower", "bb_width", "ema_fast", "rsi", "atr", "bar_range", "range_avg"]
            if any(pd.isna(bar[c]) for c in req_cols):
                continue
            if pd.isna(vol_avg.iloc[i]):
                continue

            close      = float(bar["Close"])
            open_      = float(bar["Open"])
            high_      = float(bar["High"])
            low_       = float(bar["Low"])
            bb_upper   = float(bar["bb_upper"])
            bb_lower   = float(bar["bb_lower"])
            bb_width   = float(bar["bb_width"])
            ema_val    = float(bar["ema_fast"])
            rsi_val    = float(bar["rsi"])
            atr_val    = float(bar["atr"])
            bar_range  = float(bar["bar_range"])
            range_avg  = float(bar["range_avg"])
            vwap_val   = float(bar["vwap"]) if not pd.isna(bar["vwap"]) else 0.0
            volume     = float(bar["Volume"])
            avg_vol    = float(vol_avg.iloc[i])

            if atr_val <= 0 or avg_vol <= 0 or bar_range <= 0:
                continue

            # ── Squeeze detection ────────────────────────────────────────────────
            lookback_widths = df["bb_width"].iloc[i - cfg["contraction_lookback"]: i].dropna()
            if len(lookback_widths) < 10:
                continue
            squeeze_threshold = lookback_widths.quantile(cfg["contraction_percentile"])
            in_squeeze = bb_width <= squeeze_threshold
            squeeze_score = round(1.0 - bb_width / squeeze_threshold, 3) if squeeze_threshold > 0 else 0.0

            vol_ratio = volume / avg_vol

            # ── RSI slope over last N bars ────────────────────────────────────────
            slope_lb = max(1, cfg["rsi_slope_lookback"])
            rsi_prev = float(df["rsi"].iloc[i - slope_lb]) if not pd.isna(df["rsi"].iloc[i - slope_lb]) else rsi_val
            rsi_slope = rsi_val - rsi_prev   # positive = rising, negative = falling

            # ── EMA slope ────────────────────────────────────────────────────────
            ema_prev = float(df["ema_fast"].iloc[i - 1]) if i >= 1 else ema_val

            # ── Prior bar compression ─────────────────────────────────────────────
            prior_range = float(df["bar_range"].iloc[i - 1]) if i >= 1 else bar_range
            prior_compressed = prior_range < cfg["compression_atr_frac"] * range_avg

            # ── LONG: BB squeeze + decisive breakout above upper band ─────────────
            rejection_reason = ""
            long_candidate = (
                in_squeeze
                and close > bb_upper
                and ema_val > ema_prev
                and close > ema_val
            )

            if long_candidate and regime not in ("BEAR_OPEN",):
                # RSI momentum check — must confirm upside strength
                rsi_ok_long = (
                    rsi_val >= cfg["rsi_long_min"]
                    and rsi_val <= cfg["rsi_overextended_long"]
                )
                rsi_rising_ok = (not cfg["require_rsi_rising_long"]) or (rsi_slope > 0)

                # Breakout quality: close in upper portion of bar
                quality_ok = (
                    (close - low_) / bar_range >= cfg["breakout_close_pct"]
                    and (close - bb_upper) >= cfg["breakout_min_atr_frac"] * atr_val
                )

                # Prior compression
                compression_ok = (not cfg["require_prior_compression"]) or prior_compressed

                # Volume
                vol_ok = vol_ratio >= cfg["vol_rel_min"]

                # VWAP
                vwap_ok = (vwap_val <= 0 or close > vwap_val)

                # Extension guard: close should not be more than 3×ATR above EMA9
                # (blow-off / over-extension that typically reverses within a bar)
                not_extended = (close - ema_val) <= 3.0 * atr_val

                if not rsi_ok_long:
                    rejection_reason = f"RSI {rsi_val:.1f} outside [{cfg['rsi_long_min']}, {cfg['rsi_overextended_long']}]"
                elif not rsi_rising_ok:
                    rejection_reason = f"RSI slope {rsi_slope:+.1f} not rising (require_rsi_rising=True)"
                elif not quality_ok:
                    rejection_reason = (
                        f"Breakout quality weak: close_pct={((close-low_)/bar_range):.2f} "
                        f"(need {cfg['breakout_close_pct']}), "
                        f"band_exceed={(close-bb_upper)/atr_val:.2f}×ATR "
                        f"(need {cfg['breakout_min_atr_frac']})"
                    )
                elif not compression_ok:
                    rejection_reason = f"Prior bar not compressed: range {prior_range:.4f} vs {cfg['compression_atr_frac']}×avg {range_avg:.4f}"
                elif not vol_ok:
                    rejection_reason = f"Volume {vol_ratio:.1f}x below minimum {cfg['vol_rel_min']}x"
                elif not vwap_ok:
                    rejection_reason = f"Close {close:.2f} below VWAP {vwap_val:.2f}"
                elif not not_extended:
                    rejection_reason = f"Over-extended: {(close-ema_val)/atr_val:.1f}×ATR above EMA9 (max 3×)"
                else:
                    # All checks pass
                    entry  = close
                    stop   = low_ - cfg["stop_atr_buffer"] * atr_val
                    risk   = entry - stop
                    if risk <= 0:
                        continue
                    target = entry + risk * cfg["r_multiple_target"]
                    rr     = (target - entry) / risk
                    if rr < cfg["min_rr"]:
                        continue

                    momentum_quality = _momentum_quality_score(
                        rsi_val, rsi_slope, vol_ratio, squeeze_score,
                        (close - low_) / bar_range, (close - bb_upper) / atr_val,
                        direction="LONG",
                    )
                    confidence = _score(
                        rsi_val, vol_ratio, regime, rr, in_squeeze,
                        squeeze_score, rsi_slope, momentum_quality, direction="LONG",
                    )

                    signals.append(DayTradeSignal(
                        symbol=symbol,
                        strategy=self.name,
                        direction="BUY",
                        timeframe=self.timeframe,
                        entry_price=round(entry, 4),
                        stop_price=round(stop, 4),
                        target_price=round(target, 4),
                        confidence=round(confidence, 2),
                        reason=(
                            f"BB momentum LONG: close {close:.2f} > upper {bb_upper:.2f} "
                            f"({(close-bb_upper)/atr_val:.2f}×ATR above band). "
                            f"RSI {rsi_val:.1f} slope {rsi_slope:+.1f}. "
                            f"Vol {vol_ratio:.1f}x. VWAP {'above' if vwap_val > 0 else 'N/A'}. "
                            f"Squeeze {squeeze_score:.2f}. R:R {rr:.1f}."
                        ),
                        regime=regime,
                        indicators={
                            "bb_upper":            round(bb_upper, 4),
                            "bb_lower":            round(bb_lower, 4),
                            "bb_width":            round(bb_width, 6),
                            "squeeze_threshold":   round(squeeze_threshold, 6),
                            "squeeze_score":       round(squeeze_score, 3),
                            "breakout_side":       "LONG",
                            "ema_fast":            round(ema_val, 4),
                            "rsi":                 round(rsi_val, 2),
                            "rsi_slope":           round(rsi_slope, 2),
                            "rel_volume":          round(vol_ratio, 2),
                            "above_vwap":          vwap_val > 0 and close > vwap_val,
                            "vwap":                round(vwap_val, 4) if vwap_val > 0 else None,
                            "atr":                 round(atr_val, 4),
                            "breakout_close_pct":  round((close - low_) / bar_range, 3),
                            "band_exceed_atr":     round((close - bb_upper) / atr_val, 3),
                            "prior_compressed":    prior_compressed,
                            "momentum_quality":    round(momentum_quality, 3),
                            "r_r":                 round(rr, 2),
                        },
                        signal_time=bar.name.isoformat(),
                    ))
                    signal_seen = True
                    continue

            # ── SHORT: BB squeeze + decisive breakdown below lower band ───────────
            short_candidate = (
                in_squeeze
                and close < bb_lower
                and ema_val < ema_prev
                and close < ema_val
            )

            if short_candidate and regime not in ("BULL_OPEN",):
                rsi_ok_short = (
                    rsi_val <= cfg["rsi_short_max"]
                    and rsi_val >= cfg["rsi_overextended_short"]
                )
                rsi_falling_ok = (not cfg["require_rsi_falling_short"]) or (rsi_slope < 0)

                quality_ok = (
                    (high_ - close) / bar_range >= cfg["breakout_close_pct"]
                    and (bb_lower - close) >= cfg["breakout_min_atr_frac"] * atr_val
                )
                compression_ok = (not cfg["require_prior_compression"]) or prior_compressed
                vol_ok         = vol_ratio >= cfg["vol_rel_min"]
                vwap_ok        = (vwap_val <= 0 or close < vwap_val)
                not_extended   = (ema_val - close) <= 3.0 * atr_val

                if not (rsi_ok_short and rsi_falling_ok and quality_ok
                        and compression_ok and vol_ok and vwap_ok and not_extended):
                    continue

                entry  = close
                stop   = high_ + cfg["stop_atr_buffer"] * atr_val
                risk   = stop - entry
                if risk <= 0:
                    continue
                target = entry - risk * cfg["r_multiple_target"]
                rr     = (entry - target) / risk
                if rr < cfg["min_rr"]:
                    continue

                momentum_quality = _momentum_quality_score(
                    rsi_val, rsi_slope, vol_ratio, squeeze_score,
                    (high_ - close) / bar_range, (bb_lower - close) / atr_val,
                    direction="SHORT",
                )
                confidence = _score(
                    rsi_val, vol_ratio, regime, rr, in_squeeze,
                    squeeze_score, rsi_slope, momentum_quality, direction="SHORT",
                )

                signals.append(DayTradeSignal(
                    symbol=symbol,
                    strategy=self.name,
                    direction="SELL",
                    timeframe=self.timeframe,
                    entry_price=round(entry, 4),
                    stop_price=round(stop, 4),
                    target_price=round(target, 4),
                    confidence=round(confidence, 2),
                    reason=(
                        f"BB momentum SHORT: close {close:.2f} < lower {bb_lower:.2f} "
                        f"({(bb_lower-close)/atr_val:.2f}×ATR below band). "
                        f"RSI {rsi_val:.1f} slope {rsi_slope:+.1f}. "
                        f"Vol {vol_ratio:.1f}x. VWAP {'below' if vwap_val > 0 else 'N/A'}. "
                        f"Squeeze {squeeze_score:.2f}. R:R {rr:.1f}."
                    ),
                    regime=regime,
                    indicators={
                        "bb_upper":            round(bb_upper, 4),
                        "bb_lower":            round(bb_lower, 4),
                        "bb_width":            round(bb_width, 6),
                        "squeeze_threshold":   round(squeeze_threshold, 6),
                        "squeeze_score":       round(squeeze_score, 3),
                        "breakout_side":       "SHORT",
                        "ema_fast":            round(ema_val, 4),
                        "rsi":                 round(rsi_val, 2),
                        "rsi_slope":           round(rsi_slope, 2),
                        "rel_volume":          round(vol_ratio, 2),
                        "above_vwap":          vwap_val > 0 and close > vwap_val,
                        "vwap":                round(vwap_val, 4) if vwap_val > 0 else None,
                        "atr":                 round(atr_val, 4),
                        "breakout_close_pct":  round((high_ - close) / bar_range, 3),
                        "band_exceed_atr":     round((bb_lower - close) / atr_val, 3),
                        "prior_compressed":    prior_compressed,
                        "momentum_quality":    round(momentum_quality, 3),
                        "r_r":                 round(rr, 2),
                    },
                    signal_time=bar.name.isoformat(),
                ))
                signal_seen = True

        return signals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _today_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize(ET)
    else:
        idx = idx.tz_convert(ET)
    df = df.copy()
    df.index = idx
    today = idx[-1].date()
    return df[idx.date == today]


def _momentum_quality_score(
    rsi: float,
    rsi_slope: float,
    vol_ratio: float,
    squeeze_score: float,
    close_pct_in_range: float,   # how far close is from the weak end (0=weak, 1=strong)
    band_exceed_atr: float,      # how far close broke beyond band in ATR units
    direction: str,
) -> float:
    """
    0.0–1.0 composite score for the quality of a momentum breakout.
    Higher = cleaner, stronger breakout signal.
    Used in the indicators dict for UI explainability and in confidence scoring.
    """
    score = 0.0
    # RSI strength relative to threshold
    if direction == "LONG":
        score += min((rsi - 55) / 25, 0.25)       # 0 at RSI=55, 0.25 at RSI=80
        score += min(max(rsi_slope / 10, 0), 0.20) # 0 for flat, 0.20 for slope≥10
    else:
        score += min((45 - rsi) / 25, 0.25)
        score += min(max(-rsi_slope / 10, 0), 0.20)
    # Volume
    score += min((vol_ratio - 1.0) / 2.0, 0.20)   # 0 at 1×, 0.20 at 3×
    # Squeeze depth
    score += min(max(squeeze_score, 0), 0.15)
    # Breakout candle quality
    score += min(close_pct_in_range * 0.10, 0.10)
    score += min(band_exceed_atr * 0.05, 0.10)
    return round(min(max(score, 0.0), 1.0), 3)


def _score(
    rsi: float,
    vol_ratio: float,
    regime: str,
    rr: float,
    in_squeeze: bool,
    squeeze_score: float,
    rsi_slope: float,
    momentum_quality: float,
    direction: str,
) -> float:
    """Confidence score 0.0–1.0 for the signal."""
    base = 0.50

    # Squeeze confirmation
    if in_squeeze:
        base += 0.05
    base += min(squeeze_score * 0.08, 0.08)  # deeper squeeze = better

    # Volume
    if vol_ratio > 2.5:
        base += 0.10
    elif vol_ratio > 2.0:
        base += 0.07
    elif vol_ratio > 1.5:
        base += 0.04

    # R:R
    if rr >= 3.0:
        base += 0.07
    elif rr >= 2.5:
        base += 0.04
    elif rr >= 2.0:
        base += 0.02

    # Momentum quality composite
    base += momentum_quality * 0.12

    # Regime alignment
    if direction == "LONG":
        if regime in ("BULL_OPEN", "TREND_UP"):
            base += 0.08
        elif regime == "BEAR_OPEN":
            base -= 0.20
        elif regime == "CHOPPY":
            base -= 0.05       # allowed but discounted
    else:
        if regime in ("BEAR_OPEN", "TREND_DOWN"):
            base += 0.08
        elif regime == "BULL_OPEN":
            base -= 0.20
        elif regime == "CHOPPY":
            base -= 0.05

    return round(min(max(base, 0.0), 1.0), 3)
