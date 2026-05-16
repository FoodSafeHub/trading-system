"""
Config Adjuster — auto-tune strategy parameters for a symbol's characteristics.

Returns an adjusted config dict and a human-readable list of changes so the UI
can show the user exactly what was tweaked and why.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.strategy.daytrading.brain.symbol_profiles import SymbolProfile


@dataclass
class ConfigAdjustment:
    strategy: str
    symbol: str
    original: dict
    adjusted: dict
    changes: list[str]   # human-readable: "orb_minutes: 15 → 10 (tighter range for stable name)"
    reason_summary: str  # one-line summary for the UI banner

    @property
    def has_changes(self) -> bool:
        return bool(self.changes)


class ConfigAdjuster:
    """
    Auto-tune strategy parameters based on a symbol's intraday profile.
    Does NOT mutate strategy objects — returns a new config dict to pass
    into generate_signals() or run_backtest().
    """

    @staticmethod
    def adjust(strategy_name: str, profile: SymbolProfile) -> ConfigAdjustment:
        fn = {
            "ORBBreakout":         ConfigAdjuster._adjust_orb,
            "VWAPMeanReversion":   ConfigAdjuster._adjust_vwap,
            "EMAMomentum":         ConfigAdjuster._adjust_ema,
            "VolumeSpikeReversal": ConfigAdjuster._adjust_vsr,
            "OpeningGapFade":      ConfigAdjuster._adjust_gap,
            "BollingerMomentum":   ConfigAdjuster._adjust_bb,
            "SupertrendTrend":     ConfigAdjuster._adjust_st,
        }.get(strategy_name)

        if fn is None:
            return ConfigAdjustment(
                strategy=strategy_name, symbol=profile.symbol,
                original={}, adjusted={}, changes=[],
                reason_summary="No auto-tuning defined for this strategy.",
            )
        return fn(profile)

    # ── ORB Breakout ──────────────────────────────────────────────────────────
    @staticmethod
    def _adjust_orb(p: SymbolProfile) -> ConfigAdjustment:
        from app.services.strategy.daytrading.strategies.orb_breakout import ORBBreakout
        orig = dict(ORBBreakout.default_config)
        adj = dict(orig)
        changes: list[str] = []

        if p.is_low_volatility:
            # Tighter range, lower targets, less demanding filters
            if adj["orb_minutes"] != 10:
                changes.append(f"Opening range: {adj['orb_minutes']} min → 10 min (tighter for stable name)")
                adj["orb_minutes"] = 10
            if adj["tp_multiplier"] != 1.2:
                changes.append(f"Profit target multiplier: {adj['tp_multiplier']}× → 1.2× (range too narrow for large targets)")
                adj["tp_multiplier"] = 1.2
            if adj["vol_multiple"] != 1.2:
                changes.append(f"Volume threshold: {adj['vol_multiple']}× → 1.2× (lower vol expected on stable names)")
                adj["vol_multiple"] = 1.2
            if adj["rsi_min"] != 40:
                changes.append(f"RSI filter: {adj['rsi_min']}+ → 40+ (lower bar for entry confirmation)")
                adj["rsi_min"] = 40
            if adj["max_orb_atr_ratio"] != 3.5:
                changes.append(f"Max ORB/ATR ratio: {adj['max_orb_atr_ratio']} → 3.5 (stable names have tighter ATR)")
                adj["max_orb_atr_ratio"] = 3.5
            if adj["min_rr"] != 1.5:
                changes.append(f"Min R:R: {adj['min_rr']} → 1.5 (realistic for narrow ranges)")
                adj["min_rr"] = 1.5
            summary = f"Auto-tuned for low-volatility symbol ({p.volatility_pct:.1f}% ATR): tighter range and targets."

        elif p.is_high_volatility:
            # Wider config — high-vol names have wide ORB ranges by definition
            if adj["tp_multiplier"] != 2.5:
                changes.append(f"Profit target multiplier: {adj['tp_multiplier']}x -> 2.5x (high-vol names have follow-through)")
                adj["tp_multiplier"] = 2.5
            if adj["vol_multiple"] != 1.5:
                changes.append(f"Volume threshold: {adj['vol_multiple']}x -> 1.5x (confirm real breakout vs noise)")
                adj["vol_multiple"] = 1.5
            if adj["rsi_min"] != 48:
                changes.append(f"RSI filter: {adj['rsi_min']}+ -> 48+ (momentum confirmation on volatile name)")
                adj["rsi_min"] = 48
            # Key fix: max_orb_atr_ratio=2.5 was calibrated for SPY-like vol.
            # NVDA/TSLA at 2-4% ATR will almost always exceed 2.5x — widen it.
            new_orb_ratio = round(min(4.5, 2.5 + p.volatility_pct * 0.4), 1)
            if adj["max_orb_atr_ratio"] != new_orb_ratio:
                changes.append(
                    f"Max ORB/ATR ratio: {adj['max_orb_atr_ratio']} -> {new_orb_ratio} "
                    f"(wide ORB expected on {p.volatility_pct:.1f}% ATR symbol)"
                )
                adj["max_orb_atr_ratio"] = new_orb_ratio
            # Widen stop so it isn't immediately hit on volatile bars
            if p.volatility_pct >= 2.0 and adj.get("atr_stop_mult", 1.0) < 1.5:
                changes.append(
                    f"ATR stop mult: {adj.get('atr_stop_mult', 1.0)} -> 1.5 "
                    f"(wider stop needed on {p.volatility_pct:.1f}% ATR symbol)"
                )
                adj["atr_stop_mult"] = 1.5
            summary = (
                f"Auto-tuned for high-volatility symbol ({p.volatility_pct:.1f}% ATR): "
                f"wider ORB tolerance ({new_orb_ratio}x), wider stops, bigger targets."
            )

        else:
            summary = f"Standard parameters used ({p.volatility_pct:.1f}% ATR — mid-range volatility)."

        return ConfigAdjustment(
            strategy="ORBBreakout", symbol=p.symbol,
            original=orig, adjusted=adj, changes=changes,
            reason_summary=summary,
        )

    # ── VWAP Mean Reversion ───────────────────────────────────────────────────
    @staticmethod
    def _adjust_vwap(p: SymbolProfile) -> ConfigAdjustment:
        from app.services.strategy.daytrading.strategies.vwap_mean_reversion import VWAPMeanReversion
        orig = dict(VWAPMeanReversion.default_config)
        adj = dict(orig)
        changes: list[str] = []

        if p.is_low_volatility:
            if adj.get("vwap_atr_distance", 0.4) != 0.3:
                changes.append(f"VWAP distance: {adj.get('vwap_atr_distance', 0.4)}×ATR → 0.3×ATR (tighter pullbacks for stable name)")
                adj["vwap_atr_distance"] = 0.3
            if adj.get("atr_stop_mult", 1.2) != 1.0:
                changes.append(f"Stop distance: {adj.get('atr_stop_mult', 1.2)}×ATR → 1.0×ATR (tighter stops)")
                adj["atr_stop_mult"] = 1.0
            summary = f"Auto-tuned for low-volatility symbol: tighter VWAP distance and stops."

        elif p.is_high_volatility:
            if adj.get("vwap_atr_distance", 0.4) != 0.6:
                changes.append(f"VWAP distance: {adj.get('vwap_atr_distance', 0.4)}×ATR → 0.6×ATR (wider pullbacks on volatile name)")
                adj["vwap_atr_distance"] = 0.6
            if adj.get("rsi_oversold", 38) != 33:
                changes.append(f"RSI oversold: {adj.get('rsi_oversold', 38)} → 33 (deeper oversold required on volatile name)")
                adj["rsi_oversold"] = 33
            summary = f"Auto-tuned for high-volatility symbol: wider pullback distance, deeper RSI."

        else:
            summary = f"Standard VWAP parameters ({p.volatility_pct:.1f}% ATR)."

        return ConfigAdjustment(
            strategy="VWAPMeanReversion", symbol=p.symbol,
            original=orig, adjusted=adj, changes=changes,
            reason_summary=summary,
        )

    # ── EMA Momentum ─────────────────────────────────────────────────────────
    @staticmethod
    def _adjust_ema(p: SymbolProfile) -> ConfigAdjustment:
        from app.services.strategy.daytrading.strategies.ema_momentum import EMAMomentum
        orig = dict(EMAMomentum.default_config)
        adj = dict(orig)
        changes: list[str] = []

        if p.is_low_volatility:
            if adj.get("ema_slow", 21) != 34:
                changes.append(f"Slow EMA: {adj.get('ema_slow', 21)} → 34 (smoother signal for stable name)")
                adj["ema_slow"] = 34
            if adj.get("atr_tp_mult", 2.5) != 2.0:
                changes.append(f"Profit target: {adj.get('atr_tp_mult', 2.5)}×ATR → 2.0×ATR (realistic for low-vol)")
                adj["atr_tp_mult"] = 2.0
            summary = "Auto-tuned for low-volatility symbol: slower EMA, tighter target."

        elif p.is_high_volatility:
            if adj.get("atr_tp_mult", 2.5) != 3.0:
                changes.append(f"Profit target: {adj.get('atr_tp_mult', 2.5)}×ATR → 3.0×ATR (let winners run on high-vol)")
                adj["atr_tp_mult"] = 3.0
            summary = "Auto-tuned for high-volatility symbol: wider profit target."

        else:
            summary = f"Standard EMA parameters ({p.volatility_pct:.1f}% ATR)."

        return ConfigAdjustment(
            strategy="EMAMomentum", symbol=p.symbol,
            original=orig, adjusted=adj, changes=changes,
            reason_summary=summary,
        )

    # ── Volume Spike Reversal ─────────────────────────────────────────────────
    @staticmethod
    def _adjust_vsr(p: SymbolProfile) -> ConfigAdjustment:
        from app.services.strategy.daytrading.strategies.volume_spike_reversal import VolumeSpikeReversal
        orig = dict(VolumeSpikeReversal.default_config)
        adj = dict(orig)
        changes: list[str] = []

        if p.is_low_volatility:
            if adj.get("spike_multiple", 2.5) != 2.0:
                changes.append(f"Volume spike threshold: {adj.get('spike_multiple', 2.5)}× → 2.0× (lower bar for stable names)")
                adj["spike_multiple"] = 2.0
            if adj.get("rsi_max", 38) != 42:
                changes.append(f"RSI oversold ceiling: {adj.get('rsi_max', 38)} → 42 (stable names don't reach deep oversold)")
                adj["rsi_max"] = 42
            summary = "Auto-tuned for low-volatility symbol: lower spike threshold, higher RSI ceiling."

        elif p.is_high_volatility:
            if adj.get("spike_multiple", 2.5) != 3.0:
                changes.append(f"Volume spike threshold: {adj.get('spike_multiple', 2.5)}× → 3.0× (require strong spike on volatile name)")
                adj["spike_multiple"] = 3.0
            summary = "Auto-tuned for high-volatility symbol: require stronger volume spike."

        else:
            summary = f"Standard VSR parameters ({p.volatility_pct:.1f}% ATR)."

        return ConfigAdjustment(
            strategy="VolumeSpikeReversal", symbol=p.symbol,
            original=orig, adjusted=adj, changes=changes,
            reason_summary=summary,
        )

    # ── Opening Gap Fade ─────────────────────────────────────────────────────
    @staticmethod
    def _adjust_gap(p: SymbolProfile) -> ConfigAdjustment:
        from app.services.strategy.daytrading.strategies.opening_gap_fade import OpeningGapFade
        orig = dict(OpeningGapFade.default_config)
        adj = dict(orig)
        changes: list[str] = []

        if p.gap_frequency_pct < 5.0:
            # Gaps are rare — need to widen the net
            if adj.get("gap_min_pct", 0.5) != 0.4:
                changes.append(f"Min gap: {adj.get('gap_min_pct', 0.5)}% → 0.4% (lower bar since gaps are rare on {p.symbol})")
                adj["gap_min_pct"] = 0.4
            summary = f"Auto-tuned for low gap-frequency symbol ({p.gap_frequency_pct:.0f}% gap days): lower minimum gap."

        elif p.gap_frequency_pct > 12.0:
            # Gaps are common — be selective
            if adj.get("gap_min_pct", 0.5) != 0.75:
                changes.append(f"Min gap: {adj.get('gap_min_pct', 0.5)}% → 0.75% (filter noise on frequently gapping symbol)")
                adj["gap_min_pct"] = 0.75
            summary = f"Auto-tuned for high gap-frequency symbol ({p.gap_frequency_pct:.0f}% gap days): require larger gap."

        else:
            summary = f"Standard gap fade parameters ({p.gap_frequency_pct:.0f}% of days have gaps)."

        return ConfigAdjustment(
            strategy="OpeningGapFade", symbol=p.symbol,
            original=orig, adjusted=adj, changes=changes,
            reason_summary=summary,
        )

    # ── Bollinger Momentum ────────────────────────────────────────────────────
    @staticmethod
    def _adjust_bb(p: SymbolProfile) -> ConfigAdjustment:
        from app.services.strategy.daytrading.strategies.bollinger_momentum import BollingerMomentum
        orig = dict(BollingerMomentum.default_config)
        adj = dict(orig)
        changes: list[str] = []

        if p.is_low_volatility:
            # Wider squeeze window — low-vol names don't compress as sharply
            if adj.get("contraction_percentile", 0.25) != 0.35:
                changes.append(f"Contraction percentile: {adj['contraction_percentile']} -> 0.35 (wider squeeze window for stable name)")
                adj["contraction_percentile"] = 0.35
            # Lower vol bar — stable names never spike 1.2× on breakouts
            if adj.get("vol_rel_min", 1.2) != 1.0:
                changes.append(f"Volume minimum: {adj['vol_rel_min']}x -> 1.0x (low vol name)")
                adj["vol_rel_min"] = 1.0
            # RSI thresholds widen — low-vol names show weaker momentum readings
            if adj.get("rsi_long_min", 55) != 50:
                changes.append("rsi_long_min: 55 -> 50 (weaker RSI signals on low-vol name)")
                adj["rsi_long_min"] = 50
            if adj.get("rsi_short_max", 45) != 50:
                changes.append("rsi_short_max: 45 -> 50 (weaker RSI on low-vol short)")
                adj["rsi_short_max"] = 50
            # Smaller target — low-vol names move less
            if adj.get("r_multiple_target", 2.0) != 1.5:
                changes.append(f"R target: {adj['r_multiple_target']}R -> 1.5R (tighter range for stable name)")
                adj["r_multiple_target"] = 1.5
            summary = "Auto-tuned for low-volatility: wider squeeze window, lower vol bar, relaxed RSI, smaller target."

        elif p.is_high_volatility:
            # Stricter squeeze — volatile names are always "wide", need deeper contraction
            if adj.get("contraction_percentile", 0.25) != 0.15:
                changes.append(f"Contraction percentile: {adj['contraction_percentile']} -> 0.15 (stricter squeeze on volatile name)")
                adj["contraction_percentile"] = 0.15
            # Higher RSI floor — volatile names show stronger momentum on real breakouts
            if adj.get("rsi_long_min", 55) != 60:
                changes.append("rsi_long_min: 55 -> 60 (require stronger RSI on volatile name)")
                adj["rsi_long_min"] = 60
            if adj.get("rsi_short_max", 45) != 40:
                changes.append("rsi_short_max: 45 -> 40 (require weaker RSI on volatile short)")
                adj["rsi_short_max"] = 40
            # Wider stop buffer — high-vol bars have large wicks
            if adj.get("stop_atr_buffer", 0.25) != 0.4:
                changes.append(f"Stop ATR buffer: {adj['stop_atr_buffer']} -> 0.4 (wider buffer for volatile name)")
                adj["stop_atr_buffer"] = 0.4
            # Larger target — volatile names make bigger moves after squeeze
            if adj.get("r_multiple_target", 2.0) != 2.5:
                changes.append(f"R target: {adj['r_multiple_target']}R -> 2.5R (larger moves on {p.volatility_pct:.1f}% ATR)")
                adj["r_multiple_target"] = 2.5
            summary = f"Auto-tuned for high-volatility ({p.volatility_pct:.1f}% ATR): stricter squeeze, stronger RSI required, wider stop buffer, larger target."

        else:
            summary = f"Standard BB momentum parameters ({p.volatility_pct:.1f}% ATR)."

        return ConfigAdjustment(
            strategy="BollingerMomentum", symbol=p.symbol,
            original=orig, adjusted=adj, changes=changes,
            reason_summary=summary,
        )

    # ── Supertrend Trend ──────────────────────────────────────────────────────
    @staticmethod
    def _adjust_st(p: SymbolProfile) -> ConfigAdjustment:
        from app.services.strategy.daytrading.strategies.supertrend_trend import SupertrendTrend
        orig = dict(SupertrendTrend.default_config)
        adj = dict(orig)
        changes: list[str] = []

        if p.is_low_volatility:
            if adj.get("st_multiplier", 3.0) != 2.0:
                changes.append(f"ST multiplier: {adj.get('st_multiplier', 3.0)} -> 2.0 (tighter bands on stable name)")
                adj["st_multiplier"] = 2.0
            if adj.get("pullback_atr_dist", 0.8) != 0.5:
                changes.append(f"Pullback distance: {adj.get('pullback_atr_dist', 0.8)}x ATR -> 0.5x (closer pullbacks on low-vol)")
                adj["pullback_atr_dist"] = 0.5
            if adj.get("atr_stop_mult", 1.0) != 0.8:
                changes.append(f"ATR stop mult: {adj.get('atr_stop_mult', 1.0)} -> 0.8 (tighter stop on stable name)")
                adj["atr_stop_mult"] = 0.8
            summary = f"Auto-tuned for low-volatility: tighter ST bands, closer pullback window, tighter stop."

        elif p.is_high_volatility:
            if adj.get("st_multiplier", 3.0) != 3.5:
                changes.append(f"ST multiplier: {adj.get('st_multiplier', 3.0)} -> 3.5 (wider bands needed on {p.volatility_pct:.1f}% ATR)")
                adj["st_multiplier"] = 3.5
            if adj.get("pullback_atr_dist", 0.8) != 1.2:
                changes.append(f"Pullback distance: {adj.get('pullback_atr_dist', 0.8)}x ATR -> 1.2x (wider pullbacks on volatile name)")
                adj["pullback_atr_dist"] = 1.2
            if adj.get("atr_stop_mult", 1.0) != 1.5:
                changes.append(f"ATR stop mult: {adj.get('atr_stop_mult', 1.0)} -> 1.5 (wider stop needed on {p.volatility_pct:.1f}% ATR name)")
                adj["atr_stop_mult"] = 1.5
            if adj.get("r_multiple_target", 2.0) != 2.5:
                changes.append(f"R target: {adj.get('r_multiple_target', 2.0)}R -> 2.5R (larger moves on high-vol)")
                adj["r_multiple_target"] = 2.5
            summary = f"Auto-tuned for high-volatility ({p.volatility_pct:.1f}% ATR): wider bands, bigger pullback window, wider stop, larger target."

        else:
            summary = f"Standard Supertrend parameters ({p.volatility_pct:.1f}% ATR)."

        return ConfigAdjustment(
            strategy="SupertrendTrend", symbol=p.symbol,
            original=orig, adjusted=adj, changes=changes,
            reason_summary=summary,
        )
