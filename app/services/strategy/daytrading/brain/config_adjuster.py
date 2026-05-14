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
            # Standard or wider config — let winners run
            if adj["tp_multiplier"] != 2.5:
                changes.append(f"Profit target multiplier: {adj['tp_multiplier']}× → 2.5× (high-vol names have follow-through)")
                adj["tp_multiplier"] = 2.5
            if adj["vol_multiple"] != 1.5:
                changes.append(f"Volume threshold: {adj['vol_multiple']}× → 1.5× (confirm real breakout vs noise)")
                adj["vol_multiple"] = 1.5
            if adj["rsi_min"] != 48:
                changes.append(f"RSI filter: {adj['rsi_min']}+ → 48+ (momentum confirmation on volatile name)")
                adj["rsi_min"] = 48
            summary = f"Auto-tuned for high-volatility symbol ({p.volatility_pct:.1f}% ATR): wider targets, tighter volume filter."

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
