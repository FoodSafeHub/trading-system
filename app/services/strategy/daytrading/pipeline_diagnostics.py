"""
PipelineDiagnostics — tracks every step of the signal generation and backtest
pipeline so that "no trades" results can be diagnosed precisely.

Populated by runner.run_signals(), run_backtest(), and run_backtest_all().
Returned in every API response and displayed in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineDiagnostics:
    # ── Data quality ─────────────────────────────────────────────────────────
    symbol: str = ""
    period: str = ""
    bars_loaded_5m: int = 0
    bars_loaded_15m: int = 0
    bars_in_market_hours: int = 0
    trading_days_found: int = 0
    trading_days_skipped_short: int = 0   # days with < 4 bars
    earliest_bar: str = ""
    latest_bar: str = ""
    timezone: str = ""
    data_warning: str = ""               # non-empty = data issue detected

    # ── Regime ───────────────────────────────────────────────────────────────
    regime_distribution: dict[str, int] = field(default_factory=dict)
    days_skipped_by_regime: int = 0      # regime_allows_strategy returned False

    # ── Raw signal generation ─────────────────────────────────────────────────
    raw_signals_generated: int = 0
    raw_buy_signals: int = 0
    raw_sell_signals: int = 0
    raw_hold_signals: int = 0            # direction == HOLD (filtered pre-backtest)
    strategies_run: list[str] = field(default_factory=list)
    strategies_skipped_by_regime: list[str] = field(default_factory=list)

    # ── Signal rejection breakdown ────────────────────────────────────────────
    rejected_by_brain_total: int = 0
    rejected_by_regime: int = 0
    rejected_by_volume: int = 0
    rejected_by_rr: int = 0
    rejected_by_time: int = 0
    rejected_by_extension: int = 0
    rejected_by_kill_switch: int = 0
    rejected_other: int = 0
    rejection_reasons: list[str] = field(default_factory=list)   # top 5 unique reasons

    # ── Post-filter ───────────────────────────────────────────────────────────
    accepted_signals: int = 0

    # ── Backtest execution ────────────────────────────────────────────────────
    trades_opened: int = 0
    trades_closed: int = 0
    trades_skipped_no_future_bars: int = 0

    # ── First-hour window (9:30–10:30 AM ET) ─────────────────────────────────
    first_hour_bars_loaded: int = 0
    first_hour_raw_signals: int = 0
    first_hour_brain_rejections: int = 0
    first_hour_executed_trades: int = 0
    first_hour_top_rejection_reason: str = ""
    first_hour_rejection_counts: dict[str, int] = field(default_factory=dict)

    # ── Root cause diagnosis ──────────────────────────────────────────────────
    root_cause: str = ""      # human-readable single-sentence root cause
    diagnosis_steps: list[str] = field(default_factory=list)

    def finalise(self) -> None:
        """
        Compute root_cause and diagnosis_steps after the pipeline has run.
        Call this once at the end of run_backtest() / run_signals().
        """
        steps: list[str] = []

        # ── Data check ───────────────────────────────────────────────────────
        if self.bars_loaded_5m == 0:
            self.root_cause = "No 5m data returned from data source."
            steps.append(f"DATA: 0 bars loaded for {self.symbol} ({self.period})")
            self.diagnosis_steps = steps
            return

        steps.append(
            f"DATA: {self.bars_loaded_5m} 5m bars loaded "
            f"({self.earliest_bar} — {self.latest_bar}, tz={self.timezone})"
        )
        if self.trading_days_found == 0:
            self.root_cause = "No trading days found in downloaded data."
            steps.append("DATA: 0 trading days extracted — check date/timezone alignment")
            self.diagnosis_steps = steps
            return

        steps.append(
            f"DATA: {self.trading_days_found} trading days, "
            f"{self.trading_days_skipped_short} skipped (< 4 bars)"
        )
        if self.data_warning:
            steps.append(f"WARNING: {self.data_warning}")

        # ── Regime ───────────────────────────────────────────────────────────
        regime_str = ", ".join(f"{k}:{v}" for k, v in sorted(self.regime_distribution.items()))
        steps.append(f"REGIME: distribution = [{regime_str}], skipped by regime = {self.days_skipped_by_regime}")

        if self.strategies_skipped_by_regime:
            steps.append(f"REGIME: strategies blocked = {self.strategies_skipped_by_regime}")

        # ── Raw signals ───────────────────────────────────────────────────────
        steps.append(
            f"RAW SIGNALS: {self.raw_signals_generated} generated "
            f"(BUY={self.raw_buy_signals}, SELL={self.raw_sell_signals}, "
            f"HOLD={self.raw_hold_signals})"
        )

        if self.raw_signals_generated == 0:
            self.root_cause = (
                "No raw strategy signals were generated. "
                "The strategy found no qualifying setups in the data."
            )
            steps.append(
                "ROOT CAUSE: Strategy conditions were never satisfied "
                "(check volume filters, RSI thresholds, ORB height limits, time cutoffs)"
            )
            self.diagnosis_steps = steps
            return

        # ── Brain filtering ───────────────────────────────────────────────────
        steps.append(
            f"BRAIN: accepted={self.accepted_signals}, "
            f"rejected={self.rejected_by_brain_total} "
            f"(regime={self.rejected_by_regime}, "
            f"vol={self.rejected_by_volume}, rr={self.rejected_by_rr}, "
            f"time={self.rejected_by_time}, ext={self.rejected_by_extension}, "
            f"kill={self.rejected_by_kill_switch}, other={self.rejected_other})"
        )

        if self.rejection_reasons:
            for r in self.rejection_reasons[:5]:
                steps.append(f"  rejection sample: {r}")

        if self.accepted_signals == 0 and self.raw_signals_generated > 0:
            top_reason = _top_rejection(self)
            self.root_cause = (
                f"All {self.raw_signals_generated} raw signals were rejected by brain filters. "
                f"Top reason: {top_reason}."
            )
            steps.append(f"ROOT CAUSE: Brain blocked everything — top rejection = {top_reason}")
            self.diagnosis_steps = steps
            return

        # ── Execution ─────────────────────────────────────────────────────────
        steps.append(
            f"EXECUTION: {self.trades_opened} trades opened, "
            f"{self.trades_closed} closed, "
            f"{self.trades_skipped_no_future_bars} skipped (no future bars)"
        )

        if self.trades_opened == 0 and self.accepted_signals > 0:
            self.root_cause = (
                f"{self.accepted_signals} signals passed brain filters but no trades were opened. "
                "Check position sizing, entry price vs available bars, or execution logic."
            )
            steps.append("ROOT CAUSE: Accepted signals existed but no positions were opened")
            self.diagnosis_steps = steps
            return

        if self.trades_opened > 0:
            self.root_cause = (
                f"Pipeline working: {self.trades_opened} trades executed from "
                f"{self.raw_signals_generated} raw signals."
            )
            steps.append(f"STATUS: OK — {self.trades_opened} trades executed")

        # ── First-hour summary ────────────────────────────────────────────────
        if self.first_hour_bars_loaded > 0 or self.first_hour_raw_signals > 0:
            fh_parts = [
                f"bars={self.first_hour_bars_loaded}",
                f"raw_signals={self.first_hour_raw_signals}",
                f"brain_rejected={self.first_hour_brain_rejections}",
                f"executed={self.first_hour_executed_trades}",
            ]
            if self.first_hour_top_rejection_reason:
                fh_parts.append(f"top_rejection={self.first_hour_top_rejection_reason}")
            steps.append(f"FIRST HOUR (9:30–10:30): {', '.join(fh_parts)}")

        self.diagnosis_steps = steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "period": self.period,
            "data": {
                "bars_loaded_5m": self.bars_loaded_5m,
                "bars_loaded_15m": self.bars_loaded_15m,
                "bars_in_market_hours": self.bars_in_market_hours,
                "trading_days_found": self.trading_days_found,
                "trading_days_skipped_short": self.trading_days_skipped_short,
                "earliest_bar": self.earliest_bar,
                "latest_bar": self.latest_bar,
                "timezone": self.timezone,
                "data_warning": self.data_warning,
            },
            "regime": {
                "distribution": self.regime_distribution,
                "days_skipped_by_regime": self.days_skipped_by_regime,
                "strategies_skipped_by_regime": self.strategies_skipped_by_regime,
            },
            "signals": {
                "strategies_run": self.strategies_run,
                "raw_signals_generated": self.raw_signals_generated,
                "raw_buy_signals": self.raw_buy_signals,
                "raw_sell_signals": self.raw_sell_signals,
                "raw_hold_signals": self.raw_hold_signals,
            },
            "brain_filter": {
                "accepted_signals": self.accepted_signals,
                "rejected_by_brain_total": self.rejected_by_brain_total,
                "rejected_by_regime": self.rejected_by_regime,
                "rejected_by_volume": self.rejected_by_volume,
                "rejected_by_rr": self.rejected_by_rr,
                "rejected_by_time": self.rejected_by_time,
                "rejected_by_extension": self.rejected_by_extension,
                "rejected_by_kill_switch": self.rejected_by_kill_switch,
                "rejected_other": self.rejected_other,
                "rejection_reasons": self.rejection_reasons,
            },
            "execution": {
                "trades_opened": self.trades_opened,
                "trades_closed": self.trades_closed,
                "trades_skipped_no_future_bars": self.trades_skipped_no_future_bars,
            },
            "first_hour": {
                "bars_loaded": self.first_hour_bars_loaded,
                "raw_signals": self.first_hour_raw_signals,
                "brain_rejections": self.first_hour_brain_rejections,
                "executed_trades": self.first_hour_executed_trades,
                "top_rejection_reason": self.first_hour_top_rejection_reason,
                "rejection_counts": self.first_hour_rejection_counts,
            },
            "diagnosis": {
                "root_cause": self.root_cause,
                "steps": self.diagnosis_steps,
            },
        }


def _top_rejection(d: PipelineDiagnostics) -> str:
    counts = {
        "regime_blocked": d.rejected_by_regime,
        "low_volume": d.rejected_by_volume,
        "rr_too_low": d.rejected_by_rr,
        "time_cutoff": d.rejected_by_time,
        "too_extended": d.rejected_by_extension,
        "kill_switch": d.rejected_by_kill_switch,
        "other": d.rejected_other,
    }
    top = max(counts, key=counts.get)
    return top if counts[top] > 0 else "unknown"


def _categorise_rejection(reason: str) -> str:
    """Map a brain rejection reason string to a rejection category code."""
    r = (reason or "").lower()
    if any(k in r for k in ("regime", "state", "market", "bear", "bull", "choppy")):
        return "regime"
    if any(k in r for k in ("volume", "vol", "liquidity")):
        return "volume"
    if any(k in r for k in ("r:r", "rr", "risk/reward", "risk reward", "reward")):
        return "rr"
    if any(k in r for k in ("time", "cutoff", "late", "hour", "15:", "after")):
        return "time"
    if any(k in r for k in ("extend", "stretch", "vwap", "far from")):
        return "extension"
    if any(k in r for k in ("kill", "switch", "daily loss", "max trades", "consecutive")):
        return "kill_switch"
    return "other"
