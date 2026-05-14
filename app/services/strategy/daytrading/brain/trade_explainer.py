"""
Trade Explainer — explains WHY trades fired or didn't in plain English.

Used when a backtest returns 0 trades, or when a signal is accepted/rejected,
so the user always gets actionable feedback instead of a blank result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.strategy.daytrading.brain.symbol_profiles import SymbolProfile
from app.services.strategy.daytrading.brain.strategy_selector import SelectionResult

ROOT_CAUSES = {
    "strategy_not_for_this_symbol": "Strategy is structurally unsuited for this symbol's volatility",
    "market_was_choppy":            "Market was in a choppy, directionless regime during the period",
    "setup_too_strict":             "Strategy filters are too strict for this symbol's price action",
    "data_too_short":               "Period is too short to capture enough setups",
    "regime_mismatch":              "Market regime blocked this strategy for most/all days",
    "data_issue":                   "Data quality or availability problem",
}


@dataclass
class BacktestExplanation:
    symbol: str
    strategy: str
    period: str
    why_no_trades: str
    root_cause: str
    root_cause_label: str
    recommendations: list[str]
    alternative_strategy: str
    alternative_period: str
    fit_score: float          # 0–1: how well the strategy fits this symbol
    urgency: str              # "high" | "medium" | "low" — how wrong was the choice

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "period": self.period,
            "why_no_trades": self.why_no_trades,
            "root_cause": self.root_cause,
            "root_cause_label": self.root_cause_label,
            "recommendations": self.recommendations,
            "alternative_strategy": self.alternative_strategy,
            "alternative_period": self.alternative_period,
            "fit_score": round(self.fit_score, 2),
            "urgency": self.urgency,
        }


class TradeExplainer:
    """
    Diagnose zero-trade or low-trade backtests and explain the result
    in plain language a day trader can act on immediately.
    """

    @staticmethod
    def explain_empty_backtest(
        symbol: str,
        strategy: str,
        period: str,
        profile: SymbolProfile,
        selection: SelectionResult,
        regime_counts: dict[str, int] | None = None,
    ) -> BacktestExplanation:
        """
        Produce a full explanation for why a backtest returned zero trades.

        regime_counts: optional dict of {regime: days_count} from the backtest run,
                       e.g. {"BULL_OPEN": 18, "BEAR_OPEN": 12, "CHOPPY": 30}
        """
        fit_score = selection.symbol_fit_scores.get(strategy, 0.0)
        alt = selection.primary if selection.primary != strategy else (selection.secondary or "EMAMomentum")

        # ── Determine root cause ──────────────────────────────────────────────
        root_cause, why, recommendations, urgency = TradeExplainer._diagnose(
            symbol, strategy, period, profile, selection, fit_score, regime_counts or {}
        )

        alt_period = _suggest_period(strategy, period)

        return BacktestExplanation(
            symbol=symbol,
            strategy=strategy,
            period=period,
            why_no_trades=why,
            root_cause=root_cause,
            root_cause_label=ROOT_CAUSES.get(root_cause, root_cause),
            recommendations=recommendations,
            alternative_strategy=alt,
            alternative_period=alt_period,
            fit_score=fit_score,
            urgency=urgency,
        )

    @staticmethod
    def _diagnose(
        symbol: str,
        strategy: str,
        period: str,
        profile: SymbolProfile,
        selection: SelectionResult,
        fit_score: float,
        regime_counts: dict[str, int],
    ) -> tuple[str, str, list[str], str]:
        """Returns (root_cause, why_text, recommendations, urgency)."""

        sym = symbol
        vol = profile.volatility_pct
        alt = selection.primary if selection.primary != strategy else (selection.secondary or "EMAMomentum")

        # ── Strategy not for this symbol (worst mismatch) ─────────────────────
        if strategy in profile.avoid_strategies or fit_score < 0.25:
            why = _orb_on_stable(sym, vol, profile) if strategy == "ORBBreakout" and profile.is_low_volatility else (
                _vwap_on_volatile(sym, vol) if strategy == "VWAPMeanReversion" and profile.is_high_volatility else
                f"{strategy} is not a statistical fit for {sym} ({profile.market_type}). "
                f"The strategy's entry conditions require characteristics ({vol:.1f}% ATR does not provide) "
                f"that this symbol doesn't exhibit in its normal trading behavior."
            )
            recs = [
                f"Switch to **{alt}** — fit score {selection.symbol_fit_scores.get(alt, 0):.0%} for {sym}.",
                f"If you want to test {strategy}, use a higher-volatility symbol like TSLA or NVDA.",
                f"Try a longer period (90d+) — more data means more edge-case days might appear.",
            ]
            if strategy == "ORBBreakout":
                recs.insert(1, f"If you must use ORB on {sym}, set orb_minutes=10 (tighter range).")
            return "strategy_not_for_this_symbol", why, recs, "high"

        # ── Regime mismatch ───────────────────────────────────────────────────
        total_days = sum(regime_counts.values()) if regime_counts else 0
        choppy_days = regime_counts.get("CHOPPY", 0)
        bear_days = regime_counts.get("BEAR_OPEN", 0)
        if total_days > 0:
            choppy_pct = choppy_days / total_days * 100
            bear_pct = bear_days / total_days * 100
            if choppy_pct > 50 and strategy in ("ORBBreakout", "EMAMomentum"):
                why = (
                    f"{sym} was in a **choppy, directionless regime for {choppy_pct:.0f}% of the period**. "
                    f"{strategy} requires trending conditions. During choppy sessions, breakouts fail "
                    f"within 1–2 bars and momentum signals reverse immediately. "
                    f"The strategy's filters correctly rejected these low-quality setups."
                )
                recs = [
                    f"Try **VWAPMeanReversion** instead — it's designed for range-bound sessions.",
                    f"Run the same strategy on a trending period (try a date range with a strong market trend).",
                    f"The strategy is working as designed — it's rejecting bad setups.",
                ]
                return "market_was_choppy", why, recs, "medium"

            if bear_pct > 60 and strategy in ("ORBBreakout", "VWAPMeanReversion"):
                why = (
                    f"{sym} was in a **bearish regime for {bear_pct:.0f}% of the period**. "
                    f"{strategy} is a long-biased strategy and is disabled during BEAR_OPEN sessions. "
                    f"The strategy correctly sat out during adverse conditions."
                )
                recs = [
                    f"Try **EMAMomentum** (short side) or **VolumeSpikeReversal** for bear regimes.",
                    f"Test on a different 60-day window with more bullish conditions.",
                    f"The strategy is working correctly — it protects capital in downtrends.",
                ]
                return "regime_mismatch", why, recs, "medium"

        # ── Setup too strict (fit score moderate but still 0 trades) ─────────
        if 0.25 <= fit_score < 0.55:
            why = (
                f"{strategy} on {sym} produced zero trades despite being a reasonable fit "
                f"(fit score {fit_score:.0%}). The entry filters — volume threshold, RSI "
                f"confirmation, and price structure — were all met simultaneously on too few bars. "
                f"This can happen in low-volatility periods even for well-suited symbols."
            )
            recs = [
                f"Run with **auto-tuned parameters** — the config adjuster will loosen filters for {sym}.",
                f"Try a **longer period** (90–180d) to capture more setup opportunities.",
                f"Try **{alt}** which has a higher fit score for {sym}.",
                f"Check if the period includes a holiday-shortened week (low volume).",
            ]
            return "setup_too_strict", why, recs, "low"

        # ── Period too short ──────────────────────────────────────────────────
        period_days = int(period.replace("d", "")) if period.endswith("d") else 60
        if period_days < 45:
            why = (
                f"The {period} period may be too short to capture enough {strategy} setups. "
                f"Some strategies fire 1–3 times per week at best — a short period can easily "
                f"land entirely in a low-activity market phase."
            )
            recs = [
                f"Try **60d or 90d** to capture more market regimes.",
                f"ORB and gap fades need trending days — use a period that includes at least 2–3 trend weeks.",
            ]
            return "data_too_short", why, recs, "low"

        # ── Generic fallback ──────────────────────────────────────────────────
        why = (
            f"{strategy} on {sym} for {period} returned no trades. "
            f"All entry conditions — price structure, volume, RSI, regime — "
            f"were never simultaneously satisfied. This can happen in unusually calm markets "
            f"or when the symbol moves in a way that never triggers the setup."
        )
        recs = [
            f"Try **{alt}** which is better suited to {sym}.",
            f"Extend the period to 90d or 180d.",
            f"Try on a more volatile symbol (TSLA, NVDA, AMD).",
        ]
        return "setup_too_strict", why, recs, "low"


# ── Narrative helpers ─────────────────────────────────────────────────────────

def _orb_on_stable(symbol: str, vol: float, profile: SymbolProfile) -> str:
    orb_width_est = profile.avg_daily_range_pct * 0.2   # rough: ORB ≈ 20% of daily range
    return (
        f"**ORB Breakout requires a wide, decisive opening range** — but {symbol} is a "
        f"low-volatility name with {vol:.1f}% average ATR. "
        f"Its 15-minute opening range is estimated at only ~{orb_width_est:.2f}% wide, "
        f"which is often smaller than a single bar's wick. "
        f"The strategy's fakeout filter (price must close above ORB high + 0.1% buffer) "
        f"correctly rejects these tiny, unreliable breaks. "
        f"In 60 days on {symbol}, you'd statistically expect 0–2 valid ORB setups."
    )


def _vwap_on_volatile(symbol: str, vol: float) -> str:
    return (
        f"**VWAP Mean Reversion requires stable, mean-reverting price action** — but {symbol} "
        f"has {vol:.1f}% ATR, making it a momentum/trend name. "
        f"When {symbol} deviates from VWAP, it frequently continues away from it "
        f"rather than reverting. The strategy's pullback setups turn into continuation moves, "
        f"causing stop-outs before the target is reached."
    )


def _suggest_period(strategy: str, current_period: str) -> str:
    days = int(current_period.replace("d", "")) if current_period.endswith("d") else 60
    if days < 60:
        return "60d"
    if days < 90:
        return "90d"
    return "180d"
