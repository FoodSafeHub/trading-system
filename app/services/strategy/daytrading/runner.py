from __future__ import annotations

import time as _time
from dataclasses import asdict
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from app.services.strategy.daytrading.brain import DayTradingBrain
from app.services.strategy.daytrading.brain.config_adjuster import ConfigAdjuster
from app.services.strategy.daytrading.brain.strategy_selector import StrategySelector
from app.services.strategy.daytrading.brain.symbol_profiles import SymbolAnalyzer
from app.services.strategy.daytrading.brain.trade_explainer import TradeExplainer
from app.services.strategy.daytrading.execution.fill_simulator import FillConfig, FillSimulator
from app.services.strategy.daytrading.market_open import (
    ET,
    apply_choppy_penalty,
    compute_vwap,
    get_spy_regime,
    is_market_open,
    market_status,
    regime_allows_strategy,
)
from app.services.strategy.daytrading.models import DayTradeSignal
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP

# Module-level brain singleton
_brain = DayTradingBrain()


def fetch_intraday(symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Download intraday bars from yfinance, convert to ET timezone."""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC").tz_convert(ET)
    else:
        idx = idx.tz_convert(ET)
    df.index = idx
    return df


def profile_symbol(symbol: str, period: str = "60d") -> dict[str, Any]:
    """
    Download intraday data and compute a full SymbolProfile.
    Returns a dict ready for JSON serialisation.
    """
    df_5m = fetch_intraday(symbol, interval="5m", period=period)
    if df_5m.empty:
        return {"error": f"No data available for {symbol}"}
    profile = SymbolAnalyzer.analyze(df_5m, symbol)
    return profile.to_dict()


def run_signals(
    symbol: str,
    enabled_strategies: list[str] | None = None,
    custom_configs: dict[str, dict] | None = None,
    apply_brain: bool = True,
    today_trades: list[dict] | None = None,
    initial_capital: float = 100_000.0,
    open_positions: int = 0,
) -> dict[str, Any]:
    """
    Fetch latest intraday data, detect regime, run all enabled strategies.
    If apply_brain=True, signals are filtered through the brain pipeline.
    Returns a dict with regime info, brain status, and list of signal dicts.
    """
    status = market_status()
    df_5m = fetch_intraday(symbol, interval="5m", period="5d")
    df_15m = fetch_intraday(symbol, interval="15m", period="60d")

    spy_df = fetch_intraday("SPY", interval="5m", period="2d") if symbol != "SPY" else df_5m
    regime, spy_vs_vwap, gap_pct, gap_type = get_spy_regime(spy_df)

    raw_signals: list[dict] = []

    for strategy in ALL_STRATEGIES:
        if enabled_strategies and strategy.name not in enabled_strategies:
            continue
        if not regime_allows_strategy(regime, strategy.name):
            continue

        cfg = (custom_configs or {}).get(strategy.name)
        try:
            sigs = strategy.generate_signals(df_5m, df_15m, symbol, cfg, regime)
        except Exception:
            sigs = []

        for sig in sigs:
            sig.confidence = apply_choppy_penalty(sig.confidence, regime)
            raw_signals.append(asdict(sig))

    brain_status_dict: dict = {}
    accepted_signals: list[dict] = []
    rejected_signals: list[dict] = []

    if apply_brain and raw_signals:
        brain_status = _brain.build_status(
            df_5m, spy_df if symbol != "SPY" else None,
            today_trades=today_trades,
            initial_capital=initial_capital,
            open_positions=open_positions,
        )
        brain_status_dict = {
            "market_state": brain_status.market_state,
            "state_confidence": brain_status.state_confidence,
            "state_reasons": brain_status.state_reasons,
            "enabled_strategies": brain_status.enabled_strategies,
            "disabled_strategies": brain_status.disabled_strategies,
            "kill_switch": brain_status.kill_switch,
            "kill_switch_reason": brain_status.kill_switch_reason,
            "trades_today": brain_status.trades_today,
            "losses_in_a_row": brain_status.losses_in_a_row,
            "daily_pnl_pct": brain_status.daily_pnl_pct,
            "size_multiplier": brain_status.size_multiplier,
            "routing_summary": brain_status.routing_summary,
        }

        decisions = _brain.filter_signals(
            raw_signals,
            account_state={
                "today_trades": today_trades or [],
                "initial_capital": initial_capital,
                "open_positions": open_positions,
            },
        )
        for dec in decisions:
            sig_out = dict(dec.signal or {})
            sig_out["brain_accepted"] = dec.accepted
            sig_out["brain_reason"] = dec.rejection_reason or dec.explanation
            sig_out["brain_size_multiplier"] = dec.size_multiplier
            sig_out["brain_market_state"] = dec.market_state
            if dec.accepted:
                accepted_signals.append(sig_out)
            else:
                rejected_signals.append(sig_out)
    else:
        accepted_signals = raw_signals

    return {
        "symbol": symbol,
        "regime": regime,
        "spy_vs_vwap_pct": spy_vs_vwap,
        "market_status": status,
        "signal_count": len(accepted_signals),
        "signals": accepted_signals,
        "rejected_signals": rejected_signals,
        "raw_signal_count": len(raw_signals),
        "brain": brain_status_dict,
    }


def run_backtest(
    symbol: str,
    strategy_name: str,
    period: str = "60d",
    initial_capital: float = 10_000.0,
    position_pct: float = 0.95,
    interval: str = "5m",
    custom_config: dict | None = None,
    use_auto_config: bool = True,
    commission_per_share: float = 0.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    """
    Bar-by-bar simulation of a single strategy.
    Returns trade log, performance metrics, symbol profile, and explanation
    if no trades fired.
    """
    strategy = STRATEGY_MAP.get(strategy_name)
    if strategy is None:
        return {"error": f"Unknown strategy: {strategy_name}"}

    df_5m = fetch_intraday(symbol, interval="5m", period=period)
    df_15m = fetch_intraday(symbol, interval="15m", period="730d" if interval == "15m" else period)

    if df_5m.empty:
        return {"error": "No data available"}

    # ── Symbol profile + auto-config ─────────────────────────────────────────
    profile = SymbolAnalyzer.analyze(df_5m, symbol)
    adjustment = ConfigAdjuster.adjust(strategy_name, profile)
    effective_config = adjustment.adjusted if (use_auto_config and not custom_config) else (custom_config or {})

    # ── Strategy selection check ──────────────────────────────────────────────
    # We'll compute the dominant regime first to pass to the selector
    dates_all = sorted(set(df_5m.index.date))
    regime_counts: dict[str, int] = {"BULL_OPEN": 0, "BEAR_OPEN": 0, "CHOPPY": 0}

    fill_sim = FillSimulator(FillConfig(
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
    ))
    trades: list[dict] = []
    equity = initial_capital
    total_commission = 0.0
    total_slippage = 0.0

    for date in dates_all:
        day_5m = df_5m[df_5m.index.date == date]
        day_15m = df_15m[df_15m.index.date == date] if not df_15m.empty else pd.DataFrame()

        if len(day_5m) < 4:
            continue

        # simplified regime: compare open vs prior close
        prev_days = df_5m[df_5m.index.date < date]
        if prev_days.empty:
            regime = "CHOPPY"
        else:
            prior_close = float(prev_days["Close"].iloc[-1])
            open_p = float(day_5m["Open"].iloc[0])
            current = float(day_5m["Close"].iloc[-1])
            vwap_series = compute_vwap(day_5m)
            vwap_val = float(vwap_series.iloc[-1])
            if open_p > prior_close and current > vwap_val:
                regime = "BULL_OPEN"
            elif open_p < prior_close and current < vwap_val:
                regime = "BEAR_OPEN"
            else:
                regime = "CHOPPY"

        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        if not regime_allows_strategy(regime, strategy_name):
            continue

        try:
            signals = strategy.generate_signals(day_5m, day_15m, symbol, effective_config, regime)
        except Exception:
            continue

        for sig in signals:
            if sig.direction == "HOLD":
                continue

            entry = sig.entry_price
            stop = sig.stop_price
            target = sig.target_price
            position_size = (equity * position_pct) / entry
            risk_per_share = abs(entry - stop)

            # simulate outcome using remaining bars on the same day
            sig_time = pd.Timestamp(sig.signal_time)
            future_bars = day_5m[day_5m.index > sig_time]

            outcome = "OPEN"
            exit_price = entry
            exit_time = sig_time
            hold_bars = 0

            for _, bar in future_bars.iterrows():
                hold_bars += 1
                bar_high = float(bar["High"])
                bar_low = float(bar["Low"])

                if sig.direction == "BUY":
                    if bar_low <= stop:
                        exit_price = stop
                        outcome = "STOPPED"
                        exit_time = bar.name
                        break
                    if bar_high >= target:
                        exit_price = target
                        outcome = "TARGET"
                        exit_time = bar.name
                        break
                else:  # SELL SHORT
                    if bar_high >= stop:
                        exit_price = stop
                        outcome = "STOPPED"
                        exit_time = bar.name
                        break
                    if bar_low <= target:
                        exit_price = target
                        outcome = "TARGET"
                        exit_time = bar.name
                        break

                if hold_bars >= strategy.default_config.get("max_hold_bars", 60):
                    exit_price = float(bar["Close"])
                    outcome = "TIME_EXIT"
                    exit_time = bar.name
                    break

            if outcome == "OPEN":
                exit_price = float(future_bars["Close"].iloc[-1]) if not future_bars.empty else entry
                exit_time = future_bars.index[-1] if not future_bars.empty else sig_time
                outcome = "EOD_EXIT"

            # Fill-simulator: apply commission + slippage to both legs
            vol_pct = profile.volatility_pct if profile else 1.0
            entry_fill = fill_sim.fill_entry(sig.direction, entry, position_size, vol_pct)
            exit_fill  = fill_sim.fill_exit(sig.direction, exit_price, position_size, vol_pct)
            fill_summary = fill_sim.summarize_trade(
                symbol, sig.direction, position_size, entry_fill, exit_fill
            )

            pnl = fill_summary.net_pnl
            gross_pnl = fill_summary.gross_pnl
            pnl_pct = pnl / (entry * position_size) * 100 if entry * position_size > 0 else 0.0
            total_commission += fill_summary.total_commission
            total_slippage   += fill_summary.total_slippage
            equity += pnl

            trades.append({
                "date": str(date),
                "symbol": symbol,
                "strategy": strategy_name,
                "direction": sig.direction,
                "entry_price": round(entry_fill.fill_price, 4),
                "exit_price": round(exit_fill.fill_price, 4),
                "stop_price": round(stop, 4),
                "target_price": round(target, 4),
                "entry_time": _ts_str(sig_time),
                "exit_time": _ts_str(exit_time),
                "hold_bars": hold_bars,
                "pnl": round(pnl, 2),
                "gross_pnl": round(gross_pnl, 2),
                "commission": round(fill_summary.total_commission, 4),
                "slippage": round(fill_summary.total_slippage, 4),
                "pnl_pct": round(pnl_pct, 4),
                "outcome": outcome,
                "regime": regime,
                "confidence": sig.confidence,
            })

    result = _compute_metrics(trades, initial_capital, equity, symbol, strategy_name,
                              total_commission=total_commission, total_slippage=total_slippage)

    # ── Attach profile + config adjustment to every result ────────────────────
    result["symbol_profile"] = profile.to_dict()
    result["config_adjustment"] = {
        "changes": adjustment.changes,
        "reason_summary": adjustment.reason_summary,
        "has_changes": adjustment.has_changes,
        "adjusted_config": adjustment.adjusted,
    }

    # ── If 0 trades, produce a full explanation ───────────────────────────────
    if not trades:
        dominant_regime = max(regime_counts, key=regime_counts.get)
        # Use BULL_OPEN for symbol-fit scoring so regime doesn't zero-out
        # strategies that are valid for the symbol but happen to be off
        # during this period's dominant regime. Regime mismatch is explained
        # separately inside TradeExplainer._diagnose.
        selection = StrategySelector.select(symbol, "BULL_OPEN", profile)
        explanation = TradeExplainer.explain_empty_backtest(
            symbol=symbol,
            strategy=strategy_name,
            period=period,
            profile=profile,
            selection=selection,
            regime_counts=regime_counts,
        )
        result["explanation"] = explanation.to_dict()
        result["recommended_alternative"] = selection.primary
        result["selection"] = {
            "primary": selection.primary,
            "secondary": selection.secondary,
            "enabled": selection.enabled,
            "disabled_reasons": selection.disabled_reasons,
            "symbol_fit_scores": selection.symbol_fit_scores,
            "recommendation_text": selection.recommendation_text,
        }

    return result


def _ts_str(ts) -> str:
    """Return a plain tz-naive ISO string (no offset) for consistent parsing."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(ET).tz_localize(None)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _compute_metrics(
    trades: list[dict], initial_capital: float, final_equity: float,
    symbol: str, strategy_name: str,
    total_commission: float = 0.0, total_slippage: float = 0.0,
) -> dict[str, Any]:
    if not trades:
        return {
            "symbol": symbol, "strategy": strategy_name,
            "trades": [], "metrics": {"total_trades": 0},
            "analysis": _empty_analysis(),
        }

    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    total_pnl = df["pnl"].sum()
    win_rate = len(wins) / len(df) * 100
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_curve = [initial_capital]
    running = initial_capital
    for p in df["pnl"]:
        running += p
        equity_curve.append(running)

    peak = initial_capital
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    returns = df["pnl_pct"] / 100
    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0.0

    df["entry_hour"] = pd.to_datetime(df["entry_time"]).dt.hour
    df["entry_dow"] = pd.to_datetime(df["entry_time"]).dt.day_name()
    best_hour = df.groupby("entry_hour")["pnl"].mean().idxmax() if not df.empty else None
    best_dow = df.groupby("entry_dow")["pnl"].mean().idxmax() if not df.empty else None

    # max consecutive losses
    results = (df["pnl"] > 0).tolist()
    max_cl = cur_cl = 0
    for r in results:
        cur_cl = 0 if r else cur_cl + 1
        max_cl = max(max_cl, cur_cl)

    analysis = _build_analysis(df, wins, losses, equity_curve, strategy_name)

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "trades": trades,
        "equity_curve": equity_curve,
        "analysis": analysis,
        "metrics": {
            "total_trades": len(df),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((final_equity - initial_capital) / initial_capital * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_hold_bars": round(df["hold_bars"].mean(), 1),
            "avg_pnl_per_trade": round(df["pnl"].mean(), 2),
            "avg_pnl_pct": round(df["pnl_pct"].mean(), 4),
            "max_consecutive_losses": max_cl,
            "best_hour": int(best_hour) if best_hour is not None else None,
            "best_day_of_week": best_dow,
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 2),
            "gross_pnl": round(df["gross_pnl"].sum() if "gross_pnl" in df.columns else total_pnl, 2),
            "total_commission": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
            "net_pnl": round(total_pnl, 2),
        },
    }


def _empty_analysis() -> dict:
    return {
        "regime_breakdown": {},
        "hourly_performance": {},
        "dow_performance": {},
        "outcome_breakdown": {},
        "win_loss_streaks": {"max_wins": 0, "max_losses": 0},
        "avg_mae": 0.0,
        "avg_mfe": 0.0,
        "expectancy": 0.0,
        "diagnosis": [],
        "strengths": [],
        "weaknesses": [],
    }


def _build_analysis(
    df: pd.DataFrame,
    wins: pd.DataFrame,
    losses: pd.DataFrame,
    equity_curve: list,
    strategy_name: str,
) -> dict:
    """
    Deep trade analysis — surfaces patterns a trader can act on.
    Breaks down performance by regime, time of day, day of week, and outcome type.
    Adds diagnosis: what the strategy is doing right and wrong.
    """
    analysis: dict = {}

    # ── Regime breakdown ─────────────────────────────────────────────────────
    regime_stats: dict = {}
    for regime, grp in df.groupby("regime"):
        w = (grp["pnl"] > 0).sum()
        regime_stats[regime] = {
            "trades": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl": round(grp["pnl_pct"].mean(), 4),
            "total_pnl": round(grp["pnl"].sum(), 2),
        }
    analysis["regime_breakdown"] = regime_stats

    # ── Hourly performance ────────────────────────────────────────────────────
    hourly: dict = {}
    for hour, grp in df.groupby("entry_hour"):
        w = (grp["pnl"] > 0).sum()
        hourly[int(hour)] = {
            "trades": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
        }
    analysis["hourly_performance"] = hourly

    # ── Day of week ───────────────────────────────────────────────────────────
    dow: dict = {}
    for day, grp in df.groupby("entry_dow"):
        w = (grp["pnl"] > 0).sum()
        dow[str(day)] = {
            "trades": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
        }
    analysis["dow_performance"] = dow

    # ── Outcome breakdown ─────────────────────────────────────────────────────
    outcome_stats: dict = {}
    for outcome, grp in df.groupby("outcome"):
        w = (grp["pnl"] > 0).sum()
        outcome_stats[str(outcome)] = {
            "count": int(len(grp)),
            "win_rate": round(w / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
        }
    analysis["outcome_breakdown"] = outcome_stats

    # ── Win/Loss streaks ──────────────────────────────────────────────────────
    results_seq = (df["pnl"] > 0).tolist()
    max_wins = max_losses = cur_w = cur_l = 0
    for r in results_seq:
        if r:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_wins = max(max_wins, cur_w)
        max_losses = max(max_losses, cur_l)
    analysis["win_loss_streaks"] = {"max_wins": max_wins, "max_losses": max_losses}

    # ── Expectancy ────────────────────────────────────────────────────────────
    wr = len(wins) / len(df) if len(df) > 0 else 0
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 0
    expectancy = round(wr * avg_win - (1 - wr) * avg_loss, 4)
    analysis["expectancy"] = expectancy

    # ── Avg win / avg loss sizes ──────────────────────────────────────────────
    analysis["avg_win_pct"] = round(avg_win, 4)
    analysis["avg_loss_pct"] = round(avg_loss, 4)
    analysis["payoff_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    # ── Hold time analysis ────────────────────────────────────────────────────
    analysis["avg_win_hold_bars"] = round(wins["hold_bars"].mean(), 1) if len(wins) > 0 else 0
    analysis["avg_loss_hold_bars"] = round(losses["hold_bars"].mean(), 1) if len(losses) > 0 else 0

    # ── Natural language diagnosis ────────────────────────────────────────────
    diagnosis: list[str] = []
    strengths: list[str] = []
    weaknesses: list[str] = []

    win_rate = wr * 100

    # Win rate assessment
    if win_rate >= 60:
        strengths.append(f"Strong win rate of {win_rate:.1f}% — strategy is selective and accurate.")
    elif win_rate >= 50:
        strengths.append(f"Positive win rate {win_rate:.1f}% — profitable with good R:R.")
    elif win_rate >= 45:
        weaknesses.append(f"Win rate {win_rate:.1f}% is marginal — needs better filters or larger R:R targets.")
    else:
        weaknesses.append(f"Win rate {win_rate:.1f}% is below breakeven threshold — strategy needs rethinking.")

    # Payoff ratio
    pr = analysis["payoff_ratio"]
    if pr >= 2.0:
        strengths.append(f"Excellent payoff ratio {pr:.1f}:1 — winners are much larger than losers.")
    elif pr >= 1.5:
        strengths.append(f"Good payoff ratio {pr:.1f}:1 — risk/reward is working.")
    elif pr >= 1.0:
        diagnosis.append(f"Payoff ratio {pr:.1f}:1 is acceptable but targets could be extended.")
    else:
        weaknesses.append(f"Payoff ratio {pr:.1f}:1 means losers are bigger than winners — tighten stops or widen targets.")

    # Expectancy
    if expectancy > 0.3:
        strengths.append(f"Strong positive expectancy of +{expectancy:.3f}% per trade.")
    elif expectancy > 0:
        diagnosis.append(f"Positive expectancy +{expectancy:.3f}% — marginal edge, look for higher-confidence setups.")
    else:
        weaknesses.append(f"Negative expectancy {expectancy:.3f}% — strategy loses money on average per trade.")

    # Hold time: are wins being cut too early?
    if analysis["avg_win_hold_bars"] < analysis["avg_loss_hold_bars"]:
        weaknesses.append(
            f"Wins held {analysis['avg_win_hold_bars']:.0f} bars vs losses {analysis['avg_loss_hold_bars']:.0f} bars — "
            "cutting winners too early or letting losers run. Consider trailing stops."
        )
    else:
        strengths.append(
            f"Wins held longer ({analysis['avg_win_hold_bars']:.0f} bars) than losses ({analysis['avg_loss_hold_bars']:.0f} bars) — good trade management."
        )

    # Regime performance
    best_regime = max(regime_stats.items(), key=lambda x: x[1]["win_rate"]) if regime_stats else None
    worst_regime = min(regime_stats.items(), key=lambda x: x[1]["win_rate"]) if regime_stats else None
    if best_regime:
        diagnosis.append(f"Best regime: {best_regime[0]} ({best_regime[1]['win_rate']:.0f}% win rate, {best_regime[1]['trades']} trades).")
    if worst_regime and worst_regime[0] != (best_regime[0] if best_regime else None):
        if worst_regime[1]["win_rate"] < 40:
            weaknesses.append(f"Weak in {worst_regime[0]} regime ({worst_regime[1]['win_rate']:.0f}% win rate) — consider disabling in this regime.")

    # Outcome: EOD exits
    eod = outcome_stats.get("EOD_EXIT", {})
    if eod.get("count", 0) > len(df) * 0.3:
        pct = eod["count"] / len(df) * 100
        if eod.get("avg_pnl_pct", 0) < 0:
            weaknesses.append(f"{pct:.0f}% of trades hit end-of-day exit with avg loss — signals are being taken too late in the session.")
        else:
            diagnosis.append(f"{pct:.0f}% of trades resolved at end of day — these are time-forced exits, not technical exits.")

    # Max consecutive losses
    if max_losses >= 5:
        weaknesses.append(f"Max {max_losses} consecutive losses — consider a daily loss limit of 3 stopped trades.")
    elif max_losses <= 2:
        strengths.append(f"Max consecutive losses is only {max_losses} — low drawdown risk.")

    analysis["diagnosis"] = diagnosis
    analysis["strengths"] = strengths
    analysis["weaknesses"] = weaknesses

    return analysis


def run_backtest_all(
    symbol: str,
    period: str = "60d",
    initial_capital: float = 10_000.0,
) -> list[dict[str, Any]]:
    results = []
    for strategy in ALL_STRATEGIES:
        result = run_backtest(symbol, strategy.name, period, initial_capital)
        if "metrics" in result:
            m = result["metrics"]
            results.append({
                "strategy": strategy.name,
                "trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate", 0),
                "profit_factor": m.get("profit_factor", 0),
                "total_pnl": m.get("total_pnl", 0),
                "avg_pnl_pct": m.get("avg_pnl_pct", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                "best_hour": m.get("best_hour"),
                "best_day_of_week": m.get("best_day_of_week"),
            })
    results.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)
    return results


def run_backtest_with_brain(
    symbol: str,
    strategy_name: str,
    period: str = "60d",
    initial_capital: float = 10_000.0,
    position_pct: float = 0.95,
) -> dict[str, Any]:
    """
    Run the same bar-by-bar backtest but filter every signal through the brain.
    Returns both raw and brain-filtered metrics side by side for comparison.
    """
    raw = run_backtest(symbol, strategy_name, period, initial_capital, position_pct)

    strategy = STRATEGY_MAP.get(strategy_name)
    if strategy is None:
        return raw

    df_5m = fetch_intraday(symbol, interval="5m", period=period)
    df_15m = fetch_intraday(symbol, interval="15m", period="730d")
    spy_df = fetch_intraday("SPY", interval="5m", period=period) if symbol != "SPY" else df_5m

    if df_5m.empty:
        return raw

    local_brain = DayTradingBrain()
    filtered_trades: list[dict] = []
    equity = initial_capital
    brain_total_commission = 0.0
    brain_total_slippage = 0.0
    fill_sim = FillSimulator(FillConfig(slippage_bps=2.0, commission_per_share=0.0))
    profile = SymbolAnalyzer.analyze(fetch_intraday(symbol, "5m", "60d"), symbol)
    dates = sorted(set(df_5m.index.date))

    for date in dates:
        day_5m = df_5m[df_5m.index.date == date]
        day_15m = df_15m[df_15m.index.date == date] if not df_15m.empty else pd.DataFrame()
        day_spy = spy_df[spy_df.index.date == date] if not spy_df.empty else pd.DataFrame()

        if len(day_5m) < 4:
            continue

        # ── Full brain classifier (replaces simplified open/close heuristic) ──
        ref_df = day_spy if not day_spy.empty else day_5m
        try:
            from app.services.strategy.daytrading.brain.market_state import (
                classify_market_state, TREND_UP, TREND_DOWN, HIGH_VOL, NEWS_RISK,
            )
            from app.services.strategy.daytrading.brain.strategy_router import route_strategies
            ms = classify_market_state(day_5m, ref_df if not ref_df.empty else None)
            routing = route_strategies(ms)
            local_brain._last_market_state = ms
            local_brain._last_routing = routing

            # Map brain state → legacy regime string for strategy.generate_signals compat
            if ms.state == TREND_UP:
                regime = "BULL_OPEN"
            elif ms.state == TREND_DOWN:
                regime = "BEAR_OPEN"
            elif ms.state in (HIGH_VOL, NEWS_RISK):
                # High-vol / news: skip new entries (too risky)
                continue
            else:
                regime = "CHOPPY"

            # Skip if brain confidence is too low to trade
            if ms.confidence < 0.25:
                regime = "CHOPPY"

            # Brain-based position size multiplier: bull=1.0, bear=0.5, choppy=0.7
            _brain_regime_size = {
                "BULL_OPEN": 1.0,
                "BEAR_OPEN": 0.5,
                "CHOPPY": 0.7,
            }
            brain_day_size_mult = _brain_regime_size.get(regime, 0.7)

        except Exception:
            ms = None
            regime = "CHOPPY"
            brain_day_size_mult = 0.7

        if not regime_allows_strategy(regime, strategy_name):
            continue

        try:
            signals = strategy.generate_signals(day_5m, day_15m, symbol, None, regime)
        except Exception:
            continue

        for sig in signals:
            if sig.direction == "HOLD":
                continue

            sig_dict = asdict(sig)
            decisions = local_brain.filter_signals(
                [sig_dict],
                account_state={
                    "today_trades": [t for t in filtered_trades if t.get("date") == str(date)],
                    "initial_capital": initial_capital,
                    "open_positions": 0,
                },
            )
            if not decisions or not decisions[0].accepted:
                continue

            size_mult = decisions[0].size_multiplier * brain_day_size_mult

            entry = sig.entry_price
            stop = sig.stop_price
            target = sig.target_price
            position_size = (equity * position_pct * size_mult) / entry

            sig_time = pd.Timestamp(sig.signal_time)
            future_bars = day_5m[day_5m.index > sig_time]

            outcome = "OPEN"
            exit_price = entry
            exit_time = sig_time
            hold_bars = 0

            for _, bar in future_bars.iterrows():
                hold_bars += 1
                bar_high = float(bar["High"])
                bar_low = float(bar["Low"])
                if sig.direction == "BUY":
                    if bar_low <= stop:
                        exit_price = stop; outcome = "STOPPED"; exit_time = bar.name; break
                    if bar_high >= target:
                        exit_price = target; outcome = "TARGET"; exit_time = bar.name; break
                else:
                    if bar_high >= stop:
                        exit_price = stop; outcome = "STOPPED"; exit_time = bar.name; break
                    if bar_low <= target:
                        exit_price = target; outcome = "TARGET"; exit_time = bar.name; break
                if hold_bars >= strategy.default_config.get("max_hold_bars", 60):
                    exit_price = float(bar["Close"]); outcome = "TIME_EXIT"; exit_time = bar.name; break

            if outcome == "OPEN":
                exit_price = float(future_bars["Close"].iloc[-1]) if not future_bars.empty else entry
                exit_time = future_bars.index[-1] if not future_bars.empty else sig_time
                outcome = "EOD_EXIT"

            vol_pct = profile.volatility_pct if profile else 1.0
            entry_fill = fill_sim.fill_entry(sig.direction, entry, position_size, vol_pct)
            exit_fill  = fill_sim.fill_exit(sig.direction, exit_price, position_size, vol_pct)
            fill_summary = fill_sim.summarize_trade(
                symbol, sig.direction, position_size, entry_fill, exit_fill
            )

            pnl = fill_summary.net_pnl
            gross_pnl = fill_summary.gross_pnl
            pnl_pct = pnl / (entry * position_size) * 100 if entry * position_size > 0 else 0.0
            brain_total_commission += fill_summary.total_commission
            brain_total_slippage   += fill_summary.total_slippage
            equity += pnl

            filtered_trades.append({
                "date": str(date),
                "symbol": symbol,
                "strategy": strategy_name,
                "direction": sig.direction,
                "entry_price": round(entry_fill.fill_price, 4),
                "exit_price": round(exit_fill.fill_price, 4),
                "stop_price": round(stop, 4),
                "target_price": round(target, 4),
                "entry_time": _ts_str(sig_time),
                "exit_time": _ts_str(exit_time),
                "hold_bars": hold_bars,
                "pnl": round(pnl, 2),
                "gross_pnl": round(gross_pnl, 2),
                "commission": round(fill_summary.total_commission, 4),
                "slippage": round(fill_summary.total_slippage, 4),
                "pnl_pct": round(pnl_pct, 4),
                "outcome": outcome,
                "regime": regime,
                "confidence": sig.confidence,
                "brain_size_multiplier": size_mult,
                "market_state": ms.state if ms else "UNKNOWN",
                "market_state_confidence": round(ms.confidence, 3) if ms else 0.0,
            })

    brain_result = _compute_metrics(filtered_trades, initial_capital, equity, symbol, strategy_name,
                                    total_commission=brain_total_commission, total_slippage=brain_total_slippage)

    # Combine both into a comparison dict
    raw_m = raw.get("metrics", {})
    brain_m = brain_result.get("metrics", {})
    comparison = {}
    for key in ["total_trades", "win_rate", "profit_factor", "total_pnl", "max_drawdown_pct", "sharpe_ratio"]:
        comparison[key] = {
            "raw": raw_m.get(key, 0),
            "brain": brain_m.get(key, 0),
        }

    # Raw vs Simulated P&L table (fill simulation impact)
    raw_net = raw_m.get("net_pnl", raw_m.get("total_pnl", 0))
    brain_net = brain_m.get("net_pnl", brain_m.get("total_pnl", 0))
    fill_impact = {
        "raw_net_pnl": raw_net,
        "brain_net_pnl": brain_net,
        "brain_total_commission": round(brain_total_commission, 2),
        "brain_total_slippage": round(brain_total_slippage, 2),
        "brain_gross_pnl": brain_m.get("gross_pnl", brain_net),
        "pnl_reduction_pct": round(
            (brain_m.get("gross_pnl", brain_net) - brain_net) / abs(brain_m.get("gross_pnl", brain_net)) * 100, 1
        ) if brain_m.get("gross_pnl", brain_net) != 0 else 0.0,
    }

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "raw": raw,
        "brain_filtered": brain_result,
        "comparison": comparison,
        "fill_impact": fill_impact,
    }


def run_scan(
    symbols: list[str],
    enabled_strategies: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Scan multiple symbols for current signals."""
    all_signals: list[dict] = []
    for symbol in symbols:
        _time.sleep(0.5)
        try:
            result = run_signals(symbol, enabled_strategies)
            all_signals.extend(result.get("signals", []))
        except Exception:
            continue
    all_signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return all_signals
