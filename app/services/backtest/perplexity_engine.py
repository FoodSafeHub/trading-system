from __future__ import annotations

"""
Backtester for Perplexity swing trading strategies.
Runs each strategy bar-by-bar with no lookahead bias.
Fills at the next bar's open price.
Respects stop_price and target_price if set.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from app.services.market_data.provider import get_ohlcv
from app.services.market_regime import MarketRegime, get_regime_risk_caps
from app.services.performance_breakdown import bucket_atr_pct
from app.services.performance_metrics import (
    calc_cagr,
    calc_total_return,
    calculate_performance_from_pairs,
    pair_trade_records,
)
from app.services.perplexity.suitability import load_suitability_config
from app.services.risk.position_sizer import calculate_position_size
from app.services.strategy.perplexity.base import PerplexityStrategy


def calc_total_return(equity_curve: list) -> float:
    """(E_final / E_initial) - 1, as a percentage."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    e_initial = equity_curve[0]["equity"]
    e_final   = equity_curve[-1]["equity"]
    if e_initial == 0:
        return 0.0
    return round((e_final / e_initial - 1) * 100, 2)


def calc_cagr(equity_curve: list) -> float:
    """Compound annual growth rate assuming 252 trading days/year, as a percentage."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    e_initial = equity_curve[0]["equity"]
    e_final   = equity_curve[-1]["equity"]
    if e_initial <= 0 or e_final <= 0:
        return 0.0
    years = len(equity_curve) / 252
    if years <= 0:
        return 0.0
    return round(((e_final / e_initial) ** (1.0 / years) - 1) * 100, 2)


@dataclass
class PerplexityBacktestResult:
    strategy_name: str
    symbol: str
    period: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return_pct: float     # equity-curve total return %
    cagr: float                 # compound annual growth rate %
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float        # gross wins / gross losses
    capital_employed: float     # sum of all buy-side position costs
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float
    expectancy_r: Optional[float]
    average_holding_days: float
    average_r_multiple: Optional[float]
    trade_pairs: List[dict] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _current_atr(df: pd.DataFrame, period: int = 14) -> float:
    atr = _atr_series(df, period)
    last = atr.iloc[-1]
    return float(last) if not pd.isna(last) else float(df["Close"].iloc[-1]) * 0.02


def _regime_from_spy(spy_close: pd.Series, as_of_date: pd.Timestamp) -> MarketRegime:
    """
    Compute market regime from SPY SMA200 as of a specific date.
    Uses the SPY close series pre-fetched for the full backtest period.
    Falls back to BULL if SPY data doesn't reach that date.
    """
    history = spy_close.loc[spy_close.index <= as_of_date]
    if len(history) < 50:
        return MarketRegime.BULL
    sma200 = float(history.rolling(200).mean().iloc[-1]) if len(history) >= 200 else float(history.mean())
    c = float(history.iloc[-1])
    if c < sma200 * 0.80:
        return MarketRegime.DEEP_BEAR
    if c < sma200:
        return MarketRegime.BEAR
    return MarketRegime.BULL


def run_perplexity_backtest(
    strategy: PerplexityStrategy,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
    risk_pct_per_trade: float = 0.01,
    max_position_pct: float = 0.20,        # never put more than 20% of capital in one trade
    position_pct: float = 0.0,             # >0 = fixed % of capital per trade (overrides risk sizing)
) -> PerplexityBacktestResult:
    df_full = get_ohlcv(symbol, period=period)
    if df_full.empty or len(df_full) < 60:
        raise ValueError(f"Not enough data for {symbol} (need at least 60 bars)")

    # Fetch SPY for regime detection — always 10y so historical backtests work.
    # If the symbol IS SPY, reuse its own data to avoid a redundant download.
    try:
        spy_raw = df_full["Close"] if symbol.upper() == "SPY" else get_ohlcv("SPY", period="10y")["Close"]
    except Exception:
        spy_raw = df_full["Close"]   # fallback: use symbol itself as regime proxy

    # Warm up 210 bars so SMA(200) is valid from the first active bar.
    # For very short datasets (< 350 bars), cap at 60% so some trading still occurs.
    if len(df_full) >= 350:
        lookback = 210
    else:
        lookback = min(210, max(60, len(df_full) * 6 // 10))

    dates = [str(d)[:10] for d in df_full.index]
    capital = initial_capital
    position = 0.0
    position_cost = 0.0
    entry_price_rec: Optional[float] = None   # recorded fill price for trailing stop math
    initial_risk: Optional[float] = None       # entry - original stop (1R in dollars/share)
    entry_stop: Optional[float] = None
    entry_target: Optional[float] = None
    open_risk_usd: float = 0.0
    trades: List[dict] = []
    equity_curve: List[dict] = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_returns: List[float] = []
    prev_equity = initial_capital

    for i in range(lookback, len(df_full)):
        df_slice = df_full.iloc[:i]
        current_close = float(df_full["Close"].iloc[i])
        fill_price    = float(df_full["Open"].iloc[i])   # always valid — i is bounded by range()
        today = dates[i]

        # ── Check stop / target on open if in position ──────────
        if position > 0 and entry_stop is not None:
            open_price = float(df_full["Open"].iloc[i])
            # Gap down through stop
            if open_price <= entry_stop:
                proceeds = open_price * position
                pnl = proceeds - position_cost
                capital += proceeds
                trades.append(_trade("SELL (stop)", today, open_price, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None
            # Gap up through target
            elif entry_target and open_price >= entry_target:
                proceeds = open_price * position
                pnl = proceeds - position_cost
                capital += proceeds
                trades.append(_trade("SELL (target)", today, open_price, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None

        # ── Trailing stop: once price moves 1R, trail at 0.5R below high-water mark ──
        # This locks in profit on mid-term trades without exiting too early on momentum.
        if position > 0 and entry_stop is not None and entry_price_rec and initial_risk:
            high_today = float(df_full["High"].iloc[i])
            profit_per_share = high_today - entry_price_rec
            if profit_per_share >= initial_risk:
                # Move stop to: high_water_mark - 0.5R (trail tightly after 1R gain)
                trail_stop = high_today - 0.5 * initial_risk
                if trail_stop > entry_stop:
                    entry_stop = trail_stop

        # ── Intraday stop / target on close ─────────────────────
        if position > 0 and entry_stop is not None:
            low_today = float(df_full["Low"].iloc[i])
            high_today = float(df_full["High"].iloc[i])
            if low_today <= entry_stop:
                pnl = entry_stop * position - position_cost
                capital += entry_stop * position
                trades.append(_trade("SELL (stop)", today, entry_stop, position,
                                     entry_stop * position, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None
            elif entry_target and high_today >= entry_target:
                pnl = entry_target * position - position_cost
                capital += entry_target * position
                trades.append(_trade("SELL (target)", today, entry_target, position,
                                     entry_target * position, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None

        # ── Run strategy signal ──────────────────────────────────
        regime = _regime_from_spy(spy_raw, df_full.index[i - 1])
        regime_caps = get_regime_risk_caps(regime)
        suitability_config = None
        try:
            suitability_config = load_suitability_config()
        except Exception:
            suitability_config = None

        volatility_bucket = bucket_atr_pct(_current_atr(df_slice))
        if position == 0:
            try:
                sig = strategy.run(
                    symbol,
                    df_slice,
                    regime=regime,
                    volatility_bucket=volatility_bucket,
                    suitability_config=suitability_config,
                )
            except Exception:
                sig = None

            if sig and sig.direction == "BUY":
                if position_pct > 0:
                    # Fixed % of capital per trade — deploys position_pct of current capital
                    alloc = capital * min(position_pct, max_position_pct)
                    qty = alloc / fill_price if fill_price > 0 else 0.0
                    trade_risk = (fill_price - sig.stop_price) * qty if sig.stop_price else 0.0
                elif sig.stop_price and sig.stop_price < fill_price:
                    # Risk-based sizing: use regime-specific risk caps
                    sz = calculate_position_size(
                        symbol=symbol,
                        entry_price=fill_price,
                        stop_price=sig.stop_price,
                        account_value=capital,
                        risk_pct_per_trade=regime_caps["risk_pct_per_trade"],
                        max_position_size_usd=capital * max_position_pct,
                        max_account_risk_pct=regime_caps["max_account_risk_pct"],
                        current_open_risk_usd=open_risk_usd,
                    )
                    qty = sz.shares if sz.viable else 0.0
                    trade_risk = sz.risk_amount if sz.viable else 0.0
                else:
                    # No stop defined — fall back to investing 10% of capital
                    qty = (capital * 0.10) / fill_price if fill_price > 0 else 0
                    trade_risk = 0.0

                qty = round(qty, 6)
                if qty >= 0.001 and fill_price * qty <= capital:
                    cost = fill_price * qty
                    capital -= cost
                    position = qty
                    position_cost = cost
                    open_risk_usd += trade_risk
                    entry_stop      = sig.stop_price
                    entry_target    = sig.target_price
                    entry_price_rec = fill_price
                    initial_risk    = (fill_price - sig.stop_price) if sig.stop_price else None
                    atr_pct = round(_current_atr(df_slice) / fill_price * 100, 3) if fill_price else 0.0
                    trades.append({
                        "date": today, "side": "BUY",
                        "price": round(fill_price, 2), "quantity": round(qty, 4),
                        "value": round(cost, 2), "pnl": None,
                        "stop": round(sig.stop_price, 2) if sig.stop_price else None,
                        "target": round(sig.target_price, 2) if sig.target_price else None,
                        "confidence": sig.confidence,
                        "reason": sig.reason,
                        "risk_usd": round(trade_risk, 2),
                        "regime": regime.value,
                        "atr_pct": atr_pct,
                        "volatility_bucket": volatility_bucket,
                        "strategy_name": strategy.name,
                        "symbol": symbol,
                    })

        elif position > 0:
            try:
                sig = strategy.run(
                    symbol,
                    df_slice,
                    regime=regime,
                    volatility_bucket=volatility_bucket,
                    suitability_config=suitability_config,
                )
            except Exception:
                sig = None

            if sig and sig.direction == "SELL":
                proceeds = fill_price * position
                pnl = proceeds - position_cost
                capital += proceeds
                trades.append({
                    "date": today, "side": "SELL",
                    "price": round(fill_price, 2), "quantity": position,
                    "value": round(proceeds, 2), "pnl": round(pnl, 2),
                    "stop": None, "target": None,
                    "confidence": sig.confidence if sig else None,
                    "reason": sig.reason if sig else "",
                    "risk_usd": None,
                    "regime": regime.value,
                    "strategy_name": strategy.name,
                    "symbol": symbol,
                })
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None

        # ── Mark-to-market ───────────────────────────────────────
        equity = capital + position * current_close
        equity_curve.append({"date": today, "equity": round(equity, 2)})

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(ret)
        prev_equity = equity

    # Close any open position at last close
    if position > 0:
        last_price = float(df_full["Close"].iloc[-1])
        proceeds = last_price * position
        pnl = proceeds - position_cost
        capital += proceeds
        trades.append(_trade("SELL (close)", dates[-1], last_price, position, proceeds, pnl))

    final_capital = capital
    total_pnl = final_capital - initial_capital

    sell_trades = [t for t in trades if "SELL" in t["side"]]
    buy_trades  = [t for t in trades if t["side"] == "BUY"]
    winning = [t for t in sell_trades if (t.get("pnl") or 0) > 0]
    losing  = [t for t in sell_trades if (t.get("pnl") or 0) <= 0]
    win_rate = len(winning) / len(sell_trades) * 100 if sell_trades else 0.0
    # total_trades = round trips (sell count), not raw trade records (buy+sell)
    total_completed_trades = len(sell_trades)

    capital_employed = sum(t["value"] for t in buy_trades) if buy_trades else 0.0
    total_return = calc_total_return(equity_curve)
    cagr         = calc_cagr(equity_curve)

    gross_wins   = sum(t["pnl"] for t in winning)
    gross_losses = abs(sum(t["pnl"] for t in losing))
    # None = no losing trades at all (perfect record) — displayed as "—" in UI
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None

    sharpe = None
    if len(daily_returns) > 1:
        avg = statistics.mean(daily_returns)
        std = statistics.stdev(daily_returns)
        if std > 0:
            sharpe = round((avg / std) * (252 ** 0.5), 2)

    trade_pairs = pair_trade_records(trades)
    trade_perf = calculate_performance_from_pairs(
        strategy_name=strategy.name,
        symbol=symbol,
        period=period,
        initial_capital=initial_capital,
        trade_pairs=trade_pairs,
        equity_curve=equity_curve,
    )

    return PerplexityBacktestResult(
        strategy_name=strategy.name,
        symbol=symbol,
        period=period,
        start_date=dates[lookback] if dates else "",
        end_date=dates[-1] if dates else "",
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_pnl=round(total_pnl, 2),
        capital_employed=round(capital_employed, 2),
        total_return_pct=total_return,
        cagr=cagr,
        total_trades=total_completed_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate_pct=round(win_rate, 2),
        profit_factor=profit_factor,
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=sharpe,
        avg_win_pct=trade_perf.avg_win_pct,
        avg_loss_pct=trade_perf.avg_loss_pct,
        expectancy_pct=trade_perf.expectancy_pct,
        expectancy_r=trade_perf.expectancy_r,
        average_holding_days=trade_perf.average_holding_days,
        average_r_multiple=trade_perf.average_r_multiple,
        trade_pairs=trade_pairs,
        trades=trades,
        equity_curve=equity_curve,
    )


def _trade(side: str, date: str, price: float, qty: float, value: float, pnl: float) -> dict:
    return {
        "date": date, "side": side,
        "price": round(price, 2), "quantity": qty,
        "value": round(value, 2), "pnl": round(pnl, 2),
        "stop": None, "target": None, "confidence": None, "reason": "", "risk_usd": None,
    }
