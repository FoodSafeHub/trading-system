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

from app.services.backtest.costs import CostModel
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

# Fallback used only when the strategy doesn't declare a max_hold_bars budget.
# Engine never silently overrides a strategy that does declare one.
_DEFAULT_MAX_HOLD_BARS = 60

# ── Short-position convention ────────────────────────────────────────────────
# `position` carries the SIGNED share count: positive = long, negative = short.
# `abs(position)` is the number of shares borrowed and sold-to-open. For a
# short trade, `position_cost` is the dollars RECEIVED at the sell-to-open
# (positive number); P&L at cover is `position_cost - cover_value - commission`.
#
# This is MINIMAL short support for daily-candlestick momentum strategies that
# need a short-entry path (Daily_NR_Breakout, Daily_Engulfing_Volume,
# Daily_Three_Bar_Push, Daily_Hammer_Star). It does NOT model:
#   * margin requirements / Reg-T / portfolio margin
#   * stock-borrow cost (HTB rates)
#   * short-sale restrictions / locate failures
#   * uptick / SSR rules
# Those are intentionally out of scope. For full short modeling, build a
# dedicated execution layer; do not extend this engine.


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
    Point-in-time market regime from SPY close series.

    Delegates to the live :func:`detect_market_regime` (the same function the
    scheduler / scanner / live signal path call via ``get_current_regime``) so
    the backtest cannot disagree with live on the same SPY history. The
    previous inline classifier diverged from live on two material points:
        * BULL gate omitted the SMA(50) >= SMA(200) cross check, so the
          "death-cross" window above SMA200 was labelled BULL in backtest
          and BEAR in live.
        * DEEP_BEAR threshold was ``close < 0.80 * SMA200`` instead of
          ``close < SMA200 AND drawdown from 52w high <= -20%`` — different
          condition, different label.

    Build a minimal ``DataFrame`` with a ``Close`` column so the live function
    can consume it; the bar warm-up requirements (200 closes) are inherited.

    Falls back to BULL only when the SPY history hasn't accumulated enough
    bars yet (same fallback the prior implementation used).
    """
    history = spy_close.loc[spy_close.index <= as_of_date]
    if len(history) < 200:
        # Live function requires 200 bars; for early backtest dates we keep the
        # legacy BULL fallback rather than throwing — matches prior behaviour.
        return MarketRegime.BULL
    # Import here to avoid a circular import at module load (market_regime
    # imports get_ohlcv which transitively imports backtest helpers in tests).
    from app.services.market_regime import detect_market_regime
    return detect_market_regime(pd.DataFrame({"Close": history}))


def run_perplexity_backtest(
    strategy: PerplexityStrategy,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
    risk_pct_per_trade: float = 0.01,
    max_position_pct: float = 0.20,        # never put more than 20% of capital in one trade
    position_pct: float = 0.0,             # >0 = fixed % of capital per trade (overrides risk sizing)
    df_full: Optional[pd.DataFrame] = None,   # inject pre-fetched OHLCV to skip the download
    spy_close: Optional[pd.Series] = None,    # inject pre-fetched SPY Close for regime detection
    cost_model: Optional[CostModel] = None,   # opt-in slippage/commission; None == identity (no behaviour change)
) -> PerplexityBacktestResult:
    # Callers that compare many strategies on one symbol can fetch the bars once
    # and pass them in (df_full/spy_close), avoiding a redundant Yahoo download per
    # strategy. When omitted, behave exactly as before.
    if df_full is None:
        df_full = get_ohlcv(symbol, period=period)
    if df_full.empty or len(df_full) < 60:
        if df_full.empty:
            raise ValueError(f"No data returned for {symbol} over period {period}")
        listed = str(df_full.index[0])[:10]
        raise ValueError(
            f"{symbol} has only {len(df_full)} trading days of history (listed ~{listed}); "
            f"this strategy needs at least 60 bars. Pick a longer-established stock."
        )

    # Fetch SPY for regime detection — always 10y so historical backtests work.
    # If the symbol IS SPY, reuse its own data to avoid a redundant download.
    if spy_close is not None:
        spy_raw = spy_close
    else:
        try:
            spy_raw = df_full["Close"] if symbol.upper() == "SPY" else get_ohlcv("SPY", period="10y")["Close"]
        except Exception:
            spy_raw = df_full["Close"]   # fallback: use symbol itself as regime proxy

    # Point-in-time momentum-regime support for momentum strategies (daily
    # candle patterns). The strategies' default _momentum_snapshot calls the
    # LIVE get_momentum_regime() which would leak today's tape into every
    # historical bar. We pre-fetch the index/VIX/breadth series ONCE here,
    # then compute a per-bar snapshot via get_momentum_regime_at() and pass
    # it through to the strategy via kwargs["momentum_snapshot"].
    #
    # Indian symbols gate on Nifty 50 + India VIX (per _momentum_snapshot's
    # market routing). Fetch both market sets and pick by symbol.
    _is_india = False
    try:
        from app.services.markets import is_india_symbol as _is_india_symbol
        _is_india = bool(_is_india_symbol(symbol))
    except Exception:
        pass

    _mom_index_close: Optional[pd.Series] = None
    _mom_vix_close: Optional[pd.Series] = None
    _mom_breadth: Optional[dict[str, pd.Series]] = None
    try:
        from app.services.market_regime_advanced import (
            _MARKET_CFG, get_momentum_regime_at,
        )
        _mom_market = "india" if _is_india else "us"
        _mom_cfg = _MARKET_CFG[_mom_market]
        # Reuse already-fetched SPY series for the US case.
        if _mom_market == "us":
            _mom_index_close = spy_raw
        else:
            try:
                _mom_index_close = get_ohlcv(_mom_cfg["index"], period="10y")["Close"]
            except Exception:
                _mom_index_close = None
        try:
            _mom_vix_close = get_ohlcv(_mom_cfg["vix"], period="10y")["Close"]
        except Exception:
            _mom_vix_close = None
        # Breadth is heavy: 10 daily fetches per backtest. Skip it on the
        # backtest hot-path -- the classifier degrades gracefully when
        # breadth is None (same code path as live when ^SPXA50R is down).
        _mom_breadth = None
    except Exception:
        get_momentum_regime_at = None   # type: ignore[assignment]

    # Warm up 210 bars so SMA(200) is valid from the first active bar.
    # For very short datasets (< 350 bars), cap at 60% so some trading still occurs.
    if len(df_full) >= 350:
        lookback = 210
    else:
        lookback = min(210, max(60, len(df_full) * 6 // 10))

    # Honor the strategy's own max_hold_bars budget. Strategies declare this in
    # their config dict (e.g. RSI-2: 10, Fib pullback: 8). The engine previously
    # ignored it — trades sat until stop/target/SELL fired, which produced
    # systematically worse results for capitulation/mean-reversion strategies
    # that depend on a tight time budget.
    try:
        max_hold_bars = int(getattr(strategy, "config", {}).get("max_hold_bars", _DEFAULT_MAX_HOLD_BARS))
    except Exception:
        max_hold_bars = _DEFAULT_MAX_HOLD_BARS
    if max_hold_bars <= 0:
        max_hold_bars = _DEFAULT_MAX_HOLD_BARS

    dates = [str(d)[:10] for d in df_full.index]
    capital = initial_capital
    position = 0.0
    position_cost = 0.0
    entry_price_rec: Optional[float] = None   # recorded fill price for trailing stop math
    initial_risk: Optional[float] = None       # entry - original stop (1R in dollars/share)
    entry_stop: Optional[float] = None
    entry_target: Optional[float] = None
    open_risk_usd: float = 0.0
    bars_held: int = 0                         # bars since current position opened
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

        # Count a holding day BEFORE intra-bar exit checks so the bar a stop
        # fires on is counted. Strategies budget in bars-of-exposure, not
        # bars-survived. Symmetric across long (>0) and short (<0) positions.
        if position != 0:
            bars_held += 1

        # ── Check stop / target on open if in position ──────────
        if position > 0 and entry_stop is not None:
            open_price = float(df_full["Open"].iloc[i])
            # Gap down through stop
            if open_price <= entry_stop:
                sell_px = open_price if cost_model is None else cost_model.apply_sell(open_price)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                pnl = proceeds - commission - position_cost
                capital += proceeds - commission
                trades.append(_trade("SELL (stop)", today, sell_px, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0
            # Gap up through target
            elif entry_target and open_price >= entry_target:
                sell_px = open_price if cost_model is None else cost_model.apply_sell(open_price)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                pnl = proceeds - commission - position_cost
                capital += proceeds - commission
                trades.append(_trade("SELL (target)", today, sell_px, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0
        # Short-side mirror: stop is ABOVE entry, target is BELOW entry.
        elif position < 0 and entry_stop is not None:
            open_price = float(df_full["Open"].iloc[i])
            shares = abs(position)
            # Gap UP through stop -> buy-to-cover at open
            if open_price >= entry_stop:
                cover_px = open_price if cost_model is None else cost_model.apply_buy(open_price)
                cover_value = cover_px * shares
                commission = 0.0 if cost_model is None else cost_model.entry_commission(shares, cover_value)
                pnl = position_cost - cover_value - commission
                capital -= cover_value + commission
                trades.append(_trade("COVER (stop)", today, cover_px, shares, cover_value, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0
            # Gap DOWN through target -> buy-to-cover at open
            elif entry_target and open_price <= entry_target:
                cover_px = open_price if cost_model is None else cost_model.apply_buy(open_price)
                cover_value = cover_px * shares
                commission = 0.0 if cost_model is None else cost_model.entry_commission(shares, cover_value)
                pnl = position_cost - cover_value - commission
                capital -= cover_value + commission
                trades.append(_trade("COVER (target)", today, cover_px, shares, cover_value, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0

        # ── Trailing stop: once price moves 1R, trail at 0.5R below high-water mark ──
        # No lookahead: high-water mark = high through the PRIOR bar (the most
        # recent fully-closed bar). The previous version used `iloc[i]`, which
        # could ratchet the stop using THIS bar's high and then fire the stop
        # on the same bar's low — a same-bar future read.
        if position > 0 and entry_stop is not None and entry_price_rec and initial_risk and i > 0:
            high_prior = float(df_full["High"].iloc[i - 1])
            profit_per_share = high_prior - entry_price_rec
            if profit_per_share >= initial_risk:
                trail_stop = high_prior - 0.5 * initial_risk
                if trail_stop > entry_stop:
                    entry_stop = trail_stop
        # Short-side mirror: ratchet stop DOWN once price has moved 1R against
        # the short (i.e. price has fallen 1R below entry).
        elif position < 0 and entry_stop is not None and entry_price_rec and initial_risk and i > 0:
            low_prior = float(df_full["Low"].iloc[i - 1])
            profit_per_share = entry_price_rec - low_prior   # gain for short = entry - low
            if profit_per_share >= initial_risk:
                trail_stop = low_prior + 0.5 * initial_risk
                if trail_stop < entry_stop:
                    entry_stop = trail_stop

        # ── Intraday stop / target on close ─────────────────────
        if position > 0 and entry_stop is not None:
            low_today = float(df_full["Low"].iloc[i])
            high_today = float(df_full["High"].iloc[i])
            if low_today <= entry_stop:
                sell_px = entry_stop if cost_model is None else cost_model.apply_sell(entry_stop)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                pnl = proceeds - commission - position_cost
                capital += proceeds - commission
                trades.append(_trade("SELL (stop)", today, sell_px, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0
            elif entry_target and high_today >= entry_target:
                sell_px = entry_target if cost_model is None else cost_model.apply_sell(entry_target)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                pnl = proceeds - commission - position_cost
                capital += proceeds - commission
                trades.append(_trade("SELL (target)", today, sell_px, position, proceeds, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0
        # Short-side mirror.
        elif position < 0 and entry_stop is not None:
            low_today = float(df_full["Low"].iloc[i])
            high_today = float(df_full["High"].iloc[i])
            shares = abs(position)
            if high_today >= entry_stop:
                cover_px = entry_stop if cost_model is None else cost_model.apply_buy(entry_stop)
                cover_value = cover_px * shares
                commission = 0.0 if cost_model is None else cost_model.entry_commission(shares, cover_value)
                pnl = position_cost - cover_value - commission
                capital -= cover_value + commission
                trades.append(_trade("COVER (stop)", today, cover_px, shares, cover_value, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0
            elif entry_target and low_today <= entry_target:
                cover_px = entry_target if cost_model is None else cost_model.apply_buy(entry_target)
                cover_value = cover_px * shares
                commission = 0.0 if cost_model is None else cost_model.entry_commission(shares, cover_value)
                pnl = position_cost - cover_value - commission
                capital -= cover_value + commission
                trades.append(_trade("COVER (target)", today, cover_px, shares, cover_value, pnl))
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0

        # ── Time exit: max_hold_bars budget exceeded ─────────────
        # Closes at this bar's CLOSE to mirror strategy-driven SELL convention
        # (the engine's existing SELL block uses fill_price = open, but a time
        # exit conceptually fires AT the close of the budget bar — same as how
        # the day-trading simulator treats max_hold).
        if position > 0 and bars_held >= max_hold_bars:
            sell_px = current_close if cost_model is None else cost_model.apply_sell(current_close)
            proceeds = sell_px * position
            commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
            pnl = proceeds - commission - position_cost
            capital += proceeds - commission
            trades.append(_trade("SELL (time)", today, sell_px, position, proceeds, pnl))
            position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0
        elif position < 0 and bars_held >= max_hold_bars:
            shares = abs(position)
            cover_px = current_close if cost_model is None else cost_model.apply_buy(current_close)
            cover_value = cover_px * shares
            commission = 0.0 if cost_model is None else cost_model.entry_commission(shares, cover_value)
            pnl = position_cost - cover_value - commission
            capital -= cover_value + commission
            trades.append(_trade("COVER (time)", today, cover_px, shares, cover_value, pnl))
            position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0

        # ── Run strategy signal ──────────────────────────────────
        regime = _regime_from_spy(spy_raw, df_full.index[i - 1])
        regime_caps = get_regime_risk_caps(regime)
        suitability_config = None
        try:
            suitability_config = load_suitability_config()
        except Exception:
            suitability_config = None

        volatility_bucket = bucket_atr_pct(_current_atr(df_slice))

        # Point-in-time momentum-regime snapshot for daily-candle pattern
        # strategies. Pass it through kwargs; non-momentum strategies ignore
        # the kwarg (their run() signature is **kwargs-tolerant via the base
        # class). NaN-safe: failures return None and the strategies fall back
        # to their existing live snapshot path (which is what we replaced).
        _mom_snapshot_today = None
        if get_momentum_regime_at is not None and _mom_index_close is not None:
            try:
                _mom_snapshot_today = get_momentum_regime_at(
                    as_of_date=df_full.index[i - 1],
                    index_close=_mom_index_close,
                    vix_close=_mom_vix_close,
                    breadth_close_by_symbol=_mom_breadth,
                    market="india" if _is_india else "us",
                )
            except Exception:
                _mom_snapshot_today = None
        if position == 0:
            try:
                sig = strategy.run(
                    symbol,
                    df_slice,
                    regime=regime,
                    volatility_bucket=volatility_bucket,
                    suitability_config=suitability_config,
                    momentum_snapshot=_mom_snapshot_today,
                )
            except Exception:
                sig = None

            # ── SHORT entry: SELL signal while flat ─────────────────
            # For a short trade the stop must be ABOVE entry (price moves up
            # against us = loss). If the signal lacks that geometry it's not a
            # valid short — skip rather than guessing. Daily-candle momentum
            # patterns are the primary customers.
            if (
                sig and sig.direction == "SELL"
                and sig.stop_price and sig.stop_price > fill_price
            ):
                # Risk-based sizing: calculate_position_size is long-only
                # (rejects stop >= entry). For shorts we reuse the same
                # formula inline: risk_dollars / risk_per_share, capped by
                # max position notional.
                if position_pct > 0:
                    alloc = capital * min(position_pct, max_position_pct)
                    qty = alloc / fill_price if fill_price > 0 else 0.0
                    trade_risk = (sig.stop_price - fill_price) * qty
                else:
                    risk_per_share = sig.stop_price - fill_price
                    risk_dollars = capital * regime_caps["risk_pct_per_trade"]
                    # Don't exceed open-risk budget for the account
                    risk_budget_remaining = max(
                        0.0,
                        capital * regime_caps["max_account_risk_pct"] - open_risk_usd,
                    )
                    risk_dollars = min(risk_dollars, risk_budget_remaining)
                    qty = risk_dollars / risk_per_share if risk_per_share > 0 else 0.0
                    # Notional cap
                    max_qty_by_notional = (capital * max_position_pct) / fill_price if fill_price > 0 else 0.0
                    qty = min(qty, max_qty_by_notional)
                    trade_risk = risk_per_share * qty

                qty = round(qty, 6)
                # sell-to-open: cost_model applies sell-side slippage to fill
                sell_open_px = fill_price if cost_model is None else cost_model.apply_sell(fill_price)
                if qty >= 0.001:
                    proceeds = sell_open_px * qty
                    commission = 0.0 if cost_model is None else cost_model.entry_commission(qty, proceeds)
                    # Cash flow on short open: receive proceeds, pay commission.
                    # NOTE: we don't reserve a margin requirement — minimal model.
                    capital += proceeds - commission
                    position = -qty                       # signed: short
                    position_cost = proceeds - commission # dollars NET received at open
                    open_risk_usd += trade_risk
                    entry_stop      = sig.stop_price       # above entry
                    entry_target    = sig.target_price     # below entry
                    entry_price_rec = sell_open_px
                    initial_risk    = (sig.stop_price - sell_open_px)  # 1R in $/share
                    bars_held = 0
                    atr_pct = round(_current_atr(df_slice) / fill_price * 100, 3) if fill_price else 0.0
                    trades.append({
                        "date": today, "side": "SHORT",
                        "price": round(sell_open_px, 2), "quantity": round(qty, 4),
                        "value": round(proceeds, 2), "pnl": None,
                        "stop": round(sig.stop_price, 2),
                        "target": round(sig.target_price, 2) if sig.target_price else None,
                        "confidence": sig.confidence,
                        "reason": sig.reason,
                        "risk_usd": round(trade_risk, 2),
                        "regime": regime.value,
                        "atr_pct": atr_pct,
                        "volatility_bucket": volatility_bucket,
                        "strategy_name": strategy.name,
                        "symbol": symbol,
                        "commission": round(commission, 4),
                    })

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
                # cost-adjusted buy price: cost_model=None → buy_px==fill_price (identity)
                buy_px = fill_price if cost_model is None else cost_model.apply_buy(fill_price)
                if qty >= 0.001 and buy_px * qty <= capital:
                    cost = buy_px * qty
                    commission = 0.0 if cost_model is None else cost_model.entry_commission(qty, cost)
                    capital -= cost + commission
                    position = qty
                    position_cost = cost + commission
                    open_risk_usd += trade_risk
                    entry_stop      = sig.stop_price
                    entry_target    = sig.target_price
                    entry_price_rec = buy_px
                    initial_risk    = (buy_px - sig.stop_price) if sig.stop_price else None
                    bars_held = 0
                    atr_pct = round(_current_atr(df_slice) / fill_price * 100, 3) if fill_price else 0.0
                    trades.append({
                        "date": today, "side": "BUY",
                        "price": round(buy_px, 2), "quantity": round(qty, 4),
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
                        "commission": round(commission, 4),
                    })

        elif position > 0:
            try:
                sig = strategy.run(
                    symbol,
                    df_slice,
                    regime=regime,
                    volatility_bucket=volatility_bucket,
                    suitability_config=suitability_config,
                    momentum_snapshot=_mom_snapshot_today,
                )
            except Exception:
                sig = None

            if sig and sig.direction == "SELL":
                sell_px = fill_price if cost_model is None else cost_model.apply_sell(fill_price)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                pnl = proceeds - commission - position_cost
                capital += proceeds - commission
                trades.append({
                    "date": today, "side": "SELL",
                    "price": round(sell_px, 2), "quantity": position,
                    "value": round(proceeds, 2), "pnl": round(pnl, 2),
                    "stop": None, "target": None,
                    "confidence": sig.confidence if sig else None,
                    "reason": sig.reason if sig else "",
                    "risk_usd": None,
                    "regime": regime.value,
                    "strategy_name": strategy.name,
                    "symbol": symbol,
                    "commission": round(commission, 4),
                })
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0

        elif position < 0:
            # Strategy-initiated cover: a BUY signal while we're short means
            # the bearish thesis is broken — cover the position.
            try:
                sig = strategy.run(
                    symbol,
                    df_slice,
                    regime=regime,
                    volatility_bucket=volatility_bucket,
                    suitability_config=suitability_config,
                    momentum_snapshot=_mom_snapshot_today,
                )
            except Exception:
                sig = None

            if sig and sig.direction == "BUY":
                shares = abs(position)
                cover_px = fill_price if cost_model is None else cost_model.apply_buy(fill_price)
                cover_value = cover_px * shares
                commission = 0.0 if cost_model is None else cost_model.entry_commission(shares, cover_value)
                pnl = position_cost - cover_value - commission
                capital -= cover_value + commission
                trades.append({
                    "date": today, "side": "COVER",
                    "price": round(cover_px, 2), "quantity": shares,
                    "value": round(cover_value, 2), "pnl": round(pnl, 2),
                    "stop": None, "target": None,
                    "confidence": sig.confidence if sig else None,
                    "reason": sig.reason if sig else "",
                    "risk_usd": None,
                    "regime": regime.value,
                    "strategy_name": strategy.name,
                    "symbol": symbol,
                    "commission": round(commission, 4),
                })
                position = 0.0; position_cost = 0.0; entry_stop = None; entry_target = None; open_risk_usd = 0.0; entry_price_rec = None; initial_risk = None; bars_held = 0

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
        sell_px = last_price if cost_model is None else cost_model.apply_sell(last_price)
        proceeds = sell_px * position
        commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
        pnl = proceeds - commission - position_cost
        capital += proceeds - commission
        trades.append(_trade("SELL (close)", dates[-1], sell_px, position, proceeds, pnl))
    elif position < 0:
        last_price = float(df_full["Close"].iloc[-1])
        shares = abs(position)
        cover_px = last_price if cost_model is None else cost_model.apply_buy(last_price)
        cover_value = cover_px * shares
        commission = 0.0 if cost_model is None else cost_model.entry_commission(shares, cover_value)
        pnl = position_cost - cover_value - commission
        capital -= cover_value + commission
        trades.append(_trade("COVER (close)", dates[-1], cover_px, shares, cover_value, pnl))

    final_capital = capital
    total_pnl = final_capital - initial_capital

    # Long exits: "SELL" / "SELL (...)"; short exits: "COVER" / "COVER (...)".
    # Both are round-trip closes — count and PF-weight them together.
    exit_trades = [t for t in trades if "SELL" in t["side"] or "COVER" in t["side"]]
    # Entry legs (no realized P&L on the entry record itself):
    entry_trades = [t for t in trades if t["side"] in ("BUY", "SHORT")]
    winning = [t for t in exit_trades if (t.get("pnl") or 0) > 0]
    losing  = [t for t in exit_trades if (t.get("pnl") or 0) <= 0]
    win_rate = len(winning) / len(exit_trades) * 100 if exit_trades else 0.0
    total_completed_trades = len(exit_trades)

    # Backwards-compat aliases (some callers reference these names)
    sell_trades = exit_trades
    buy_trades = entry_trades

    capital_employed = sum(t["value"] for t in entry_trades) if entry_trades else 0.0
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
