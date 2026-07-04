from __future__ import annotations

"""
Backtesting engine — simulates strategy execution against historical OHLCV data.

For each bar, runs the strategy on all data up to that point (no lookahead),
simulates fills at next-bar open price, and tracks P&L.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import pandas as pd

from app.services.backtest.costs import CostModel
from app.services.market_data.provider import get_ohlcv
from app.services.strategy.rules import evaluate_strategy, PositionState


@dataclass
class BacktestTrade:
    date: str
    symbol: str
    side: str           # BUY or SELL
    price: float
    quantity: float
    value: float        # price * quantity
    signal_from: str    # strategy name


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    period: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    trades: List[BacktestTrade]
    equity_curve: List[dict]   # [{date, equity}]
    total_pnl: float


def run_backtest(
    strategy_name: str,
    symbol: str,
    strategy_type: str,
    params: dict,
    period: str = "1y",
    initial_capital: float = 100_000.0,
    quantity: float = 1.0,
    df: pd.DataFrame | None = None,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """
    Simulate a strategy over historical data.
    Uses next-bar open as the fill price to avoid lookahead bias.

    df: optional pre-fetched OHLCV. When provided, skips the get_ohlcv call —
    used by grid-search calibration to reuse one fetch across many param combos.

    cost_model: optional slippage/commission model. DEFAULT None == zero cost,
    in which case every fill price, quantity, and equity point is identical to
    the pre-cost engine (Phase 0 behaviour-neutral contract). When supplied,
    buys fill worse, sells fill worse, and commissions/taxes are deducted from
    cash — so reported returns become net of trading frictions.
    """
    if df is None:
        df = get_ohlcv(symbol, period=period)
    if df.empty or len(df) < 30:
        # Distinguish "no data anywhere" from "stock too newly listed". A short
        # series means the fetch worked but the stock simply hasn't traded long
        # enough to backtest — common for recent IPOs/relistings on NSE.
        if df.empty:
            raise ValueError(f"No data returned for {symbol} over period {period}")
        listed = str(df.index[0])[:10]
        raise ValueError(
            f"{symbol} has only {len(df)} trading days of history (listed ~{listed}); "
            f"backtests need at least 30 bars. Pick a longer-established stock or "
            f"a shorter period."
        )

    closes = df["Close"].dropna()
    opens  = df["Open"].dropna()
    dates  = [str(d)[:10] for d in df.index]

    capital = initial_capital
    position = 0.0          # shares held
    position_cost = 0.0     # total cost basis
    entry_price = 0.0       # fill price of the open position (for trailing-stop overlay)
    highest_close = 0.0     # highest close seen since entry (Chandelier trail)
    bars_held = 0           # bars elapsed since entry (for time-stop exit policies)
    trades: List[BacktestTrade] = []
    equity_curve: List[dict] = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    daily_returns: List[float] = []
    prev_equity = initial_capital

    # Approach C: tight trailing stop state.
    # When the assigned strategy fires SELL, instead of exiting immediately we
    # switch into "tight trail" mode — the position stays open and we track a
    # 2% trailing stop from the signal price. The position only closes when the
    # high since the signal drops by trail_pct%.
    # approach_c=True is set via params["approach_c"] = True.
    _approach_c = bool(params.get("approach_c", False))
    _tight_trail_pct = float(params.get("tight_trail_pct", 2.0))
    _in_tight_trail = False    # True when SELL signal fired, trail is active
    _trail_high = 0.0          # highest high seen since trail activated
    _trail_stop = 0.0          # current stop = _trail_high * (1 - trail_pct/100)
    _signal_price = 0.0        # close price when SELL signal fired (for audit)

    # Need at least 30 bars of history before we start signalling
    lookback = 35

    for i in range(lookback, len(df)):
        price_series = closes.iloc[:i]
        current_close = float(closes.iloc[i])
        # Fill at next bar open if available, else current close
        fill_price = float(opens.iloc[i]) if i < len(opens) else current_close
        today = dates[i]

        df_slice = df.iloc[:i]
        # The rule decides on data through bar i-1 (price_series/df_slice end at
        # i-1) and we fill at bar i's open — no lookahead. The trailing overlay
        # must see the SAME frame, so trail against the peak close through i-1,
        # not current_close (= close[i], which the rule cannot see yet).
        if position > 0:
            highest_close = max(highest_close, float(closes.iloc[i - 1]))
            bars_held += 1
        pos_state = (
            PositionState(entry_price=entry_price, highest_close=highest_close,
                          bars_held=bars_held)
            if position > 0 else None
        )
        # ── Hard stop-loss: checked BEFORE strategy signal ───────────────
        # Fires at close price; sells at next-bar open (same fill logic as
        # strategy exits). Default 8% loss from entry, configurable via
        # params["stop_loss_pct"]. Set to 0 to disable.
        _stop_loss_pct = float(params.get("stop_loss_pct", 8.0))
        if position > 0 and _stop_loss_pct > 0 and entry_price > 0:
            _loss_pct = (current_close - entry_price) / entry_price * 100.0
            if _loss_pct <= -_stop_loss_pct:
                sell_px = fill_price if cost_model is None else cost_model.apply_sell(fill_price)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                capital += proceeds - commission
                trades.append(BacktestTrade(
                    date=today, symbol=symbol, side="SELL",
                    price=sell_px, quantity=position, value=proceeds,
                    signal_from=f"{strategy_name}:stop_loss_{_stop_loss_pct:.0f}pct",
                ))
                position = 0.0
                position_cost = 0.0
                entry_price = 0.0
                highest_close = 0.0
                bars_held = 0
                # Skip strategy evaluation this bar — position already closed
                equity = capital
                equity_curve.append({"date": today, "equity": round(equity, 2)})
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity * 100
                if dd > max_drawdown:
                    max_drawdown = dd
                ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                daily_returns.append(ret)
                prev_equity = equity
                continue

        # ── Approach C: tight trail active — check stop before strategy eval ──
        if _approach_c and _in_tight_trail and position > 0:
            bar_high = float(df["High"].iloc[i]) if "High" in df.columns else current_close
            bar_low  = float(df["Low"].iloc[i])  if "Low"  in df.columns else current_close
            # Ratchet the trail high upward with this bar's high
            if bar_high > _trail_high:
                _trail_high = bar_high
                _trail_stop = round(_trail_high * (1 - _tight_trail_pct / 100), 2)
            # Exit when low touches or breaches the stop
            if bar_low <= _trail_stop:
                sell_px = max(_trail_stop, float(opens.iloc[i])) if i < len(opens) else _trail_stop
                if cost_model is not None:
                    sell_px = cost_model.apply_sell(sell_px)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                pnl = proceeds - position_cost
                capital += proceeds - commission
                trades.append(BacktestTrade(
                    date=today, symbol=symbol, side="SELL",
                    price=sell_px, quantity=position, value=proceeds,
                    signal_from=f"{strategy_name}:tight_trail_{_tight_trail_pct:.0f}pct"
                                f"_signal@{_signal_price:.2f}",
                ))
                position = 0.0; position_cost = 0.0; entry_price = 0.0
                highest_close = 0.0; bars_held = 0
                _in_tight_trail = False; _trail_high = 0.0; _trail_stop = 0.0; _signal_price = 0.0
                equity = capital
                equity_curve.append({"date": today, "equity": round(equity, 2)})
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, dd)
                ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                daily_returns.append(ret)
                prev_equity = equity
                continue
            # Trail active but not hit — skip strategy re-evaluation this bar
            equity = capital + position * current_close
            equity_curve.append({"date": today, "equity": round(equity, 2)})
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, dd)
            ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            daily_returns.append(ret)
            prev_equity = equity
            continue

        # _backtest_mode lets rules with a live-safety gate (e.g. ceei paper_only)
        # know they are being simulated, so research is never blocked by the gate.
        signal = evaluate_strategy(
            strategy_type, symbol, price_series, {**params, "_backtest_mode": True},
            ohlcv=df_slice, position=pos_state
        )

        direction = signal.direction

        # Execute simulated trade
        if direction == "BUY" and position == 0:
            # Cost-adjusted fill: buys fill WORSE (higher). cost_model None => buy_px == fill_price.
            buy_px = fill_price if cost_model is None else cost_model.apply_buy(fill_price)
            # Use up to 95% of available capital
            affordable_qty = (capital * 0.95) / buy_px if buy_px > 0 else 0
            # quantity<=0 means "use all capital"; quantity>0 is a fixed share count cap
            actual_qty = affordable_qty if quantity <= 0 else min(quantity, affordable_qty)
            actual_qty = actual_qty if affordable_qty >= 0.01 else 0
            if actual_qty > 0:
                cost = buy_px * actual_qty
                commission = 0.0 if cost_model is None else cost_model.entry_commission(actual_qty, cost)
                capital -= cost + commission
                position = actual_qty
                position_cost = cost
                entry_price = buy_px
                highest_close = current_close
                bars_held = 0
                trades.append(BacktestTrade(
                    date=today, symbol=symbol, side="BUY",
                    price=buy_px, quantity=actual_qty, value=cost,
                    signal_from=strategy_name,
                ))

        elif direction == "SELL" and position > 0:
            if _approach_c and not _in_tight_trail:
                # Approach C: SELL signal fires → activate 2% tight trail.
                # Do NOT exit yet. Record signal price and start trailing
                # from the signal bar's close. Trail checks happen at the
                # top of the next bar's iteration.
                _signal_price = current_close
                _trail_high   = current_close
                _trail_stop   = round(current_close * (1 - _tight_trail_pct / 100), 2)
                _in_tight_trail = True
                # Record signal in trades list as a marker (no cash change)
                trades.append(BacktestTrade(
                    date=today, symbol=symbol, side="SELL_SIGNAL",
                    price=current_close, quantity=0.0, value=0.0,
                    signal_from=f"{strategy_name}:approach_c_signal",
                ))
            else:
                # Normal exit (approach_c=False, or already in trail which
                # shouldn't reach here but guard anyway)
                sell_px = fill_price if cost_model is None else cost_model.apply_sell(fill_price)
                proceeds = sell_px * position
                commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
                pnl = proceeds - position_cost
                capital += proceeds - commission
                trades.append(BacktestTrade(
                    date=today, symbol=symbol, side="SELL",
                    price=sell_px, quantity=position, value=proceeds,
                    signal_from=strategy_name,
                ))
                position = 0.0
                position_cost = 0.0
                entry_price = 0.0
                highest_close = 0.0
                bars_held = 0

        # Mark-to-market equity
        equity = capital + position * current_close
        equity_curve.append({"date": today, "equity": round(equity, 2)})

        # Drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

        # Daily return
        ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(ret)
        prev_equity = equity

    # Close any open position at last price
    final_close = float(closes.iloc[-1])
    if position > 0:
        # cost_model None => sell_px == final_close, commission 0 (identical close).
        sell_px = final_close if cost_model is None else cost_model.apply_sell(final_close)
        proceeds = sell_px * position
        commission = 0.0 if cost_model is None else cost_model.exit_commission(position, proceeds)
        trades.append(BacktestTrade(
            date=dates[-1], symbol=symbol, side="SELL (close)",
            price=sell_px, quantity=position, value=proceeds,
            signal_from=strategy_name,
        ))
        capital += proceeds - commission
        position = 0.0

    final_capital = capital

    # P&L per round trip
    # Exclude SELL_SIGNAL markers (approach_c signal events, qty=0) from round-trip counting
    buy_trades  = [t for t in trades if t.side == "BUY"]
    sell_trades = [t for t in trades if t.side in ("SELL", "SELL (close)") or
                   (t.side.startswith("SELL") and t.quantity > 0)]
    round_trips = min(len(buy_trades), len(sell_trades))

    winning = losing = 0
    for b, s in zip(buy_trades[:round_trips], sell_trades[:round_trips]):
        if s.value > b.value:
            winning += 1
        else:
            losing += 1

    win_rate = (winning / round_trips * 100) if round_trips > 0 else 0.0
    total_return = (final_capital - initial_capital) / initial_capital * 100
    total_pnl = final_capital - initial_capital

    # Sharpe ratio (annualised, risk-free = 0)
    sharpe = None
    if len(daily_returns) > 1:
        import statistics
        avg = statistics.mean(daily_returns)
        std = statistics.stdev(daily_returns)
        if std > 0:
            sharpe = round((avg / std) * (252 ** 0.5), 2)

    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        period=period,
        start_date=dates[lookback] if dates else "",
        end_date=dates[-1] if dates else "",
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2),
        total_return_pct=round(total_return, 2),
        total_pnl=round(total_pnl, 2),
        total_trades=len([t for t in trades if t.side != "SELL_SIGNAL"]),
        winning_trades=winning,
        losing_trades=losing,
        win_rate_pct=round(win_rate, 2),
        max_drawdown_pct=round(max_drawdown, 2),
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity_curve,
    )
