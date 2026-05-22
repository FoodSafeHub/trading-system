from __future__ import annotations

from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.services.backtest.engine import run_backtest
from app.services.backtest.consensus_engine import run_consensus_backtest
from app.services.strategy.engine import load_strategies_from_config

router = APIRouter(prefix="/backtest", tags=["backtest"])

# New strategy type names — used for consensus defaults
_NEW_STRATEGY_TYPES = {
    "rsi2_mean_reversion",
    "ema_macd_crossover",
    "bb_squeeze_breakout",
    "pullback_ema50",
    "vix_spike_reversal",
}

# Symbols that use the 5 new strategies by default in consensus mode
_NEW_STRATEGY_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "SPY"}


@router.get("/run/{strategy_name}")
def backtest_strategy(
    strategy_name: str,
    period: str = "1y",
    initial_capital: float = 100000.0,
    quantity: float = 0.0,
    symbol: Optional[str] = None,
):
    """
    Run a backtest for a named strategy from strategies.json.

    symbol  : override the symbol configured in strategies.json (optional)
    period  : 1mo, 3mo, 6mo, 1y, 2y
    """
    configs = load_strategies_from_config()
    config = next((c for c in configs if c.name == strategy_name), None)
    if not config:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")

    # Allow caller to override the symbol
    effective_symbol = symbol.upper() if symbol else config.symbol

    try:
        result = run_backtest(
            strategy_name=config.name,
            symbol=effective_symbol,
            strategy_type=config.type,
            params=config.params,
            period=period,
            initial_capital=initial_capital,
            quantity=quantity,
        )
        return {
            "strategy_name": result.strategy_name,
            "symbol": result.symbol,
            "period": result.period,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_pnl": result.total_pnl,
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate_pct": result.win_rate_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "equity_curve": result.equity_curve,
            "trades": [
                {"date": t.date, "side": t.side, "price": t.price,
                 "quantity": t.quantity, "value": round(t.value, 2)}
                for t in result.trades
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/consensus/{symbol}")
def backtest_consensus(
    symbol: str,
    min_agreement: int = 2,
    period: str = "1y",
    initial_capital: float = 100000.0,
    new_only: bool = True,
):
    """
    Run a consensus backtest for a symbol.

    new_only=True (default): uses only the 5 new strategies for AAPL/MSFT/GOOGL/SPY.
    new_only=False          : uses all configured strategies for the symbol.
    """
    sym = symbol.upper()
    configs = load_strategies_from_config()
    symbol_configs = [c for c in configs if c.symbol == sym]
    if not symbol_configs:
        raise HTTPException(status_code=404, detail=f"No strategies configured for symbol '{symbol}'")

    # For canonical symbols, filter to new strategies only by default
    if new_only and sym in _NEW_STRATEGY_SYMBOLS:
        new_configs = [c for c in symbol_configs if c.type in _NEW_STRATEGY_TYPES]
        if new_configs:
            symbol_configs = new_configs

    try:
        result = run_consensus_backtest(sym, symbol_configs, min_agreement, period, initial_capital)
        return {
            "symbol": result.symbol,
            "period": result.period,
            "min_agreement": result.min_agreement,
            "strategies_used": result.strategies_used,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_pnl": result.total_pnl,
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate_pct": result.win_rate_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "equity_curve": result.equity_curve,
            "trades": result.trades,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/consensus-symbols")
def list_consensus_symbols(new_only: bool = False):
    """
    List symbols that have 2+ strategies configured.

    new_only=True : only counts the 5 new strategy types.
    """
    configs = load_strategies_from_config()
    by_symbol: dict = defaultdict(list)
    for c in configs:
        if new_only and c.type not in _NEW_STRATEGY_TYPES:
            continue
        by_symbol[c.symbol].append(c.name)
    return [
        {"symbol": sym, "strategy_count": len(names), "strategies": names}
        for sym, names in sorted(by_symbol.items())
        if len(names) >= 2
    ]


@router.get("/live-signals/{symbol}")
def backtest_live_signals(symbol: str, period: str = "1y"):
    """Run all 7 strategies on the latest bar and return per-strategy signals.

    Mirrors the Perplexity 'Current Signals' UX but uses the 7 strategies
    available in the Backtest page (5 regime-aware + Bollinger + Fib).
    No DB writes — this is a read-only what-would-fire-right-now snapshot.
    """
    from app.services.scanner.scanner_service import _make_generic_configs_full
    from app.services.strategy.rules import evaluate_strategy
    from app.services.market_data.provider import get_ohlcv

    sym = symbol.upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")

    try:
        df = get_ohlcv(sym, period=period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"market data load failed: {exc}")
    if df is None or df.empty or len(df) < 30:
        raise HTTPException(status_code=400,
                            detail=f"not enough history for {sym} ({period})")

    prices = df["Close"].dropna()
    last_close = float(prices.iloc[-1]) if len(prices) else None

    out: list[dict] = []
    for cfg in _make_generic_configs_full(sym):
        try:
            sig = evaluate_strategy(cfg.type, sym, prices, cfg.params, ohlcv=df)
            indicators = sig.indicators or {}
            entry = sig.price_at_signal if sig.direction in ("BUY", "SELL") else None
            out.append({
                "strategy_name": cfg.name,
                "strategy_type": cfg.type,
                "direction": sig.direction,
                "strength": sig.strength,
                "price_at_signal": sig.price_at_signal,
                "last_close": last_close,
                "entry_price": entry,
                # Best-effort: many rules stash stop/target in indicators
                "stop_price": indicators.get("stop_price") or indicators.get("stop"),
                "target_price": indicators.get("target_price") or indicators.get("target"),
                "indicators": indicators,
                "reason": indicators.get("reason") or indicators.get("note") or "",
            })
        except Exception as exc:
            out.append({
                "strategy_name": cfg.name,
                "strategy_type": cfg.type,
                "direction": "HOLD",
                "error": str(exc),
                "indicators": {},
            })
    return {"symbol": sym, "last_close": last_close,
            "as_of": str(df.index[-1])[:19] if len(df.index) else None,
            "signals": out}


@router.get("/run-generic/{symbol}/{strategy_type}")
def backtest_run_generic(
    symbol: str,
    strategy_type: str,
    period: str = "1y",
    initial_capital: float = 100000.0,
):
    """Run a single strategy on an arbitrary symbol using factory defaults.

    Same response shape as /backtest/run/{strategy_name} so the dashboard's
    Single Strategy view can render it without a code path split. Resolves
    the strategy via _make_generic_configs_full(), which covers all 7 types
    (5 regime-aware + Bollinger + Fibonacci).
    """
    from app.services.scanner.scanner_service import _make_generic_configs_full
    sym = symbol.upper().strip()
    stype = (strategy_type or "").strip().lower()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")
    configs = _make_generic_configs_full(sym)
    cfg = next((c for c in configs if c.type == stype), None)
    if cfg is None:
        valid = sorted({c.type for c in configs})
        raise HTTPException(
            status_code=404,
            detail=f"Unknown strategy_type '{strategy_type}'. Valid: {valid}",
        )
    try:
        result = run_backtest(
            strategy_name=cfg.name,
            symbol=sym,
            strategy_type=cfg.type,
            params=cfg.params,
            period=period,
            initial_capital=initial_capital,
            quantity=0,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "period": result.period,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_pnl": result.total_pnl,
        "total_return_pct": result.total_return_pct,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate_pct": result.win_rate_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "equity_curve": result.equity_curve,
        "trades": [
            {"date": t.date, "side": t.side, "price": t.price,
             "quantity": t.quantity, "value": round(t.value, 2)}
            for t in result.trades
        ],
    }


@router.get("/custom-consensus/{symbol}")
def backtest_custom_consensus(
    symbol: str,
    min_agreement: int = 2,
    period: str = "1y",
    initial_capital: float = 100000.0,
):
    """Consensus backtest for an arbitrary symbol using all 5 new strategy types.

    This is what the scanner runs under the hood, so a candidate flagged on the
    scanner page (e.g. BRK-B_Pullback_EMA50) can be re-tested here with the same
    factory defaults. Falls back to the regular /consensus endpoint when the
    symbol has explicit configs in strategies.json.
    """
    from app.services.scanner.scanner_service import _make_generic_configs
    sym = symbol.upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")
    configs = _make_generic_configs(sym)
    try:
        result = run_consensus_backtest(sym, configs, min_agreement, period, initial_capital)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "symbol": result.symbol,
        "period": result.period,
        "min_agreement": result.min_agreement,
        "strategies_used": result.strategies_used,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_pnl": result.total_pnl,
        "total_return_pct": result.total_return_pct,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate_pct": result.win_rate_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "equity_curve": result.equity_curve,
        "trades": result.trades,
    }


@router.get("/custom-compare-all/{symbol}")
def backtest_custom_compare_all(
    symbol: str,
    period: str = "1y",
    initial_capital: float = 100000.0,
):
    """Run each of the 5 scanner strategies INDEPENDENTLY on an arbitrary symbol.

    Mirrors the Perplexity Compare All UX: one row per strategy with win-rate,
    profit factor, expectancy, return, drawdown, sharpe — so the user can see
    which strategy works best for the ticker before promoting it to auto-trade.

    Uses the FULL 7-strategy set (5 regime-aware + 2 legacy Bollinger/Fib) so
    the user gets the same coverage they see for predefined symbols in
    Single Strategy mode.
    """
    from app.services.scanner.scanner_service import _make_generic_configs_full

    sym = symbol.upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")

    configs = _make_generic_configs_full(sym)
    results: list[dict] = []
    for cfg in configs:
        try:
            r = run_backtest(
                strategy_name=cfg.name,
                symbol=sym,
                strategy_type=cfg.type,
                params=cfg.params,
                period=period,
                initial_capital=initial_capital,
                quantity=0,
            )
        except Exception as exc:
            results.append({"strategy_name": cfg.name, "error": str(exc)})
            continue

        # Compute per-roundtrip P&L so we can derive profit_factor / avg_win / expectancy.
        buys = [t for t in r.trades if t.side == "BUY"]
        sells = [t for t in r.trades if "SELL" in t.side]
        n_rt = min(len(buys), len(sells))
        wins_pnl: list[float] = []
        losses_pnl: list[float] = []
        win_pct_list: list[float] = []
        loss_pct_list: list[float] = []
        for b, s in zip(buys[:n_rt], sells[:n_rt]):
            pnl = s.value - b.value
            pct = (pnl / b.value * 100) if b.value > 0 else 0.0
            if pnl > 0:
                wins_pnl.append(pnl)
                win_pct_list.append(pct)
            else:
                losses_pnl.append(pnl)
                loss_pct_list.append(pct)

        gross_win = sum(wins_pnl)
        gross_loss = abs(sum(losses_pnl))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (None if not wins_pnl else float("inf"))
        avg_win_pct = (sum(win_pct_list) / len(win_pct_list)) if win_pct_list else 0.0
        avg_loss_pct = (sum(loss_pct_list) / len(loss_pct_list)) if loss_pct_list else 0.0
        wr_frac = (r.win_rate_pct / 100.0) if r.win_rate_pct else 0.0
        expectancy_pct = wr_frac * avg_win_pct + (1 - wr_frac) * avg_loss_pct

        # Years between start and end → CAGR
        cagr = 0.0
        try:
            from datetime import date as _date
            if r.start_date and r.end_date:
                d0 = _date.fromisoformat(r.start_date)
                d1 = _date.fromisoformat(r.end_date)
                years = max((d1 - d0).days / 365.25, 1e-6)
                if r.initial_capital > 0 and r.final_capital > 0:
                    cagr = ((r.final_capital / r.initial_capital) ** (1 / years) - 1) * 100
        except Exception:
            cagr = 0.0

        # Profit factor JSON-safe (inf → None so Streamlit shows "—")
        pf_out = profit_factor
        if pf_out is not None and pf_out == float("inf"):
            pf_out = None

        results.append({
            "strategy_name": r.strategy_name,
            "symbol": r.symbol,
            "period": r.period,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "initial_capital": r.initial_capital,
            "final_capital": r.final_capital,
            "total_pnl": r.total_pnl,
            "total_return_pct": r.total_return_pct,
            "total_trades": r.total_trades,
            "winning_trades": r.winning_trades,
            "losing_trades": r.losing_trades,
            "win_rate_pct": r.win_rate_pct,
            "profit_factor": round(pf_out, 2) if pf_out is not None else None,
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "expectancy_pct": round(expectancy_pct, 2),
            "cagr": round(cagr, 2),
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            # Detail payload for the inline "Show details" expander on the UI.
            # Avoids a second round-trip per row.
            "equity_curve": r.equity_curve,
            "trades": [
                {"date": t.date, "side": t.side, "price": t.price,
                 "quantity": t.quantity, "value": round(t.value, 2)}
                for t in r.trades
            ],
        })
    return results


@router.get("/strategies")
def list_backtest_strategies(new_only: bool = False):
    """
    List all strategies available for backtesting.

    new_only=True : only the 5 new strategies.
    """
    configs = load_strategies_from_config()
    if new_only:
        configs = [c for c in configs if c.type in _NEW_STRATEGY_TYPES]
    return [{"name": c.name, "symbol": c.symbol, "type": c.type} for c in configs]
