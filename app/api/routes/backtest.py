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


_TRAIL_KEYS = ("trail_enabled", "trail_trigger_pct", "atr_trail_mult", "atr_trail_period")


def _strip_exit_layers(params: dict, *, disable_trail: bool, disable_exit_policy: bool) -> dict:
    """Return a copy of params with Layer-2 (Chandelier trail) and/or Layer-3
    (Phase-1 exit_policy) keys removed. Used by the dashboard's exit-layer
    compare toggles so the same backtest can be re-run with layers off."""
    out = dict(params)
    if disable_trail:
        for k in _TRAIL_KEYS:
            out.pop(k, None)
    if disable_exit_policy:
        out.pop("exit_policy", None)
    return out


@router.get("/run-generic/{symbol}/{strategy_type}")
def backtest_run_generic(
    symbol: str,
    strategy_type: str,
    period: str = "1y",
    initial_capital: float = 100000.0,
    disable_trail: bool = False,
    disable_exit_policy: bool = False,
    stop_loss_pct: float = 8.0,
    exit_rsi: float = 0.0,   # 0 = use strategy default; >0 overrides the SELL RSI gate
):
    """Run a single strategy on an arbitrary symbol using factory defaults.

    Same response shape as /backtest/run/{strategy_name} so the dashboard's
    Single Strategy view can render it without a code path split. Resolves
    the strategy via _make_generic_configs_full(), which covers all 7 types
    (5 regime-aware + Bollinger + Fibonacci).

    Optional `disable_trail` / `disable_exit_policy` query params strip the
    corresponding exit layers from the config before running. The response
    echoes the effective params + overrides so the dashboard can render the
    "Effective exit policy" panel without re-deriving anything client-side.
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
    effective_params = _strip_exit_layers(
        cfg.params,
        disable_trail=disable_trail,
        disable_exit_policy=disable_exit_policy,
    )
    # Hard stop-loss injected into params — engine reads it before strategy signal
    if stop_loss_pct > 0:
        effective_params = {**effective_params, "stop_loss_pct": stop_loss_pct}
    elif "stop_loss_pct" in effective_params:
        effective_params = {k: v for k, v in effective_params.items() if k != "stop_loss_pct"}
    # RSI exit override — applies to all RSI-gated sell rules
    if exit_rsi > 0:
        for key in ("rsi_exit_threshold", "exit_rsi", "rsi_overbought"):
            effective_params = {**effective_params, key: exit_rsi}
    try:
        result = run_backtest(
            strategy_name=cfg.name,
            symbol=sym,
            strategy_type=cfg.type,
            params=effective_params,
            period=period,
            initial_capital=initial_capital,
            quantity=0,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "strategy_type": cfg.type,
        "params": effective_params,
        "overrides": {
            "disable_trail": disable_trail,
            "disable_exit_policy": disable_exit_policy,
            "stop_loss_pct": stop_loss_pct,
            "exit_rsi": exit_rsi if exit_rsi > 0 else None,
        },
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


@router.get("/walkforward/{symbol}/{strategy_type}")
def backtest_walkforward(
    symbol: str,
    strategy_type: str,
    mode: str = "simple",
    period: str = "5y",
    train_pct: float = 0.70,
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    initial_capital: float = 100000.0,
):
    """Walk-forward out-of-sample validation for a single v2 strategy.

    Drives the canonical run_backtest engine (same no-lookahead path as the
    Backtest page and scheduler, including the Chandelier overlay) on time
    slices, so the OOS numbers are directly comparable to a normal backtest.

    mode="simple"  : one IS (train_pct) / OOS (rest) split.
    mode="rolling" : sliding IS/OOS windows, OOS curves stitched into a
                     composite; reports a global Walk-Forward Efficiency.

    Resolves params via _make_generic_configs_full (factory defaults + any
    saved calibration profile), so what's validated matches what trades.
    """
    from app.services.scanner.scanner_service import _make_generic_configs_full
    from app.services.backtest.walkforward_v2 import (
        run_simple_split_v2, run_rolling_v2, to_dict,
    )

    sym = symbol.upper().strip()
    stype = (strategy_type or "").strip().lower()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")

    cfg = next((c for c in _make_generic_configs_full(sym) if c.type == stype), None)
    if cfg is None:
        valid = sorted({c.type for c in _make_generic_configs_full(sym)})
        raise HTTPException(
            status_code=404,
            detail=f"Unknown strategy_type '{strategy_type}'. Valid: {valid}",
        )

    try:
        if mode == "rolling":
            result = run_rolling_v2(
                strategy_type=stype, symbol=sym, params=cfg.params,
                period=period, train_years=train_years, test_years=test_years,
                step_years=step_years, initial_capital=initial_capital,
            )
        else:
            result = run_simple_split_v2(
                strategy_type=stype, symbol=sym, params=cfg.params,
                period=period, train_pct=train_pct, initial_capital=initial_capital,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    out = to_dict(result)
    out["strategy_name"] = cfg.name
    return out


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
    strategy_set: str = "legacy",
):
    """Run each scanner strategy INDEPENDENTLY on an arbitrary symbol.

    Mirrors the Perplexity Compare All UX: one row per strategy with win-rate,
    profit factor, expectancy, return, drawdown, sharpe — so the user can see
    which strategy works best for the ticker before promoting it to auto-trade.

    strategy_set:
      * "legacy"  (DEFAULT) — the FULL 7-strategy set (5 regime-aware + 2 legacy
        Bollinger/Fib). Output is byte-unchanged from before Phase 1.
      * "unified" — OPT-IN: the 6 consolidated Phase 1 strategies, each with its
        default exit_policy. Lets you preview the new suite without changing the
        default page behaviour.
    """
    from app.services.scanner.scanner_service import (
        _make_generic_configs_full, _make_unified_configs,
    )

    sym = symbol.upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")

    configs = (_make_unified_configs(sym) if strategy_set == "unified"
               else _make_generic_configs_full(sym))
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
            "strategy_type": cfg.type,
            "params": cfg.params,
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


# ── Scanner calibration (Optimize Filters for the Backtest page) ───────────────

_SCANNER_CALIBRATABLE = {
    # Legacy types (unchanged — calibrate exactly as before).
    "rsi2_mean_reversion", "ema_macd_crossover", "bb_squeeze_breakout",
    "pullback_ema50", "vix_spike_reversal",
    # Unified Phase 1/2 types (parallel; saved under their own "<new>:SYMBOL" key,
    # and they inherit predecessor calibration via the alias map until re-tuned).
    "rsi2_reversion", "trend_pullback", "squeeze_breakout",
    "momentum_breakout", "panic_reversal", "trend_follow",
}


def _trade_metrics(r) -> dict:
    """Derive win-rate/expectancy/profit-factor/avg-win-loss from a BacktestResult."""
    buys = [t for t in r.trades if t.side == "BUY"]
    sells = [t for t in r.trades if "SELL" in t.side]
    n_rt = min(len(buys), len(sells))
    win_pct, loss_pct, gross_win, gross_loss = [], [], 0.0, 0.0
    for b, s in zip(buys[:n_rt], sells[:n_rt]):
        pnl = s.value - b.value
        pct = (pnl / b.value * 100) if b.value > 0 else 0.0
        if pnl > 0:
            win_pct.append(pct); gross_win += pnl
        else:
            loss_pct.append(pct); gross_loss += abs(pnl)
    avg_win = sum(win_pct) / len(win_pct) if win_pct else 0.0
    avg_loss = sum(loss_pct) / len(loss_pct) if loss_pct else 0.0
    wr_frac = (r.win_rate_pct / 100.0) if r.win_rate_pct else 0.0
    expectancy = wr_frac * avg_win + (1 - wr_frac) * avg_loss
    pf = (gross_win / gross_loss) if gross_loss > 0 else (None if not win_pct else None)
    return {
        "trades": r.total_trades,
        "round_trips": n_rt,
        "win_rate_pct": round(r.win_rate_pct, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expectancy_pct": round(expectancy, 3),
        "total_return_pct": round(r.total_return_pct, 2),
        "profit_factor": round(pf, 2) if pf is not None else None,
        "max_drawdown_pct": round(r.max_drawdown_pct, 2),
    }


@router.get("/scanner-calibrate/{symbol}/{strategy_type}")
def scanner_calibrate(
    symbol: str,
    strategy_type: str,
    period: str = "5y",
    initial_capital: float = 10_000.0,
):
    """
    Optimize-Filters-and-save for a scanner strategy. Grid-searches (coordinate
    descent) the entry PARAMS each rule actually gates on — pullback proximity +
    RSI floor + reclaim wick for pullback_ema50, RSI(2) depth + ATR skip for rsi2,
    volume + RSI band for the crossover/squeeze strats, panic depth + wick for
    vix_spike — maximising per-trade expectancy with a >=40% trade-survival floor.
    Saves only if the winner materially beats the factory baseline on win-rate OR
    expectancy OR total return. The rule logic is untouched — only params change.

    This replaces the earlier indicator-separation heuristic, which analysed
    generic indicators the scanner rules don't use and failed on high-win-rate
    strategies that have too few losers to learn from.
    """
    from app.services.scanner.scanner_service import _make_generic_configs, _make_unified_configs
    from app.services.backtest.scanner_profiles import (
        ScannerParamProfile, coordinate_descent_search, grid_for, save_profile,
    )
    from app.services.market_data.provider import get_ohlcv
    from datetime import datetime

    sym = symbol.upper().strip()
    stype = (strategy_type or "").strip().lower()
    if stype not in _SCANNER_CALIBRATABLE:
        raise HTTPException(400, f"strategy_type '{strategy_type}' is not calibratable. "
                                 f"Valid: {sorted(_SCANNER_CALIBRATABLE)}")

    # cfg.params already reflects current LIVE behaviour (factory defaults plus any
    # previously-saved override, incl. predecessor calibration inherited via the
    # alias map for new types) — the right baseline for "can we do better?".
    # Legacy types come from the generic factory; the 6 unified types from the
    # parallel unified factory. Searching both keeps old behaviour intact while
    # making the new names calibratable.
    all_cfgs = list(_make_generic_configs(sym)) + list(_make_unified_configs(sym))
    cfg = next((c for c in all_cfgs if c.type == stype), None)
    if cfg is None:
        raise HTTPException(404, f"strategy_type '{stype}' not in generic or unified set")
    if not grid_for(stype):
        raise HTTPException(400, f"No tunable param grid defined for '{stype}'")

    try:
        df = get_ohlcv(sym, period=period)
        if df is None or df.empty or len(df) < 120:
            raise HTTPException(400, f"Not enough data for {sym}")

        base_params = dict(cfg.params)

        # ── Baseline ──────────────────────────────────────────────────────────
        baseline = run_backtest(
            strategy_name=cfg.name, symbol=sym, strategy_type=stype,
            params=base_params, period=period, initial_capital=initial_capital,
            quantity=0, df=df,
        )
        baseline_m = _trade_metrics(baseline)
        baseline_rt = baseline_m["round_trips"]
        if baseline_rt < 6:
            raise HTTPException(400, f"Not enough trades to calibrate "
                                     f"({baseline_rt} round-trips, need >= 6)")

        # ── Coordinate-descent search over the real gating params ───────────────
        # Objective: per-trade expectancy. Survival floor: a combo must keep >=40%
        # of the baseline round-trips, else it's disqualified (score = -inf) so we
        # never "improve" returns by simply taking far fewer (lucky) trades.
        MIN_SURVIVAL = 0.40
        MIN_RT = 5
        NEG = float("-inf")

        def _evaluate(params):
            r = run_backtest(
                strategy_name=cfg.name, symbol=sym, strategy_type=stype,
                params=params, period=period, initial_capital=initial_capital,
                quantity=0, df=df,
            )
            m = _trade_metrics(r)
            rt = m["round_trips"]
            survival = (rt / baseline_rt) if baseline_rt > 0 else 0.0
            if rt < MIN_RT or survival < MIN_SURVIVAL:
                return NEG, m
            return m["expectancy_pct"], m

        overrides, best_score, best_m, n_evals = coordinate_descent_search(
            stype, base_params, _evaluate, rounds=2,
        )

        # If search found nothing distinct from baseline, report cleanly.
        if not overrides:
            return {
                "symbol": sym, "strategy_type": stype, "saved": False,
                "skip_reason": (f"Searched {n_evals} param combos; the factory "
                                f"defaults already give the best expectancy for "
                                f"{sym} over {period} — no change improves it."),
                "proposed_overrides": {}, "evals": n_evals,
                "comparison": {"baseline": baseline_m, "filtered": baseline_m,
                               "survival_rate_pct": 100.0, "improved_by": []},
            }

        filtered_m = best_m
        survival = (filtered_m["round_trips"] / baseline_rt) if baseline_rt > 0 else 0.0

        # ── Save gate (same multi-metric thresholds as Perplexity) ──────────────
        WR_GAIN_MIN, EXP_GAIN_MIN, RET_GAIN_MIN = 3.0, 0.05, 2.0
        wr_improved  = filtered_m["win_rate_pct"]     >= baseline_m["win_rate_pct"]     + WR_GAIN_MIN
        exp_improved = filtered_m["expectancy_pct"]   >= baseline_m["expectancy_pct"]   + EXP_GAIN_MIN
        ret_improved = filtered_m["total_return_pct"] >= baseline_m["total_return_pct"] + RET_GAIN_MIN
        has_survival = survival >= MIN_SURVIVAL
        improved_by = [m for m, ok in (("win_rate", wr_improved), ("expectancy", exp_improved),
                                       ("total_return", ret_improved)) if ok]
        is_improvement = has_survival and bool(improved_by)

        comparison = {
            "baseline": baseline_m, "filtered": filtered_m,
            "survival_rate_pct": round(survival * 100, 1),
            "improved_by": improved_by,
        }

        if not is_improvement:
            reason = (f"Best of {n_evals} param combos "
                      f"({', '.join(f'{k}={v}' for k, v in overrides.items())}) "
                      f"didn't materially beat the factory defaults "
                      f"(win rate {baseline_m['win_rate_pct']:.1f}%->{filtered_m['win_rate_pct']:.1f}%, "
                      f"expectancy {baseline_m['expectancy_pct']:.2f}%->{filtered_m['expectancy_pct']:.2f}%, "
                      f"return {baseline_m['total_return_pct']:.1f}%->{filtered_m['total_return_pct']:.1f}%). "
                      f"Defaults kept.")
            return {
                "symbol": sym, "strategy_type": stype, "saved": False,
                "skip_reason": reason, "proposed_overrides": overrides,
                "evals": n_evals, "comparison": comparison,
            }

        profile = ScannerParamProfile(
            symbol=sym, strategy_type=stype,
            calibrated_at=datetime.utcnow().strftime("%Y-%m-%d"),
            n_trades=filtered_m["round_trips"],
            n_wins=int(round(filtered_m["round_trips"] * filtered_m["win_rate_pct"] / 100)),
            win_rate_pct=filtered_m["win_rate_pct"],
            param_overrides=overrides,
            survival_rate_pct=round(survival * 100, 1),
            improved_by=improved_by,
            notes=f"grid-search winner from {n_evals} combos vs {baseline_rt} baseline round-trips over {period}",
        )
        save_profile(profile)
        return {
            "symbol": sym, "strategy_type": stype, "saved": True,
            "param_overrides": overrides, "comparison": comparison,
            "evals": n_evals, "calibrated_at": profile.calibrated_at,
            "n_trades": profile.n_trades, "n_wins": profile.n_wins,
            "win_rate_pct": profile.win_rate_pct,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/scanner-profiles")
def list_scanner_profiles(strategy_type: Optional[str] = None):
    """List saved scanner calibration profiles (optionally filtered by type)."""
    from app.services.backtest.scanner_profiles import list_profiles
    from dataclasses import asdict
    return [asdict(p) for p in list_profiles(strategy_type)]


@router.get("/migrate-profiles")
def migrate_profiles(dry_run: bool = True, overwrite: bool = False):
    """Opt-in, NON-DESTRUCTIVE migration: copy predecessor calibration profiles
    to the new unified strategy names. dry_run=true (DEFAULT) only reports what
    WOULD be copied — call with dry_run=false to write. Old profiles are never
    deleted (rollback = delete the new-name copy). Note: new strategies already
    inherit old calibration via the alias map, so this is optional convenience.
    """
    from app.services.backtest.scanner_profiles import migrate_aliased_profiles
    actions = migrate_aliased_profiles(dry_run=dry_run, overwrite=overwrite)
    return {"dry_run": dry_run, "overwrite": overwrite, "actions": actions,
            "would_copy": sum(1 for a in actions if a["action"] == "copy")}


@router.get("/scanner-profiles/{symbol}/{strategy_type}")
def get_scanner_profile(symbol: str, strategy_type: str):
    """Get one saved scanner profile, or {} if none exists."""
    from app.services.backtest.scanner_profiles import load_profile
    from dataclasses import asdict
    p = load_profile(strategy_type.strip().lower(), symbol.upper())
    return asdict(p) if p else {}


@router.delete("/scanner-profiles/{symbol}/{strategy_type}")
def delete_scanner_profile(symbol: str, strategy_type: str):
    """Delete a saved scanner calibration profile (reverts to factory defaults)."""
    from app.services.backtest.scanner_profiles import delete_profile
    deleted = delete_profile(strategy_type.strip().lower(), symbol.upper())
    return {"deleted": deleted}


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
