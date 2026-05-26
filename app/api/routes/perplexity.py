from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional
import concurrent.futures
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

from app.config import get_settings
from app.services.backtest.perplexity_engine import run_perplexity_backtest
from app.services.backtest.portfolio_engine import run_portfolio_backtest
from app.services.backtest.walkforward_engine import (
    run_simple_split,
    run_rolling_walk_forward,
    run_walk_forward,
    run_walkforward_all,
)
from app.services.market_data.provider import get_ohlcv
from app.services.market_regime import get_current_regime, get_regime_risk_caps
from app.services.performance_breakdown import (
    breakdown_by_regime,
    breakdown_by_volatility,
    breakdown_by_r_bucket,
)
from app.services.perplexity.suitability import (
    load_suitability_config,
    save_suitability_config,
)
from app.services.risk.position_sizer import calculate_position_size
from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES, run_perplexity_signal

router = APIRouter(prefix="/perplexity", tags=["perplexity"])

_STRATEGY_MAP = {s.name: s for s in PERPLEXITY_STRATEGIES}


@router.get("/strategies")
def list_strategies():
    return [{"name": s.name, "enabled": s.enabled} for s in PERPLEXITY_STRATEGIES]


@router.get("/regime/{market}")
def momentum_regime(market: str = "us"):
    """Current momentum regime for a market (``us`` or ``india``).

    Tells the UI whether momentum strategies should be firing — India gates on
    Nifty 50 / India VIX, US on SPY / ^VIX.
    """
    from app.services.market_regime_advanced import get_momentum_regime
    snap = get_momentum_regime(market="india" if market.lower() == "india" else "us")
    d = snap.as_dict()
    d["market"] = "india" if market.lower() == "india" else "us"
    return d


@router.post("/strategies/{name}/toggle")
def toggle_strategy(name: str, enabled: bool = True):
    s = _STRATEGY_MAP.get(name)
    if not s:
        raise HTTPException(404, f"Strategy '{name}' not found")
    s.enabled = enabled
    return {"name": name, "enabled": s.enabled}


@router.get("/signals/{symbol}")
def get_signals(symbol: str):
    """
    Run all Perplexity strategies on the latest data for a symbol.
    BUY signals include position sizing based on account settings.
    """
    settings = get_settings()
    try:
        df = get_ohlcv(symbol.upper(), period="1y")
        if df.empty or len(df) < 60:
            raise HTTPException(400, f"Not enough data for {symbol} (need at least 60 bars)")
        regime = get_current_regime(df.index[-1])
        signals = run_perplexity_signal(symbol.upper(), df, regime=regime)
        regime_caps = get_regime_risk_caps(regime)
        results = []
        for s in signals:
            item = {
                "strategy": s.strategy_name,
                "direction": s.direction,
                "entry_price": s.entry_price,
                "stop_price": s.stop_price,
                "target_price": s.target_price,
                "confidence": round(s.confidence, 2),
                "reason": s.reason,
                "indicators": s.indicators,
                "regime": regime.value,
                "volatility_bucket": s.volatility_bucket,
                "suitability_blocked": s.suitability_blocked,
                "suitability_reason": s.suitability_reason,
                "position_size": None,
            }
            # Attach position sizing for BUY signals that have a stop price
            if s.direction == "BUY" and s.entry_price and s.stop_price and settings.position_sizing_enabled:
                sz = calculate_position_size(
                    symbol=symbol.upper(),
                    entry_price=s.entry_price,
                    stop_price=s.stop_price,
                    account_value=settings.account_value,
                    risk_pct_per_trade=regime_caps["risk_pct_per_trade"],
                    max_position_size_usd=settings.max_position_size_usd,
                    max_account_risk_pct=regime_caps["max_account_risk_pct"],
                )
                item["position_size"] = {
                    "shares": sz.shares,
                    "position_value": sz.position_value,
                    "risk_amount": sz.risk_amount,
                    "risk_pct_of_account": sz.risk_pct_of_account,
                    "stop_distance": sz.stop_distance,
                    "stop_distance_pct": sz.stop_distance_pct,
                    "capped": sz.capped,
                    "cap_reason": sz.cap_reason,
                    "viable": sz.viable,
                    "skip_reason": sz.skip_reason,
                }
            results.append(item)
        return results
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/scan")
def scan_universe(
    symbols: str = Query(..., description="Comma-separated list of tickers to scan"),
    direction: str = Query("BUY", description="Filter: BUY, SELL, or ALL"),
    min_confidence: float = Query(0.0, description="Minimum signal confidence 0–1"),
    strategies: Optional[str] = Query(None, description="Comma-separated strategy names to include (default: all enabled)"),
    max_workers: int = Query(8, description="Parallel fetch workers (max 20)"),
):
    """
    Scan a list of symbols for swing trading signals.
    Historical OHLCV bars come from yfinance (Schwab has no historical API).
    Live entry prices are refreshed from Schwab quotes; yfinance is the fallback.
    Returns one row per (symbol, strategy) where direction matches the filter.
    """
    import asyncio

    settings = get_settings()
    ticker_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not ticker_list:
        raise HTTPException(400, "No symbols provided")
    if len(ticker_list) > 500:
        raise HTTPException(400, "Maximum 500 symbols per scan")

    strategy_filter = {s.strip() for s in strategies.split(",")} if strategies else None
    max_workers = min(max_workers, 20)
    direction_filter = direction.upper()

    # Fetch live Schwab quotes for all symbols upfront (one batch call)
    live_prices: dict[str, float] = {}
    try:
        from app.services.brokers.schwab import SchwabBroker
        loop = asyncio.new_event_loop()
        try:
            schwab = SchwabBroker()
            loop.run_until_complete(schwab.authenticate())
            quotes = loop.run_until_complete(schwab.get_quotes(ticker_list))
            for sym, q in quotes.items():
                price = q.last or q.ask or q.bid
                if price and price > 0:
                    live_prices[sym] = float(price)
            logger.info("[scan] Schwab live prices loaded for %d/%d symbols", len(live_prices), len(ticker_list))
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("[scan] Schwab quotes unavailable, will use yfinance close prices: %s", exc)

    def _scan_one(symbol: str) -> list[dict]:
        try:
            df = get_ohlcv(symbol, period="1y")
            if df.empty or len(df) < 60:
                return []

            # Override the last close with live Schwab price if available
            live_price = live_prices.get(symbol)
            if live_price:
                df = df.copy()
                df.iloc[-1, df.columns.get_loc("Close")] = live_price

            regime = get_current_regime(df.index[-1])
            signals = run_perplexity_signal(symbol, df, regime=regime)
            regime_caps = get_regime_risk_caps(regime)
            hits = []
            for s in signals:
                if strategy_filter and s.strategy_name not in strategy_filter:
                    continue
                if s.suitability_blocked:
                    continue
                if direction_filter != "ALL" and s.direction != direction_filter:
                    continue
                if s.confidence < min_confidence:
                    continue
                item = {
                    "symbol":        symbol,
                    "strategy":      s.strategy_name,
                    "direction":     s.direction,
                    "entry_price":   s.entry_price,
                    "stop_price":    s.stop_price,
                    "target_price":  s.target_price,
                    "confidence":    round(s.confidence, 2),
                    "reason":        s.reason,
                    "regime":        regime.value,
                    "volatility_bucket": s.volatility_bucket,
                    "live_price":    live_price,
                    "price_source":  "schwab" if live_price else "yfinance",
                    "position_size": None,
                }
                if s.direction == "BUY" and s.entry_price and s.stop_price and settings.position_sizing_enabled:
                    try:
                        sz = calculate_position_size(
                            symbol=symbol,
                            entry_price=s.entry_price,
                            stop_price=s.stop_price,
                            account_value=settings.account_value,
                            risk_pct_per_trade=regime_caps["risk_pct_per_trade"],
                            max_position_size_usd=settings.max_position_size_usd,
                            max_account_risk_pct=regime_caps["max_account_risk_pct"],
                        )
                        item["position_size"] = {
                            "shares": sz.shares,
                            "position_value": sz.position_value,
                            "risk_amount": sz.risk_amount,
                            "risk_pct_of_account": sz.risk_pct_of_account,
                            "stop_distance_pct": sz.stop_distance_pct,
                            "viable": sz.viable,
                        }
                    except Exception:
                        pass
                hits.append(item)
            return hits
        except Exception as exc:
            logger.warning("[scan] %s failed: %s", symbol, exc)
            return []

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_one, sym): sym for sym in ticker_list}
        for fut in concurrent.futures.as_completed(futures):
            results.extend(fut.result())

    results.sort(key=lambda x: (-{"BUY": 2, "SELL": 1}.get(x["direction"], 0), -x["confidence"]))
    return {
        "scanned": len(ticker_list),
        "hits": len(results),
        "direction_filter": direction_filter,
        "min_confidence": min_confidence,
        "schwab_prices_loaded": len(live_prices),
        "results": results,
    }


@router.get("/atr/{symbol}")
def get_atr_stops(symbol: str):
    """
    Return current price + ATR(14)-based stop suggestions for the position sizer.
    Provides 1×, 1.5×, and 2× ATR stop levels so the user doesn't have to guess.
    """
    try:
        df = get_ohlcv(symbol.upper(), period="3mo")
        if df.empty or len(df) < 15:
            raise HTTPException(400, f"Not enough data for {symbol}")

        high = df["High"]
        low  = df["Low"]
        close = df["Close"]

        # True Range = max of (H-L, |H-prevC|, |L-prevC|)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])

        current_price = float(close.iloc[-1])
        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 4),
            "atr14": round(atr14, 4),
            "stop_1x_atr":   round(current_price - 1.0 * atr14, 4),
            "stop_1_5x_atr": round(current_price - 1.5 * atr14, 4),
            "stop_2x_atr":   round(current_price - 2.0 * atr14, 4),
            "stop_pct_1x":   round(1.0 * atr14 / current_price * 100, 2),
            "stop_pct_1_5x": round(1.5 * atr14 / current_price * 100, 2),
            "stop_pct_2x":   round(2.0 * atr14 / current_price * 100, 2),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/size")
def calculate_size(
    symbol: str,
    entry_price: float,
    stop_price: float,
    account_value: Optional[float] = None,
    risk_pct: Optional[float] = None,
    max_position_size_usd: Optional[float] = None,
):
    """Calculate position size for a given entry/stop. Uses account settings by default."""
    settings = get_settings()
    sz = calculate_position_size(
        symbol=symbol.upper(),
        entry_price=entry_price,
        stop_price=stop_price,
        account_value=account_value or settings.account_value,
        risk_pct_per_trade=risk_pct or settings.risk_pct_per_trade,
        max_position_size_usd=max_position_size_usd or settings.max_position_size_usd,
        max_account_risk_pct=settings.max_account_risk_pct,
    )
    return {
        "symbol": sz.symbol,
        "entry_price": sz.entry_price,
        "stop_price": sz.stop_price,
        "shares": sz.shares,
        "position_value": sz.position_value,
        "risk_amount": sz.risk_amount,
        "risk_pct_of_account": sz.risk_pct_of_account,
        "stop_distance": sz.stop_distance,
        "stop_distance_pct": sz.stop_distance_pct,
        "capped": sz.capped,
        "cap_reason": sz.cap_reason,
        "viable": sz.viable,
        "skip_reason": sz.skip_reason,
    }


@router.get("/backtest/{strategy_name}/{symbol}")
def backtest(
    strategy_name: str,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
    position_pct: float = 0.0,
    breakdown: bool = False,
):
    """Run a single Perplexity strategy backtest.
    position_pct: 0 = risk-based sizing (1% risk/trade); >0 = fixed % of capital per trade (e.g. 0.20 = 20%).
    """
    strategy = _STRATEGY_MAP.get(strategy_name)
    if not strategy:
        raise HTTPException(404, f"Strategy '{strategy_name}' not found")
    try:
        result = run_perplexity_backtest(strategy, symbol.upper(), period, initial_capital,
                                         position_pct=position_pct)
        return {
            "strategy_name": result.strategy_name,
            "symbol": result.symbol,
            "period": result.period,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_pnl": result.total_pnl,
            "capital_employed": result.capital_employed,
            "total_return_pct": result.total_return_pct,
            "cagr": result.cagr,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate_pct": result.win_rate_pct,
            "profit_factor": result.profit_factor,
            "avg_win_pct": result.avg_win_pct,
            "avg_loss_pct": result.avg_loss_pct,
            "expectancy_pct": result.expectancy_pct,
            "expectancy_r": result.expectancy_r,
            "average_holding_days": result.average_holding_days,
            "average_r_multiple": result.average_r_multiple,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "equity_curve": result.equity_curve,
            "trades": result.trades,
            **({
                "breakdown": {
                    "by_regime": {k: asdict(v) for k, v in breakdown_by_regime(result.trade_pairs, result.strategy_name, result.symbol, result.period, result.initial_capital).items()},
                    "by_volatility": {k: asdict(v) for k, v in breakdown_by_volatility(result.trade_pairs, result.strategy_name, result.symbol, result.period, result.initial_capital).items()},
                    "by_r_bucket": {k: asdict(v) for k, v in breakdown_by_r_bucket(result.trade_pairs, result.strategy_name, result.symbol, result.period, result.initial_capital).items()},
                }
            } if breakdown else {}),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/backtest-all/{symbol}")
def backtest_all(
    symbol: str,
    period: str = "5y",
    initial_capital: float = 100_000.0,
    position_pct: float = 0.0,
):
    """Run all 5 Perplexity strategies on one symbol and return a comparison table."""
    results = []
    for strategy in PERPLEXITY_STRATEGIES:
        try:
            r = run_perplexity_backtest(strategy, symbol.upper(), period, initial_capital,
                                        position_pct=position_pct)
            results.append({
                "strategy_name": r.strategy_name,
                "total_trades": r.total_trades,
                "win_rate_pct": r.win_rate_pct,
                "profit_factor": r.profit_factor,
                "avg_win_pct": r.avg_win_pct,
                "avg_loss_pct": r.avg_loss_pct,
                "expectancy_pct": r.expectancy_pct,
                "expectancy_r": r.expectancy_r,
                "average_holding_days": r.average_holding_days,
                "average_r_multiple": r.average_r_multiple,
                "total_return_pct": r.total_return_pct,
                "cagr": r.cagr,
                "total_pnl": r.total_pnl,
                "capital_employed": r.capital_employed,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
            })
        except Exception as exc:
            results.append({"strategy_name": strategy.name, "error": str(exc)})
    return results


@router.get("/suitability")
def get_suitability_config():
    return load_suitability_config()


@router.post("/suitability")
def update_suitability_config(config: dict):
    save_suitability_config(config)
    return {"updated": True, "config": config}


@router.get("/portfolio/{strategy_name}")
def portfolio_backtest(
    strategy_name: str,
    symbols: str = "SPY,QQQ,IWM,AAPL,NVDA",
    period: str = "5y",
    initial_capital: float = 100_000.0,
    position_pct: float = 0.20,
    max_open_positions: int = 5,
):
    """
    Run one strategy across a portfolio of symbols with shared capital.
    symbols: comma-separated list, e.g. SPY,QQQ,IWM,AAPL,NVDA
    position_pct: fraction of portfolio equity per position (0.20 = 20%)
    """
    strategy = _STRATEGY_MAP.get(strategy_name)
    if not strategy:
        raise HTTPException(404, f"Strategy '{strategy_name}' not found")
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        r = run_portfolio_backtest(
            strategy, sym_list, period=period,
            initial_capital=initial_capital,
            position_pct=position_pct,
            max_open_positions=max_open_positions,
        )
        return {
            "strategy_name": r.strategy_name,
            "symbols": r.symbols,
            "period": r.period,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "initial_capital": r.initial_capital,
            "final_capital": r.final_capital,
            "total_pnl": r.total_pnl,
            "total_return_pct": r.total_return_pct,
            "cagr": r.cagr,
            "total_trades": r.total_trades,
            "winning_trades": r.winning_trades,
            "losing_trades": r.losing_trades,
            "win_rate_pct": r.win_rate_pct,
            "profit_factor": r.profit_factor,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "capital_utilisation_pct": r.capital_utilisation_pct,
            "equity_curve": r.equity_curve,
            "trades": r.trades,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/filter-comparison/{strategy_name}/{symbol}")
def filter_comparison(
    strategy_name: str,
    symbol: str,
    period: str = "10y",
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
    ema_dist_min: float = 1.5,
    vol_min: float = 1.0,
    bb_pos_min: float = 0.72,
):
    """
    Run rolling walk-forward TWICE on the same strategy — once with filters OFF,
    once with the specified filters ON — entirely server-side so there is no
    race condition between set/run/reset calls.
    Returns both results for side-by-side comparison.
    """
    import copy
    from app.services.backtest.symbol_profiles import load_profile as _load_profile
    s_orig = _STRATEGY_MAP.get(strategy_name)
    if not s_orig:
        raise HTTPException(404, f"Strategy '{strategy_name}' not found")
    if not any(k.startswith("filter_") for k in s_orig.config):
        raise HTTPException(400, f"Strategy '{strategy_name}' does not support data-driven filters")

    # Load saved profile for this symbol to use its calibrated thresholds
    profile = _load_profile(strategy_name, symbol.upper())

    filter_keys = [k for k in s_orig.config if k.startswith("filter_")]

    # Work on isolated copies so concurrent requests don't corrupt each other's config
    s_before = copy.deepcopy(s_orig)
    s_after  = copy.deepcopy(s_orig)

    for k in filter_keys:
        s_before.config[k] = 0.0
        s_after.config[k]  = 0.0

    if profile:
        _apply_profile_to_config(strategy_name, s_after.config, profile)
    else:
        s_after.config["filter_ema_dist_min"] = ema_dist_min
        s_after.config["filter_vol_min"]      = vol_min
        s_after.config["filter_bb_pos_min"]   = bb_pos_min

    r_before = run_rolling_walk_forward(
        s_before, symbol.upper(), period=period,
        train_years=train_years, test_years=test_years, step_years=step_years,
        initial_capital=initial_capital, position_pct=position_pct,
    )
    r_after = run_rolling_walk_forward(
        s_after, symbol.upper(), period=period,
        train_years=train_years, test_years=test_years, step_years=step_years,
        initial_capital=initial_capital, position_pct=position_pct,
    )

    def _wf_summary(r):
        return {
            "global_wfe": r.global_wfe,
            "global_wfe_label": r.global_wfe_label,
            "global_oos_cagr": r.global_oos_cagr,
            "global_oos_total_return_pct": r.global_oos_total_return_pct,
            "avg_is_cagr": r.avg_is_cagr,
            "oos_composite_curve": r.oos_composite_curve,
            "segments": r.segments,
        }

    filters_applied = _profile_thresholds_dict(strategy_name, profile) if profile else {
        "ema_dist_min": ema_dist_min, "vol_min": vol_min, "bb_pos_min": bb_pos_min
    }

    return {
        "strategy_name": strategy_name,
        "symbol": symbol.upper(),
        "period": period,
        "filters": filters_applied,
        "before": _wf_summary(r_before),
        "after":  _wf_summary(r_after),
    }


def _apply_profile_to_config(strategy_name: str, config: dict, profile) -> None:
    """Apply calibrated filter thresholds to a strategy config dict."""
    if strategy_name == "EMA_Mean_Reversion":
        config["filter_ema_dist_min"] = profile.ema_dist_min
        config["filter_vol_min"]      = profile.vol_min
        config["filter_bb_pos_min"]   = profile.bb_pos_min
    elif strategy_name == "MA_Crossover_RSI":
        config["filter_vol_min"]        = profile.vol_min
        config["filter_ema_spread_min"] = profile.ema_spread_min
    elif strategy_name == "Breakout_Consolidation":
        config["filter_vol_min"]       = profile.vol_min
        config["filter_range_atr_max"] = profile.atr_pct_max   # stored in atr_pct_max field
    elif strategy_name == "BB_Mean_Reversion":
        config["filter_vol_min"]          = profile.vol_min
        config["filter_atr_pct_max"]      = profile.atr_pct_max
        config["filter_bb_depth_min"]     = profile.bb_depth_min
        config["filter_rsi_min"]          = profile.rsi_min
        config["filter_bb_pct_min"]       = profile.bb_pct_min
        config["filter_ema_dist_pct_min"] = profile.ema_dist_pct_min
    elif strategy_name == "Fib_Pullback_Support":
        config["filter_lower_wick_min"] = profile.lower_wick_min
        config["filter_vol_min"]        = profile.vol_min
        config["filter_body_max"]       = profile.body_max
    elif strategy_name == "RSI_Swing_Reversal":
        config["filter_vol_min"]      = profile.vol_min
        config["filter_swing_rsi_min"] = profile.swing_rsi_min


def _profile_thresholds_dict(strategy_name: str, profile) -> dict:
    """Return strategy-specific threshold keys for the API response."""
    if strategy_name == "EMA_Mean_Reversion":
        return {"ema_dist_min": profile.ema_dist_min, "vol_min": profile.vol_min, "bb_pos_min": profile.bb_pos_min}
    if strategy_name == "MA_Crossover_RSI":
        return {"vol_min": profile.vol_min, "ema_spread_min": profile.ema_spread_min}
    if strategy_name == "Breakout_Consolidation":
        return {"vol_min": profile.vol_min, "range_atr_max": profile.atr_pct_max}
    if strategy_name == "BB_Mean_Reversion":
        return {
            "vol_min":          profile.vol_min,
            "atr_pct_max":      profile.atr_pct_max,
            "bb_depth_min":     profile.bb_depth_min,
            "rsi_min":          profile.rsi_min,
            "bb_pct_min":       profile.bb_pct_min,
            "ema_dist_pct_min": profile.ema_dist_pct_min,
        }
    if strategy_name == "Fib_Pullback_Support":
        return {"lower_wick_min": profile.lower_wick_min, "vol_min": profile.vol_min,
                "body_max": profile.body_max}
    if strategy_name == "RSI_Swing_Reversal":
        return {"vol_min": profile.vol_min, "swing_rsi_min": profile.swing_rsi_min}
    return {}


@router.get("/calibrate/{strategy_name}/{symbol}")
def auto_calibrate(
    strategy_name: str,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
    verify_wf: bool = True,
):
    """
    Auto-calibrate per-symbol filter thresholds for a strategy:
    1. Run backtest → extract trade snapshots
    2. Find optimal thresholds from winning trade distributions
    3. Optionally verify with before/after walk-forward
    4. Save profile to disk
    """
    from app.services.backtest.trade_analyzer import analyze_trades as _analyze
    from app.services.backtest.symbol_profiles import calibrate_from_snapshots, save_profile
    import copy

    strategy = _STRATEGY_MAP.get(strategy_name)
    if not strategy:
        raise HTTPException(404, f"Strategy '{strategy_name}' not found")
    # Check strategy supports calibration (has at least one filter_ key)
    if not any(k.startswith("filter_") for k in strategy.config):
        raise HTTPException(400, f"Strategy '{strategy_name}' does not support per-symbol calibration")

    # Collect all filter keys for this strategy so we can zero them before backtesting
    filter_keys = [k for k in strategy.config if k.startswith("filter_")]

    try:
        df = get_ohlcv(symbol.upper(), period=period)
        if df.empty or len(df) < 120:
            raise HTTPException(400, f"Not enough data for {symbol}")

        # Work on an isolated copy — never mutate the global strategy object
        s_raw = copy.deepcopy(strategy)
        for k in filter_keys:
            s_raw.config[k] = 0.0

        result   = run_perplexity_backtest(s_raw, symbol.upper(), period,
                                           initial_capital, position_pct=position_pct)
        snapshots = _analyze(result.trades, df, strategy_name=strategy_name)

        if len(snapshots) < 6:
            raise HTTPException(400, f"Not enough trades to calibrate ({len(snapshots)} trades, need ≥ 6)")

        # Pass all captured indicators — calibrate_from_snapshots selects per strategy
        profile = calibrate_from_snapshots(strategy_name, symbol.upper(),
                                           [{"outcome":        s.outcome,
                                             "rsi":            s.rsi,
                                             "ema_dist_pct":   s.ema_dist_pct,
                                             "volume_ratio":   s.volume_ratio,
                                             "bb_pct":         s.bb_pct,
                                             "atr_pct":        s.atr_pct,
                                             "lower_wick_pct": s.lower_wick_pct,
                                             "body_pct":       s.body_pct,
                                             "ema_spread_pct": s.ema_spread_pct,
                                             "range_atr_ratio":s.range_atr_ratio,
                                             "bb_depth_pct":   s.bb_depth_pct}
                                            for s in snapshots])

        # ── IS verification: check filters actually improve win rate on the same data ──
        # Run a filtered backtest and compare win rate + survival rate vs baseline.
        # Criteria to save: filtered_win_rate >= baseline_win_rate + 3pp AND survival >= 40%
        baseline_wr  = result.win_rate_pct
        baseline_trades = result.total_trades
        s_filtered = copy.deepcopy(strategy)
        for k in filter_keys:
            s_filtered.config[k] = 0.0
        _apply_profile_to_config(strategy_name, s_filtered.config, profile)
        r_filtered = run_perplexity_backtest(
            s_filtered, symbol.upper(), period, initial_capital, position_pct=position_pct
        )
        filtered_wr      = r_filtered.win_rate_pct
        filtered_trades  = r_filtered.total_trades
        survival_rate    = filtered_trades / baseline_trades if baseline_trades > 0 else 0.0

        # ── Save gate: filters must EARN their place ──────────────────────────────────
        # A filter is worth saving if it materially improves EITHER win rate OR
        # profitability (a filter can lift expectancy/return by cutting losers even
        # when win rate barely moves), AND it keeps enough trades to be meaningful.
        # All three "improvement" measures are compared on the SAME in-sample data.
        WR_GAIN_MIN   = 3.0    # percentage points
        EXP_GAIN_MIN  = 0.05   # +0.05pp expectancy per trade
        RET_GAIN_MIN  = 2.0    # +2pp total return over the window
        wr_improved   = (filtered_wr >= baseline_wr + WR_GAIN_MIN)
        exp_improved  = (r_filtered.expectancy_pct >= result.expectancy_pct + EXP_GAIN_MIN)
        ret_improved  = (r_filtered.total_return_pct >= result.total_return_pct + RET_GAIN_MIN)
        has_survival  = (survival_rate >= 0.40)
        is_improvement = has_survival and (wr_improved or exp_improved or ret_improved)

        def _pf(v):  # profit_factor can be None when there are no losing trades
            return round(v, 2) if isinstance(v, (int, float)) else None

        # Baseline-vs-filtered comparison, returned in BOTH branches so the UI can
        # always show what the filter does — even when we decline to save it.
        comparison = {
            "baseline": {
                "trades":        result.total_trades,
                "win_rate_pct":  round(result.win_rate_pct, 1),
                "avg_win_pct":   round(result.avg_win_pct, 2),
                "avg_loss_pct":  round(result.avg_loss_pct, 2),
                "expectancy_pct": round(result.expectancy_pct, 3),
                "total_return_pct": round(result.total_return_pct, 2),
                "profit_factor": _pf(result.profit_factor),
                "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            },
            "filtered": {
                "trades":        r_filtered.total_trades,
                "win_rate_pct":  round(filtered_wr, 1),
                "avg_win_pct":   round(r_filtered.avg_win_pct, 2),
                "avg_loss_pct":  round(r_filtered.avg_loss_pct, 2),
                "expectancy_pct": round(r_filtered.expectancy_pct, 3),
                "total_return_pct": round(r_filtered.total_return_pct, 2),
                "profit_factor": _pf(r_filtered.profit_factor),
                "max_drawdown_pct": round(r_filtered.max_drawdown_pct, 2),
            },
            "survival_rate_pct": round(survival_rate * 100, 1),
            "improved_by": [m for m, ok in
                            (("win_rate", wr_improved), ("expectancy", exp_improved),
                             ("total_return", ret_improved)) if ok],
        }

        # ── Optional walk-forward verification (informational — does NOT block saving) ──
        if verify_wf and len(df) >= 500:
            try:
                wf_period = "5y" if len(df) < 1500 else "10y"
                s_wf_before = copy.deepcopy(strategy)
                s_wf_after  = copy.deepcopy(strategy)
                for k in filter_keys:
                    s_wf_before.config[k] = 0.0
                    s_wf_after.config[k]  = 0.0
                _apply_profile_to_config(strategy_name, s_wf_after.config, profile)

                r_before = run_rolling_walk_forward(
                    s_wf_before, symbol.upper(), period=wf_period,
                    train_years=2.0, test_years=1.0, step_years=1.0,
                    initial_capital=initial_capital, position_pct=position_pct,
                )
                r_after = run_rolling_walk_forward(
                    s_wf_after, symbol.upper(), period=wf_period,
                    train_years=2.0, test_years=1.0, step_years=1.0,
                    initial_capital=initial_capital, position_pct=position_pct,
                )

                profile.wfe_before      = r_before.global_wfe
                profile.wfe_after       = r_after.global_wfe
                profile.oos_cagr_before = r_before.global_oos_cagr
                profile.oos_cagr_after  = r_after.global_oos_cagr
                wfe_improved  = (r_after.global_wfe  or 0) > (r_before.global_wfe  or 0)
                cagr_improved = (r_after.global_oos_cagr or 0) > (r_before.global_oos_cagr or 0)
                profile.verified = wfe_improved or cagr_improved
            except Exception:
                pass

        # ── Save if IS improvement criterion met ──────────────────────────────────────
        # WFE is informational — it tells you if the filter also works OOS,
        # but low OOS trade counts make WFE unreliable as a gate.
        if not is_improvement:
            if not has_survival:
                reason = (
                    f"Filters kept only {survival_rate:.0%} of trades (need ≥40%) — too aggressive. "
                    f"The thresholds would over-prune this symbol's history."
                )
            else:
                reason = (
                    f"Filters didn't materially improve any metric on this data "
                    f"(win rate {baseline_wr:.1f}%→{filtered_wr:.1f}%, need +{WR_GAIN_MIN:.0f}pp; "
                    f"expectancy {result.expectancy_pct:.2f}%→{r_filtered.expectancy_pct:.2f}%; "
                    f"return {result.total_return_pct:.1f}%→{r_filtered.total_return_pct:.1f}%). "
                    f"The strategy already performs well on this symbol — no filtering needed."
                )
            return {
                "symbol": profile.symbol,
                "strategy": profile.strategy,
                "calibrated_at": profile.calibrated_at,
                "n_trades": profile.n_trades,
                "n_wins": profile.n_wins,
                "win_rate_pct": profile.win_rate_pct,
                "filtered_win_rate_pct": filtered_wr,
                "filtered_trades": filtered_trades,
                "survival_rate_pct": round(survival_rate * 100, 1),
                "thresholds": _profile_thresholds_dict(strategy_name, profile),
                "comparison": comparison,
                "verification": {
                    "wfe_before":      profile.wfe_before,
                    "wfe_after":       profile.wfe_after,
                    "oos_cagr_before": profile.oos_cagr_before,
                    "oos_cagr_after":  profile.oos_cagr_after,
                    "verified":        profile.verified,
                },
                "saved": False,
                "skip_reason": reason,
            }

        save_profile(profile)

        # Build strategy-specific thresholds response
        thresholds = _profile_thresholds_dict(strategy_name, profile)

        return {
            "symbol": profile.symbol,
            "strategy": profile.strategy,
            "calibrated_at": profile.calibrated_at,
            "n_trades": profile.n_trades,
            "n_wins": profile.n_wins,
            "win_rate_pct": profile.win_rate_pct,
            "thresholds": thresholds,
            "evidence": {
                "win_ema_dist_mean":  profile.win_ema_dist_mean,
                "loss_ema_dist_mean": profile.loss_ema_dist_mean,
                "win_vol_mean":       profile.win_vol_mean,
                "loss_vol_mean":      profile.loss_vol_mean,
                "win_bb_pos_mean":    profile.win_bb_pos_mean,
                "loss_bb_pos_mean":   profile.loss_bb_pos_mean,
            },
            "filtered_win_rate_pct": filtered_wr,
            "filtered_trades": filtered_trades,
            "survival_rate_pct": round(survival_rate * 100, 1),
            "comparison": comparison,
            "verification": {
                "wfe_before":       profile.wfe_before,
                "wfe_after":        profile.wfe_after,
                "oos_cagr_before":  profile.oos_cagr_before,
                "oos_cagr_after":   profile.oos_cagr_after,
                "verified":         profile.verified,
            },
            "saved": True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/profiles/{strategy_name}")
def list_profiles(strategy_name: str):
    """List all saved per-symbol filter profiles for a strategy."""
    from app.services.backtest.symbol_profiles import list_profiles as _list
    profiles = _list(strategy_name)
    from dataclasses import asdict
    return [asdict(p) for p in profiles]


@router.get("/profiles/{strategy_name}/{symbol}")
def get_profile(strategy_name: str, symbol: str):
    """Get a single saved filter profile for a strategy+symbol, or {} if none exists."""
    from app.services.backtest.symbol_profiles import load_profile
    from dataclasses import asdict
    profile = load_profile(strategy_name, symbol.upper())
    if profile is None:
        return {}
    return asdict(profile)


@router.delete("/profiles/{strategy_name}/{symbol}")
def delete_profile(strategy_name: str, symbol: str):
    """Delete a saved per-symbol filter profile."""
    from app.services.backtest.symbol_profiles import delete_profile as _delete
    deleted = _delete(strategy_name, symbol.upper())
    if not deleted:
        raise HTTPException(404, f"No profile for {strategy_name}/{symbol}")
    return {"deleted": True, "symbol": symbol.upper()}


@router.get("/analyze/{strategy_name}/{symbol}")
def analyze_trades(
    strategy_name: str,
    symbol: str,
    period: str = "5y",
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
):
    """
    Run a backtest then analyze winning vs losing trade entry conditions.
    Returns per-trade snapshots + pattern insights + time breakdown.
    """
    from app.services.backtest.trade_analyzer import analyze_trades as _analyze, find_patterns, time_breakdown
    strategy = _STRATEGY_MAP.get(strategy_name)
    if not strategy:
        raise HTTPException(404, f"Strategy '{strategy_name}' not found")
    try:
        df = get_ohlcv(symbol.upper(), period=period)
        if df.empty or len(df) < 120:
            raise HTTPException(400, f"Not enough data for {symbol}")
        result = run_perplexity_backtest(strategy, symbol.upper(), period, initial_capital,
                                         position_pct=position_pct)
        if result.total_trades == 0:
            return {
                "strategy_name": result.strategy_name,
                "symbol": result.symbol,
                "period": result.period,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "snapshots": [],
                "patterns": [],
                "timing": {},
                "message": (
                    f"No trades were generated for {strategy_name} on {symbol.upper()} "
                    f"over {period}. The market regime filter (golden cross + SMA200) may "
                    f"have been inactive for most of this period, or no entry conditions "
                    f"were met. Try a shorter period (3y/5y) or a different symbol."
                ),
            }
        snapshots = _analyze(result.trades, df, strategy_name=strategy_name)
        patterns  = find_patterns(snapshots)
        timing    = time_breakdown(snapshots)
        return {
            "strategy_name": result.strategy_name,
            "symbol": result.symbol,
            "period": result.period,
            "total_trades": result.total_trades,
            "win_rate_pct": result.win_rate_pct,
            "snapshots": [
                {
                    "date": s.date, "outcome": s.outcome,
                    "pnl": s.pnl, "pnl_pct": s.pnl_pct,
                    "hold_bars": s.hold_bars,
                    "rsi": s.rsi, "atr_pct": s.atr_pct,
                    "ema_dist_pct": s.ema_dist_pct,
                    "sma200_dist_pct": s.sma200_dist_pct,
                    "bb_pct": s.bb_pct, "volume_ratio": s.volume_ratio,
                    "body_pct": s.body_pct,
                    "lower_wick_pct": s.lower_wick_pct,
                    "upper_wick_pct": s.upper_wick_pct,
                    "month": s.month, "day_of_week": s.day_of_week,
                    "quarter": s.quarter,
                    "prior_trade_won": s.prior_trade_won,
                    "bars_since_last_trade": s.bars_since_last_trade,
                    "ema_spread_pct": s.ema_spread_pct,
                    "range_atr_ratio": s.range_atr_ratio,
                    "bb_depth_pct": s.bb_depth_pct,
                    "fib_level": s.fib_level,
                }
                for s in snapshots
            ],
            "patterns": [
                {
                    "indicator": p.indicator,
                    "description": p.description,
                    "win_mean": p.win_mean,
                    "loss_mean": p.loss_mean,
                    "separation": p.separation,
                    "recommendation": p.recommendation,
                    "direction": p.direction,
                    "suggested_min": p.suggested_min,
                    "suggested_max": p.suggested_max,
                }
                for p in patterns
            ],
            "timing": timing,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/walkforward-all/{symbol}")
def walk_forward_all(
    symbol: str,
    period: str = "10y",
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
):
    """
    Run rolling walk-forward for ALL 5 strategies on one symbol.
    Returns a comparison table: Global WFE, latest IS/OOS CAGR, prev OOS CAGR.
    """
    from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES
    try:
        summaries = run_walkforward_all(
            PERPLEXITY_STRATEGIES, symbol.upper(), period=period,
            train_years=train_years, test_years=test_years, step_years=step_years,
            initial_capital=initial_capital, position_pct=position_pct,
        )
        return [
            {
                "strategy_name": s.strategy_name,
                "n_windows": s.n_windows,
                "global_wfe": s.global_wfe,
                "global_wfe_label": s.global_wfe_label,
                "global_oos_cagr": s.global_oos_cagr,
                "global_oos_total_return_pct": s.global_oos_total_return_pct,
                "avg_is_cagr": s.avg_is_cagr,
                "latest_is_cagr": s.latest_is_cagr,
                "latest_oos_cagr": s.latest_oos_cagr,
                "latest_wfe": s.latest_wfe,
                "latest_wfe_label": s.latest_wfe_label,
                "prev_oos_cagr": s.prev_oos_cagr,
                "error": s.error,
            }
            for s in summaries
        ]
    except Exception as exc:
        raise HTTPException(500, str(exc))


def _seg_dict(seg) -> dict:
    return {
        "start": seg.start, "end": seg.end, "bars": seg.bars,
        "total_return_pct": seg.total_return_pct, "cagr": seg.cagr,
        "win_rate_pct": seg.win_rate_pct, "profit_factor": seg.profit_factor,
        "max_drawdown_pct": seg.max_drawdown_pct, "trades": seg.trades,
        "sharpe": seg.sharpe,
        "equity_curve": seg.equity_curve,
    }


@router.get("/walkforward/{strategy_name}/{symbol}")
def walk_forward(
    strategy_name: str,
    symbol: str,
    mode: str = "simple",               # "simple" | "rolling"
    period: str = "10y",
    train_pct: float = 0.70,            # simple mode only
    train_years: float = 3.0,           # rolling mode only
    test_years: float = 1.0,            # rolling mode only
    step_years: float = 1.0,            # rolling mode only
    initial_capital: float = 10_000.0,
    position_pct: float = 0.0,
):
    """
    Walk-forward validation.
    mode=simple : one IS/OOS split at train_pct of bars.
    mode=rolling: multiple IS/OOS windows sliding by step_years.
    """
    strategy = _STRATEGY_MAP.get(strategy_name)
    if not strategy:
        raise HTTPException(404, f"Strategy '{strategy_name}' not found")
    try:
        if mode == "rolling":
            r = run_rolling_walk_forward(
                strategy, symbol.upper(), period=period,
                train_years=train_years, test_years=test_years, step_years=step_years,
                initial_capital=initial_capital, position_pct=position_pct,
            )
            return {
                "mode": "rolling",
                "strategy_name": r.strategy_name,
                "symbol": r.symbol,
                "full_period": r.full_period,
                "train_years": r.train_years,
                "test_years": r.test_years,
                "step_years": r.step_years,
                "segments": r.segments,
                "oos_composite_curve": r.oos_composite_curve,
                "global_oos_total_return_pct": r.global_oos_total_return_pct,
                "global_oos_cagr": r.global_oos_cagr,
                "avg_is_cagr": r.avg_is_cagr,
                "global_wfe": r.global_wfe,
                "global_wfe_label": r.global_wfe_label,
            }
        else:
            r = run_simple_split(
                strategy, symbol.upper(), period=period,
                train_pct=train_pct, initial_capital=initial_capital,
                position_pct=position_pct,
            )
            return {
                "mode": "simple",
                "strategy_name": r.strategy_name,
                "symbol": r.symbol,
                "full_period": r.full_period,
                "train_pct": r.train_pct,
                "is": _seg_dict(r.is_segment),
                "oos": _seg_dict(r.oos_segment),
                "wfe": r.wfe,
                "wfe_label": r.wfe_label,
                "oos_pf_ratio": r.oos_pf_ratio,
                # legacy keys so existing UI doesn't break
                "train": {**_seg_dict(r.is_segment), "equity_curve": r.is_segment.equity_curve},
                "test":  {**_seg_dict(r.oos_segment), "equity_curve": r.oos_segment.equity_curve},
                "oos_return_ratio": r.wfe,
                "oos_pf_ratio": r.oos_pf_ratio,
            }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
