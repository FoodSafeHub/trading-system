from __future__ import annotations

import os
import requests

BASE = os.getenv("TRADING_API_BASE", "https://127.0.0.1:8001")


def _load_bearer_token() -> str:
    """Read API_BEARER_TOKEN from env, falling back to .env file.

    Streamlit doesn't auto-load .env, so we parse it ourselves on first import.
    Keeps the dashboard usable without forcing the user to export the var.
    """
    tok = os.getenv("API_BEARER_TOKEN", "")
    if tok:
        return tok
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(here, ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("API_BEARER_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


_BEARER = _load_bearer_token()


def _headers() -> dict:
    return {"Authorization": f"Bearer {_BEARER}"} if _BEARER else {}


def _merge_headers(kwargs: dict) -> dict:
    """Merge the bearer header into caller-provided headers without clobbering."""
    extra = kwargs.pop("headers", None) or {}
    merged = _headers()
    merged.update(extra)
    if merged:
        kwargs["headers"] = merged
    return kwargs


def _raise_with_body(r: requests.Response) -> None:
    """Surface the backend's detail/message in HTTP errors instead of a generic stack trace."""
    if r.status_code < 400:
        return
    detail = ""
    try:
        body = r.json()
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("message") or ""
        if not detail:
            detail = str(body)[:300]
    except Exception:
        detail = (r.text or "")[:300]
    raise requests.HTTPError(f"{r.status_code} {r.reason}: {detail}", response=r)


def _get(path: str, timeout: int = 10, **kwargs):
    kwargs = _merge_headers(kwargs)
    r = requests.get(f"{BASE}{path}", timeout=timeout, verify=False, **kwargs)
    _raise_with_body(r)
    return r.json()


def _post(path: str, timeout: int = 10, **kwargs):
    kwargs = _merge_headers(kwargs)
    r = requests.post(f"{BASE}{path}", timeout=timeout, verify=False, **kwargs)
    _raise_with_body(r)
    return r.json()


def _delete(path: str, timeout: int = 10, **kwargs):
    kwargs = _merge_headers(kwargs)
    r = requests.delete(f"{BASE}{path}", timeout=timeout, verify=False, **kwargs)
    _raise_with_body(r)
    return r.json()


def health():
    return _get("/health")

def scanner_status():
    return _get("/scanner/status")

def scanner_run(config: dict):
    return _post("/scanner/run", json=config)

def scanner_results(limit: int = 50):
    return _get(f"/scanner/results?limit={limit}")

def scanner_latest():
    return _get("/scanner/latest")

def upstox_resolve(symbol: str):
    """Validate a free-typed NSE ticker. Returns {symbol, tradeable, instrument_key}."""
    return _get(f"/upstox/resolve/{symbol}")

def upstox_universe(tier: str):
    """India tier symbol list: nifty50/nifty100/nifty200/nifty500/nse_all."""
    return _get(f"/upstox/universe/{tier}")

def momentum_regime(market: str = "india"):
    """Momentum regime snapshot for a market ('india' or 'us')."""
    return _get(f"/perplexity/regime/{market}")

def account_summary():
    # Longer timeout: under trade_routing=both this fans out to every broker, and
    # the Webull leg can take ~15s. The default 10s timeout would fire first,
    # making Home's _safe() wrapper drop ALL broker tabs (Schwab included).
    return _get("/account/summary", timeout=30)

def positions():
    return _get("/account/positions", timeout=30)

def broker_account_summary(broker: str):
    """Accounts for a specific broker, independent of global routing."""
    return _get(f"/account/{broker}/summary")

def broker_positions(broker: str):
    """Positions for a specific broker, independent of global routing."""
    return _get(f"/account/{broker}/positions")

def schwab_status():
    """Schwab connection health (token presence/expiry), no token material."""
    return _get("/schwab/status")

def webull_status():
    """Webull connection health — credential presence + live account ping."""
    return _get("/webull/status", timeout=15)

def schwab_auth_url():
    """Get the Schwab OAuth authorization URL to open in the browser."""
    return _get("/schwab/auth")

def risk_status():
    return _get("/risk/status")

def set_kill_switch(active: bool):
    return _post(f"/risk/kill-switch?active={str(active).lower()}")

def live_quotes(symbols: str):
    return _get(f"/account/quotes?symbols={symbols}")

def orders():
    return _get("/orders")

def signals():
    return _get("/signals")

def signal_runs():
    return _get("/signals/runs")

def strategy_configs():
    return _get("/strategy/configs")

def run_strategy():
    return _post("/strategy/run")

def run_scheduler_now(dry_run: bool = True):
    return _post("/strategy/scheduler/run_now", timeout=180, params={"dry_run": str(dry_run).lower()})

def scheduler_status():
    return _get("/strategy/scheduler")

def update_scheduler_config(run_bollinger: bool | None = None, run_perplexity: bool | None = None):
    params = {}
    if run_bollinger is not None:
        params["run_bollinger"] = str(run_bollinger).lower()
    if run_perplexity is not None:
        params["run_perplexity"] = str(run_perplexity).lower()
    r = requests.post(f"{BASE}/strategy/scheduler/config", params=params, timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()

def chart_data(symbol: str, period: str = "3mo", interval: str = "1d"):
    return _get(f"/strategy/chart/{symbol}", params={"period": period, "interval": interval})


def intraday_chart(symbol: str, timeframe: str = "5m", strategies: str = "all",
                   include_rejected: bool = True):
    """Unified live-chart payload: candles + backend overlays + accepted/rejected markers."""
    return _get(
        f"/chart/intraday/{symbol}",
        params={
            "timeframe": timeframe,
            "strategies": strategies,
            "include_rejected": str(include_rejected).lower(),
        },
        timeout=30,
    )


def chart_strategies():
    return _get("/chart/strategies")

def list_assignments():
    return _get("/assignments")

def upsert_assignment(symbol: str, system: str, strategy_name: str, enabled: bool = True,
                      notes: str = "", max_capital_usd: float | None = None,
                      max_shares: float | None = None,
                      broker: str = "default",
                      tight_trail_pct: float | None = None):
    return _post("/assignments", json={"symbol": symbol, "system": system,
                                       "strategy_name": strategy_name, "enabled": enabled,
                                       "notes": notes,
                                       "max_capital_usd": max_capital_usd,
                                       "max_shares": max_shares,
                                       "broker": broker,
                                       "tight_trail_pct": tight_trail_pct})


def set_assignment_broker(symbol: str, broker: str):
    r = requests.patch(f"{BASE}/assignments/{symbol}/broker",
                       params={"broker": broker},
                       timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()


def bulk_set_assignment_broker(symbols: list[str] | None = None, broker: str = "default",
                               auto_by_market: bool = False):
    """Set the broker route on many assignments at once.

    auto_by_market=True routes each symbol by its market (India -> zerodha,
    US -> default). Otherwise every listed symbol is set to `broker`. An empty
    symbols list with auto_by_market applies to all existing assignments.
    """
    return _post("/assignments/bulk-broker", json={
        "symbols": symbols or [],
        "broker": broker,
        "auto_by_market": auto_by_market,
    })

def toggle_assignment(symbol: str, enabled: bool):
    r = requests.patch(f"{BASE}/assignments/{symbol}/toggle", params={"enabled": str(enabled).lower()}, timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()

def set_assignment_cap(symbol: str, max_capital_usd: float | None):
    # When clearing the cap, send no value at all (FastAPI's Optional[float]
    # parameter accepts a missing query param as None, but rejects "").
    params = {"max_capital_usd": max_capital_usd} if max_capital_usd else {}
    r = requests.patch(f"{BASE}/assignments/{symbol}/cap",
                       params=params,
                       timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()

def set_assignment_trail(symbol: str, tight_trail_pct: float | None):
    r = requests.patch(f"{BASE}/assignments/{symbol}/trail",
                       params={"tight_trail_pct": tight_trail_pct} if tight_trail_pct else {},
                       timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()

def set_assignment_shares(symbol: str, max_shares: float | None):
    params = {"max_shares": max_shares} if max_shares else {}
    r = requests.patch(f"{BASE}/assignments/{symbol}/shares",
                       params=params,
                       timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()

def delete_assignment(symbol: str):
    r = requests.delete(f"{BASE}/assignments/{symbol}", timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()

def get_atr_stops(symbol: str):
    return _get(f"/perplexity/atr/{symbol}")

def place_order(symbol: str, side: str, order_type: str, quantity: float, limit_price: float | None = None):
    payload = {"symbol": symbol, "side": side, "order_type": order_type, "quantity": quantity}
    if limit_price:
        payload["limit_price"] = limit_price
    return _post("/orders/manual", json=payload)

# ── Perplexity strategy API ───────────────────────────────────

def perplexity_strategies():
    return _get("/perplexity/strategies")

def perplexity_toggle_strategy(name: str, enabled: bool):
    r = requests.post(f"{BASE}/perplexity/strategies/{name}/toggle",
                      params={"enabled": str(enabled).lower()}, timeout=10, verify=False, headers=_headers())
    r.raise_for_status()
    return r.json()

def perplexity_signals(symbol: str):
    return _get(f"/perplexity/signals/{symbol}", timeout=30)

def perplexity_scan(symbols: list[str], direction: str = "BUY",
                    min_confidence: float = 0.0, strategies: str | None = None,
                    max_workers: int = 8):
    params = {
        "symbols": ",".join(symbols),
        "direction": direction,
        "min_confidence": min_confidence,
        "max_workers": max_workers,
    }
    if strategies:
        params["strategies"] = strategies
    return _get("/perplexity/scan", params=params, timeout=300)

def perplexity_atr(symbol: str):
    return _get(f"/perplexity/atr/{symbol}")

def perplexity_size(symbol: str, entry_price: float, stop_price: float,
                    account_value: float, risk_pct: float, max_position_size_usd: float):
    return _get("/perplexity/size", params={
        "symbol": symbol, "entry_price": entry_price, "stop_price": stop_price,
        "account_value": account_value, "risk_pct": risk_pct,
        "max_position_size_usd": max_position_size_usd,
    })

def perplexity_backtest(strategy_name: str, symbol: str, period: str = "5y",
                        initial_capital: float = 10000, position_pct: float = 0.0,
                        breakdown: bool = True, timeout: int = 180):
    return _get(f"/perplexity/backtest/{strategy_name}/{symbol}",
                params={"period": period, "initial_capital": initial_capital,
                        "position_pct": position_pct, "breakdown": str(breakdown).lower()},
                timeout=timeout)

def perplexity_backtest_all(symbol: str, period: str = "5y", initial_capital: float = 10000,
                            position_pct: float = 0.0, timeout: int = 600):
    return _get(f"/perplexity/backtest-all/{symbol}",
                params={"period": period, "initial_capital": initial_capital, "position_pct": position_pct},
                timeout=timeout)

def perplexity_portfolio(strategy_name: str, symbols: str, period: str = "5y",
                         initial_capital: float = 10000, position_pct: float = 0.20,
                         max_open_positions: int = 5, timeout: int = 300):
    return _get(f"/perplexity/portfolio/{strategy_name}",
                params={"symbols": symbols, "period": period, "initial_capital": initial_capital,
                        "position_pct": position_pct, "max_open_positions": max_open_positions},
                timeout=timeout)

def perplexity_walkforward(strategy_name: str, symbol: str, mode: str = "rolling",
                           period: str = "10y", train_pct: float = 0.70,
                           train_years: float = 3.0, test_years: float = 1.0, step_years: float = 1.0,
                           initial_capital: float = 10000, position_pct: float = 0.0,
                           timeout: int = 600):
    return _get(f"/perplexity/walkforward/{strategy_name}/{symbol}",
                params={"mode": mode, "period": period, "train_pct": train_pct,
                        "train_years": train_years, "test_years": test_years, "step_years": step_years,
                        "initial_capital": initial_capital, "position_pct": position_pct},
                timeout=timeout)

def perplexity_walkforward_all(symbol: str, period: str = "10y",
                               train_years: float = 3.0, test_years: float = 1.0, step_years: float = 1.0,
                               initial_capital: float = 10000, position_pct: float = 0.0,
                               timeout: int = 720):
    return _get(f"/perplexity/walkforward-all/{symbol}",
                params={"period": period, "train_years": train_years, "test_years": test_years,
                        "step_years": step_years, "initial_capital": initial_capital,
                        "position_pct": position_pct},
                timeout=timeout)

def perplexity_analyze(strategy_name: str, symbol: str, period: str = "5y",
                       initial_capital: float = 10000, position_pct: float = 0.0,
                       timeout: int = 180):
    return _get(f"/perplexity/analyze/{strategy_name}/{symbol}",
                params={"period": period, "initial_capital": initial_capital, "position_pct": position_pct},
                timeout=timeout)

def perplexity_calibrate(strategy_name: str, symbol: str, period: str = "5y",
                         initial_capital: float = 10000, position_pct: float = 0.0,
                         verify_wf: bool = True, timeout: int = 600):
    return _get(f"/perplexity/calibrate/{strategy_name}/{symbol}",
                params={"period": period, "initial_capital": initial_capital,
                        "position_pct": position_pct, "verify_wf": str(verify_wf).lower()},
                timeout=timeout)

def perplexity_filter_comparison(strategy_name: str, symbol: str, period: str = "10y",
                                 train_years: float = 3.0, test_years: float = 1.0, step_years: float = 1.0,
                                 initial_capital: float = 10000, position_pct: float = 0.0,
                                 timeout: int = 900, **filter_params):
    params = {"period": period, "train_years": train_years, "test_years": test_years,
              "step_years": step_years, "initial_capital": initial_capital, "position_pct": position_pct}
    params.update(filter_params)
    return _get(f"/perplexity/filter-comparison/{strategy_name}/{symbol}", params=params, timeout=timeout)

def perplexity_profiles(strategy_name: str):
    return _get(f"/perplexity/profiles/{strategy_name}")

def perplexity_profile(strategy_name: str, symbol: str):
    return _get(f"/perplexity/profiles/{strategy_name}/{symbol}")

def perplexity_delete_profile(strategy_name: str, symbol: str):
    return _delete(f"/perplexity/profiles/{strategy_name}/{symbol}")

def perplexity_suitability():
    return _get("/perplexity/suitability")

def perplexity_save_suitability(config: dict):
    return _post("/perplexity/suitability", json=config)

# ── Day-trading market scanner ────────────────────────────────────

def daytrading_scanner_watchlist(
    max_symbols: int = 20,
    universe: str = "",
    market_state: str = "",
    min_price: float = 5.0,
    max_price: float | None = None,
    min_avg_volume: float = 1_000_000,
    min_float: float | None = None,
    max_float: float | None = None,
    universe_max_symbols: int | None = None,
    run_native_precheck: bool | None = None,
    precheck_top_k: int | None = None,
    timeout: int = 600,
):
    params: dict = {
        "max_symbols": max_symbols,
        "universe": universe,
        "market_state": market_state,
        "min_price": min_price,
        "min_avg_volume": min_avg_volume,
    }
    if max_price is not None:
        params["max_price"] = max_price
    if min_float is not None:
        params["min_float"] = min_float
    if max_float is not None:
        params["max_float"] = max_float
    if universe_max_symbols is not None:
        params["universe_max_symbols"] = universe_max_symbols
    if run_native_precheck is not None:
        # FastAPI's bool query parser accepts "true"/"false" strings
        params["run_native_precheck"] = str(run_native_precheck).lower()
    if precheck_top_k is not None:
        params["precheck_top_k"] = precheck_top_k
    return _get("/daytrading/scanner/watchlist", timeout=timeout, params=params)

def daytrading_scanner_metrics(symbol: str):
    return _get(f"/daytrading/scanner/metrics/{symbol}", timeout=60)

def autotrader_start_from_scanner(config: dict, timeout: int = 300):
    """POST /daytrading/autotrader/start-from-scanner.

    Pass any subset of: max_symbols, allowed_buckets, direction_mode, trail_mode,
    partial_tp, risk_per_trade_pct, max_daily_loss_pct, max_trades_per_day,
    max_consecutive_losses, initial_capital, broker_name, entry_mode,
    native_strategies, market_state, min_score, require_native_signal,
    prefer_native_signal, min_best_native_confidence, precheck_top_k, force.
    Backend defaults are sensible — most calls just need max_symbols.
    """
    return _post("/daytrading/autotrader/start-from-scanner", json=config, timeout=timeout)

# ── Day-trading auto-trader switch ────────────────────────────────────

def autotrader_status():
    return _get("/daytrading/autotrader/status")

def autotrader_start(config: dict):
    return _post("/daytrading/autotrader/start", json=config)

def autotrader_stop(flatten: bool = False):
    return _post(f"/daytrading/autotrader/stop?flatten={str(flatten).lower()}")

def autotrader_flatten(symbol: str | None = None):
    path = "/daytrading/autotrader/flatten"
    if symbol:
        path += f"?symbol={symbol}"
    return _post(path)

def daytrading_market_status():
    return _get("/daytrading/market-status")

def daytrading_data_source_status():
    return _get("/daytrading/data-source-status")

def autotrader_decision_summary(symbol: str | None = None, limit: int | None = None):
    params: dict = {}
    if symbol:
        params["symbol"] = symbol
    if limit:
        params["limit"] = limit
    return _get("/daytrading/autotrader/decision-summary", params=params)


def settings_get_trade_routing():
    return _get("/settings/trade-routing")


def settings_set_trade_routing(value: str):
    return _post("/settings/trade-routing", json={"trade_routing": value})


def notifications_list(limit: int = 50, unread_only: bool = False):
    return _get(f"/notifications?limit={limit}&unread_only={str(unread_only).lower()}")


def notifications_unread_count():
    return _get("/notifications/unread-count")


def notifications_mark_read(notification_id: int):
    return _post(f"/notifications/{notification_id}/read")


def notifications_mark_all_read():
    return _post("/notifications/mark-all-read")


def notifications_delete(notification_id: int):
    return _delete(f"/notifications/{notification_id}")


# ── P/L ─────────────────────────────────────────────────────────────────────

def pnl_summary(include_unrealized: bool = True):
    return _get("/pnl/summary",
                params={"include_unrealized": str(include_unrealized).lower()},
                timeout=30)


def pnl_by_symbol():
    return _get("/pnl/by-symbol", timeout=30)


def pnl_by_strategy():
    return _get("/pnl/by-strategy", timeout=30)


def pnl_closed_trades(symbol: str | None = None,
                      strategy: str | None = None,
                      limit: int = 500):
    params: dict = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    if strategy:
        params["strategy"] = strategy
    return _get("/pnl/closed-trades", params=params, timeout=30)


def pnl_equity_curve(bucket: str = "trade"):
    return _get("/pnl/equity-curve", params={"bucket": bucket}, timeout=30)


def pnl_open_positions():
    return _get("/pnl/open-positions", timeout=30)


# ── Recommendations (best historically-ranked strategy per symbol) ─────────

def recommendations_list():
    return _get("/recommendations", timeout=30)


def recommendation_get(symbol: str):
    return _get(f"/recommendations/{symbol}", timeout=30)


def recommendations_recompute(symbol: str, period: str = "5y",
                              initial_capital: float = 100_000.0):
    return _post(
        f"/recommendations/recompute/{symbol}",
        params={"period": period, "initial_capital": initial_capital},
        timeout=300,
    )


def recommendations_recompute_many(symbols: list[str], period: str = "5y",
                                   initial_capital: float = 100_000.0,
                                   timeout: int = 1800):
    return _post(
        "/recommendations/recompute",
        json={"symbols": symbols, "period": period,
              "initial_capital": initial_capital},
        timeout=timeout,
    )


def backtest_custom_consensus(symbol: str, min_agreement: int = 2,
                              period: str = "1y", initial_capital: float = 100_000.0,
                              timeout: int = 120):
    """Consensus backtest on an arbitrary symbol using the 5 scanner strategies."""
    return _get(
        f"/backtest/custom-consensus/{symbol}",
        params={"min_agreement": min_agreement, "period": period,
                "initial_capital": initial_capital},
        timeout=timeout,
    )


def backtest_custom_compare_all(symbol: str, period: str = "1y",
                                initial_capital: float = 100_000.0,
                                timeout: int = 240):
    """Compare-all backtest: each of the 5 scanner strategies run independently."""
    return _get(
        f"/backtest/custom-compare-all/{symbol}",
        params={"period": period, "initial_capital": initial_capital},
        timeout=timeout,
    )


def backtest_run_generic(symbol: str, strategy_type: str,
                         period: str = "1y", initial_capital: float = 100_000.0,
                         disable_trail: bool = False,
                         disable_exit_policy: bool = False,
                         stop_loss_pct: float = 8.0,
                         exit_rsi: float = 0.0,
                         approach_c: bool = False,
                         tight_trail_pct: float = 2.0,
                         timeout: int = 120):
    """Single-strategy backtest on an arbitrary symbol using factory defaults."""
    params = {"period": period, "initial_capital": initial_capital,
              "disable_trail": disable_trail,
              "disable_exit_policy": disable_exit_policy,
              "stop_loss_pct": stop_loss_pct}
    if exit_rsi > 0:
        params["exit_rsi"] = exit_rsi
    if approach_c:
        params["approach_c"] = "true"
        params["tight_trail_pct"] = tight_trail_pct
    return _get(f"/backtest/run-generic/{symbol}/{strategy_type}", params=params, timeout=timeout)


def backtest_live_signals(symbol: str, period: str = "1y", timeout: int = 60):
    """Run all 7 backtest strategies on the latest bar — read-only signal preview."""
    return _get(
        f"/backtest/live-signals/{symbol}",
        params={"period": period},
        timeout=timeout,
    )


def backtest_walkforward(symbol: str, strategy_type: str, mode: str = "simple",
                         period: str = "5y", train_pct: float = 0.70,
                         train_years: float = 3.0, test_years: float = 1.0,
                         step_years: float = 1.0, initial_capital: float = 100_000.0,
                         timeout: int = 300):
    """Walk-forward OOS validation for one v2 strategy (simple split or rolling)."""
    return _get(
        f"/backtest/walkforward/{symbol}/{strategy_type}",
        params={"mode": mode, "period": period, "train_pct": train_pct,
                "train_years": train_years, "test_years": test_years,
                "step_years": step_years, "initial_capital": initial_capital},
        timeout=timeout,
    )


def scanner_calibrate(symbol: str, strategy_type: str, period: str = "5y",
                      initial_capital: float = 10_000.0, timeout: int = 600):
    """Optimize-Filters-and-save for a scanner strategy (tightens per-symbol params)."""
    return _get(
        f"/backtest/scanner-calibrate/{symbol}/{strategy_type}",
        params={"period": period, "initial_capital": initial_capital},
        timeout=timeout,
    )


def scanner_profile_get(symbol: str, strategy_type: str, timeout: int = 15):
    """Return the saved calibration profile for a symbol+strategy, or {}."""
    return _get(f"/backtest/scanner-profiles/{symbol}/{strategy_type}", timeout=timeout)


def scanner_profile_delete(symbol: str, strategy_type: str, timeout: int = 15):
    """Delete a saved scanner calibration profile (reverts to factory defaults)."""
    return _delete(f"/backtest/scanner-profiles/{symbol}/{strategy_type}", timeout=timeout)
