from __future__ import annotations

import os
import requests

BASE = os.getenv("TRADING_API_BASE", "https://127.0.0.1:8001")


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
    r = requests.get(f"{BASE}{path}", timeout=timeout, verify=False, **kwargs)
    _raise_with_body(r)
    return r.json()


def _post(path: str, **kwargs):
    r = requests.post(f"{BASE}{path}", timeout=10, verify=False, **kwargs)
    _raise_with_body(r)
    return r.json()


def _delete(path: str, timeout: int = 10):
    r = requests.delete(f"{BASE}{path}", timeout=timeout, verify=False)
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

def account_summary():
    return _get("/account/summary")

def positions():
    return _get("/account/positions")

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

def scheduler_status():
    return _get("/strategy/scheduler")

def update_scheduler_config(run_bollinger: bool | None = None, run_perplexity: bool | None = None):
    params = {}
    if run_bollinger is not None:
        params["run_bollinger"] = str(run_bollinger).lower()
    if run_perplexity is not None:
        params["run_perplexity"] = str(run_perplexity).lower()
    r = requests.post(f"{BASE}/strategy/scheduler/config", params=params, timeout=10, verify=False)
    r.raise_for_status()
    return r.json()

def chart_data(symbol: str, period: str = "3mo"):
    return _get(f"/strategy/chart/{symbol}", params={"period": period})

def list_assignments():
    return _get("/assignments")

def upsert_assignment(symbol: str, system: str, strategy_name: str, enabled: bool = True,
                      notes: str = "", max_capital_usd: float | None = None):
    return _post("/assignments", json={"symbol": symbol, "system": system,
                                       "strategy_name": strategy_name, "enabled": enabled,
                                       "notes": notes, "max_capital_usd": max_capital_usd})

def toggle_assignment(symbol: str, enabled: bool):
    r = requests.patch(f"{BASE}/assignments/{symbol}/toggle", params={"enabled": str(enabled).lower()}, timeout=10, verify=False)
    r.raise_for_status()
    return r.json()

def set_assignment_cap(symbol: str, max_capital_usd: float | None):
    r = requests.patch(f"{BASE}/assignments/{symbol}/cap",
                       params={"max_capital_usd": max_capital_usd if max_capital_usd else ""},
                       timeout=10, verify=False)
    r.raise_for_status()
    return r.json()

def delete_assignment(symbol: str):
    r = requests.delete(f"{BASE}/assignments/{symbol}", timeout=10, verify=False)
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
                      params={"enabled": str(enabled).lower()}, timeout=10, verify=False)
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
                            position_pct: float = 0.0, timeout: int = 300):
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
    timeout: int = 300,
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
    return _get("/daytrading/scanner/watchlist", timeout=timeout, params=params)

def daytrading_scanner_metrics(symbol: str):
    return _get(f"/daytrading/scanner/metrics/{symbol}", timeout=60)

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
