from __future__ import annotations

import requests

BASE = "https://127.0.0.1:8001"


def _get(path: str, timeout: int = 10, **kwargs):
    r = requests.get(f"{BASE}{path}", timeout=timeout, verify=False, **kwargs)
    r.raise_for_status()
    return r.json()


def _post(path: str, **kwargs):
    r = requests.post(f"{BASE}{path}", timeout=10, verify=False, **kwargs)
    r.raise_for_status()
    return r.json()


def _delete(path: str, timeout: int = 10):
    r = requests.delete(f"{BASE}{path}", timeout=timeout, verify=False)
    r.raise_for_status()
    return r.json()


def health():
    return _get("/health")

def account_summary():
    return _get("/account/summary")

def positions():
    return _get("/account/positions")

def risk_status():
    return _get("/risk/status")

def set_kill_switch(active: bool):
    return _post(f"/risk/kill-switch?active={str(active).lower()}")

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
