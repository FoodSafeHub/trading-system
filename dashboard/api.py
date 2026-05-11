from __future__ import annotations

import requests

BASE = "http://127.0.0.1:8000"


def _get(path: str, **kwargs):
    r = requests.get(f"{BASE}{path}", timeout=10, **kwargs)
    r.raise_for_status()
    return r.json()


def _post(path: str, **kwargs):
    r = requests.post(f"{BASE}{path}", timeout=10, **kwargs)
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
    r = requests.post(f"{BASE}/strategy/scheduler/config", params=params, timeout=10)
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
    r = requests.patch(f"{BASE}/assignments/{symbol}/toggle", params={"enabled": str(enabled).lower()}, timeout=10)
    r.raise_for_status()
    return r.json()

def set_assignment_cap(symbol: str, max_capital_usd: float | None):
    r = requests.patch(f"{BASE}/assignments/{symbol}/cap",
                       params={"max_capital_usd": max_capital_usd if max_capital_usd else ""},
                       timeout=10)
    r.raise_for_status()
    return r.json()

def delete_assignment(symbol: str):
    r = requests.delete(f"{BASE}/assignments/{symbol}", timeout=10)
    r.raise_for_status()
    return r.json()

def get_atr_stops(symbol: str):
    return _get(f"/perplexity/atr/{symbol}")

def place_order(symbol: str, side: str, order_type: str, quantity: float, limit_price: float | None = None):
    payload = {"symbol": symbol, "side": side, "order_type": order_type, "quantity": quantity}
    if limit_price:
        payload["limit_price"] = limit_price
    return _post("/orders/manual", json=payload)
