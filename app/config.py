from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    database_url: str = "sqlite:///./data/trading.db"

    # ── Safety ───────────────────────────────────────────────
    live_trading_enabled: bool = False
    live_trading_confirmed: bool = False

    # ── Broker ───────────────────────────────────────────────
    active_broker: Literal["paper", "schwab", "webull", "zerodha"] = "paper"
    # Multi-broker trade routing. When "auto", falls back to active_broker
    # (backwards compatible). When "schwab"/"webull"/"zerodha", routes there only.
    # When "both", fans out every order to Schwab AND Webull.
    trade_routing: Literal["auto", "paper", "schwab", "webull", "zerodha", "both"] = "auto"

    # Schwab
    schwab_client_id: str = ""
    schwab_client_secret: str = ""
    schwab_redirect_uri: str = "https://127.0.0.1:8182/schwab/callback"
    schwab_account_number: str = ""
    schwab_access_token: str = ""
    schwab_refresh_token: str = ""
    schwab_token_expiry: str = ""

    # Webull
    webull_app_key: str = ""
    webull_app_secret: str = ""
    webull_access_token: str = ""
    webull_refresh_token: str = ""
    webull_token_expiry: str = ""
    webull_account_id: str = ""

    # Zerodha (Kite Connect) — India / NSE-BSE.
    # access_token is single-day: it expires at ~07:30 IST and CANNOT be
    # refreshed (SEBI rule). Re-login daily via GET /zerodha/login.
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    # Kite registers ONE redirect URL per app in the developer console.
    zerodha_redirect_uri: str = "https://127.0.0.1:8001/zerodha/callback"
    zerodha_access_token: str = ""

    # Upstox — used as the India MARKET-DATA source (quotes + historical bars)
    # while orders execute on Zerodha. Free API (through at least Mar 2026) and
    # avoids Zerodha's paid data add-on. Standard OAuth2; token expires daily
    # (~03:30 IST), re-login via GET /upstox/login. Empty key = feature off, and
    # the system falls back to its normal data chain.
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_redirect_uri: str = "https://127.0.0.1:8001/upstox/callback"
    upstox_access_token: str = ""

    # ── Risk limits ──────────────────────────────────────────
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 200.0
    max_orders_per_day: int = 10
    order_cooldown_seconds: int = 60
    trading_start_time: str = "09:30"
    trading_end_time: str = "16:00"
    trading_timezone: str = "America/New_York"

    # India market hours (NSE/BSE regular session). Used when the order's
    # broker is Zerodha — the risk engine picks hours by broker, not globally.
    india_market_open: str = "09:15"
    india_market_close: str = "15:30"
    india_timezone: str = "Asia/Kolkata"

    # ── Buying-power preflight ───────────────────────────────
    # When enabled, BUY orders are rejected if the broker's reported
    # buying_power is less than (order_value + buying_power_min_buffer_usd).
    # SELL orders skip the check (they free capital, not consume it).
    buying_power_check_enabled: bool = True
    buying_power_min_buffer_usd: float = 0.0

    # ── Protective stops ─────────────────────────────────────
    # When enabled, a filled BUY automatically gets a resting SELL STOP
    # placed at the broker so the position is protected even if the
    # software is down. OFF by default — the engine's own exit logic and
    # the Chandelier trail already manage exits while the scheduler runs;
    # this is belt-and-suspenders for live trading and should be turned on
    # deliberately. The stop price is the BUY signal's own stop_price when
    # the scheduler supplied one, else fill_price * (1 - protective_stop_pct/100).
    auto_protective_stop_enabled: bool = False
    protective_stop_pct: float = 8.0

    # ── Phase 0 strategy-refactor scaffolding (inert until later phases) ──────
    # Declared default-OFF now so the Phase 1/2 wiring has switches ready. NOTHING
    # reads these in Phase 0 — the backtest engine takes a cost_model OBJECT (not
    # settings) and the Perplexity runner still builds its bespoke strategy list.
    # When True (a later phase): route Perplexity through the unified rules.py
    # adapter, and let the Backtest/ranking path apply the cost model respectively.
    use_unified_perplexity: bool = False
    backtest_costs_enabled: bool = False

    # ── Scheduler ────────────────────────────────────────────
    # Interval for the daily-candle strategy cycle (Bollinger + Perplexity).
    # Day-trading has its own intraday loop (autotrader/manager.py) and is
    # NOT driven by this scheduler. Daily strategies don't benefit from
    # sub-hour ticks — hourly is plenty and avoids needless yfinance calls.
    scheduler_interval_seconds: int = 3600
    scheduler_enabled: bool = True
    # Which strategy systems participate in the auto-scheduler cycle.
    # Both can be enabled at the same time.
    scheduler_run_bollinger: bool = True    # run Bollinger strategies from strategies.json
    scheduler_run_perplexity: bool = True   # run the 5 Perplexity swing strategies

    # ── Position sizing ──────────────────────────────────────
    # Risk a fixed % of account per trade, sized by stop distance.
    position_sizing_enabled: bool = True
    risk_pct_per_trade: float = 0.01        # 1% of account per trade
    max_account_risk_pct: float = 0.06      # never commit more than 6% total open risk
    account_value: float = 100_000.0        # your total account size (update this)

    # ── Market regime settings ───────────────────────────────
    regime_benchmark_symbol: str = "SPY"
    regime_sma_long: int = 200
    regime_sma_mid: int = 50
    regime_deep_bear_drawdown: float = 0.20
    regime_risk_pct_bull: float = 0.01
    regime_risk_pct_bear: float = 0.003
    regime_risk_pct_deep_bear: float = 0.002
    regime_max_positions_bull: int = 10
    regime_max_positions_bear: int = 3
    regime_max_positions_deep_bear: int = 1
    regime_max_account_risk_bull: float = 0.20
    regime_max_account_risk_bear: float = 0.08
    regime_max_account_risk_deep_bear: float = 0.04

    # ── Signal consensus ─────────────────────────────────────
    # Minimum number of strategies that must agree on the same
    # symbol + direction before an order is placed.
    # 1 = any single signal triggers an order (old behaviour)
    # 2 = at least 2 strategies must agree (recommended)
    min_signal_agreement: int = 2

    # ── Twelve Data ─────────────────────────────────────────
    twelve_data_api_key: str = ""

    # ── API ──────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Shared bearer token. Dashboard sends Authorization: Bearer <token>.
    # Empty string disables the check (back-compat for unconfigured installs).
    api_bearer_token: str = ""

    # ── Derived ──────────────────────────────────────────────
    @property
    def is_live(self) -> bool:
        return (
            self.live_trading_enabled
            and self.live_trading_confirmed
            and self.active_broker != "paper"
        )

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.trading_timezone)

    @property
    def india_tz(self) -> ZoneInfo:
        return ZoneInfo(self.india_timezone)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _live_trading_safety_check(self) -> "Settings":
        if self.live_trading_enabled and not self.live_trading_confirmed:
            print(
                "[SAFETY] LIVE_TRADING_ENABLED=true but LIVE_TRADING_CONFIRMED=false. "
                "Live trading will not execute.",
                file=sys.stderr,
            )
        if self.is_live and self.active_broker == "schwab":
            if not self.schwab_client_id or not self.schwab_client_secret:
                raise ValueError(
                    "Live trading requires SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET"
                )
        if self.is_live and self.active_broker == "webull":
            if not self.webull_app_key or not self.webull_app_secret:
                raise ValueError(
                    "Live trading requires WEBULL_APP_KEY and WEBULL_APP_SECRET"
                )
        if self.is_live and self.active_broker == "zerodha":
            if not self.zerodha_api_key or not self.zerodha_api_secret:
                raise ValueError(
                    "Live trading requires ZERODHA_API_KEY and ZERODHA_API_SECRET"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
