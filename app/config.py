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
    active_broker: Literal["paper", "schwab", "webull"] = "paper"

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

    # ── Risk limits ──────────────────────────────────────────
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 200.0
    max_orders_per_day: int = 10
    order_cooldown_seconds: int = 60
    trading_start_time: str = "09:30"
    trading_end_time: str = "16:00"
    trading_timezone: str = "America/New_York"

    # ── Scheduler ────────────────────────────────────────────
    scheduler_interval_seconds: int = 60
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
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
