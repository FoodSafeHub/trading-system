from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import account, assignments, backtest, daytrading, health, orders, perplexity, risk, signals, strategy, schwab_auth
from app.config import get_settings
from app.db import init_db
from app.services.strategy.scheduler import start_scheduler, stop_scheduler
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)
    init_db()

    mode_banner = "PAPER MODE" if not settings.is_live else "⚠ LIVE MODE ⚠"
    logger.info("=" * 60)
    logger.info("Trading System starting — %s", mode_banner)
    logger.info("Active broker: %s", settings.active_broker)
    logger.info("API: http://%s:%s", settings.api_host, settings.api_port)
    logger.info("=" * 60)

    if settings.is_live:
        logger.warning(
            "LIVE TRADING IS ACTIVE. Real money is at risk. "
            "Ensure all strategies and risk limits are correctly configured."
        )

    start_scheduler()
    logger.info(f"[scheduler] Auto-cycle started (interval={settings.scheduler_interval_seconds}s)")

    yield

    stop_scheduler()
    logger.info("Trading System shut down.")


settings = get_settings()

app = FastAPI(
    title="Trading Automation System",
    description="Personal single-user trading automation — paper mode by default.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(account.router)
app.include_router(backtest.router)
app.include_router(orders.router)
app.include_router(signals.router)
app.include_router(risk.router)
app.include_router(strategy.router)
app.include_router(schwab_auth.router)
app.include_router(perplexity.router)
app.include_router(assignments.router)
app.include_router(daytrading.router, prefix="/daytrading")

# ── Static files ─────────────────────────────────────────────
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except Exception:
    pass  # static dir optional
