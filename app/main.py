from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import account, analyst_ratings, assignments, backtest, chart, daytrading, health, m1, notifications, orders, perplexity, pnl, recommendations, risk, rs_rotation, scanner, settings as settings_routes, signals, strategy, schwab_auth, webull_auth, zerodha_auth, upstox_auth
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
    logger.info("API: https://%s:8001", settings.api_host)
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


# Paths that bypass bearer auth. /health is the start.bat readiness probe.
# Schwab's OAuth redirect comes from Schwab's servers and can't carry our header.
# /docs and friends are static / OpenAPI scaffolding.
_AUTH_EXEMPT_PATHS = {
    "/health",
    "/schwab/auth",
    "/schwab/callback",
    "/zerodha/login",
    "/zerodha/callback",
    "/upstox/login",
    "/upstox/callback",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


@app.middleware("http")
async def bearer_token_auth(request: Request, call_next):
    """Require Authorization: Bearer <token> on every API call.

    Why: before this, any process on localhost (browser tabs, other tools)
    could hit the trading API. With LIVE_TRADING_ENABLED=true that's an
    unauthenticated kill-switch and order-entry surface.
    Empty api_bearer_token keeps the check off for back-compat — useful when
    a fresh clone hasn't generated a token yet.
    """
    if request.method == "OPTIONS":  # let CORS preflight through
        return await call_next(request)

    token = settings.api_bearer_token
    if not token:
        return await call_next(request)

    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS or path.startswith("/static/"):
        return await call_next(request)

    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return JSONResponse({"detail": "missing bearer token"}, status_code=401)
    if header[len("Bearer "):] != token:
        return JSONResponse({"detail": "invalid bearer token"}, status_code=401)

    return await call_next(request)

# ── Routes ───────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(account.router)
app.include_router(backtest.router)
app.include_router(orders.router)
app.include_router(signals.router)
app.include_router(risk.router)
app.include_router(strategy.router)
app.include_router(schwab_auth.router)
app.include_router(webull_auth.router)
app.include_router(zerodha_auth.router)
app.include_router(upstox_auth.router)
app.include_router(perplexity.router)
app.include_router(assignments.router)
app.include_router(scanner.router)
app.include_router(daytrading.router, prefix="/daytrading")
app.include_router(chart.router)
app.include_router(settings_routes.router)
app.include_router(notifications.router)
app.include_router(pnl.router)
app.include_router(recommendations.router)
app.include_router(analyst_ratings.router)
app.include_router(rs_rotation.router)
app.include_router(m1.router)

# ── Static files ─────────────────────────────────────────────
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except Exception:
    pass  # static dir optional
