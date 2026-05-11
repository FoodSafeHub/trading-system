# Trading Automation System

> **⚠ WARNING: This software can lose real money. Always test thoroughly in paper mode before enabling live trading.**

A local-first, single-user trading automation system with FastAPI, SQLite, and support for Charles Schwab and Webull (via official broker APIs only). Paper mode is the default. Live trading requires explicit opt-in with multiple safety flags.

---

## Features

- **Paper broker** — simulates fills instantly, no credentials needed
- **Schwab adapter** — OAuth 2.0 authorization code flow, full order lifecycle
- **Webull adapter scaffold** — placeholder, ready for implementation once OpenAPI is available
- **Strategy engine** — SMA/RSI, EMA crossover, MACD, Bollinger Bands (config-driven)
- **Risk engine** — kill switch, position size limits, daily loss cap, cooldown, duplicate prevention
- **Execution pipeline** — signal → risk check → preview → submit → fill tracking
- **Scheduler** — APScheduler polling loop, market-hours aware, overlap-safe
- **REST API** — FastAPI on localhost:8000 with interactive docs at `/docs`
- **Full audit trail** — every signal, decision, order, fill, and error persisted in SQLite

---

## Safety Design

Live trading requires **all three** to be true simultaneously:

```
LIVE_TRADING_ENABLED=true      # in .env
LIVE_TRADING_CONFIRMED=true    # in .env
ACTIVE_BROKER=schwab           # (not "paper")
```

Even then, the risk engine runs pre-flight checks on every order. The kill switch can halt all trading instantly via the API.

---

## Prerequisites

- Python 3.12+
- VS Code (recommended) with Python extension
- Git

---

## Setup (VS Code)

### 1. Clone / open the project

```bash
cd C:\Users\medagams\trading-system
code .
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env
```

Edit `.env` — at minimum set `ACTIVE_BROKER=paper`. Leave all live trading flags as `false`.

### 5. Initialize the database

```bash
python scripts/init_db.py
```

### 6. Start the API server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000/docs** to explore the API interactively.

---

## First Run (Paper Mode)

```bash
# Check system health
curl http://127.0.0.1:8000/health

# Check risk status
curl http://127.0.0.1:8000/risk/status

# Manually run one strategy cycle (fetches SPY data from Yahoo Finance)
curl -X POST http://127.0.0.1:8000/strategy/run

# View latest signals
curl http://127.0.0.1:8000/signals

# View paper account
curl http://127.0.0.1:8000/account/summary

# Manually place a paper order
curl -X POST http://127.0.0.1:8000/orders/manual \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "side": "BUY", "order_type": "MARKET", "quantity": 1}'

# Activate kill switch
curl -X POST "http://127.0.0.1:8000/risk/kill-switch?active=true"
```

---

## Schwab OAuth Setup

1. Register your app at [developer.schwab.com](https://developer.schwab.com/products/trader-api--individual-)
2. Set `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`, `SCHWAB_REDIRECT_URI` in `.env`
3. Set `ACTIVE_BROKER=schwab`
4. Start the server, then visit:

```
GET http://127.0.0.1:8000/schwab/auth
```

This returns the authorization URL. Open it in your browser, authorize the app, and Schwab will redirect to your callback URL. The tokens are stored in SQLite automatically.

> **Note:** Schwab requires HTTPS for the redirect URI even in development. You may need a self-signed cert or a tool like `mkcert`. The default redirect is `https://127.0.0.1:8182/schwab/callback` — update the port if needed.

---

## Strategy Configuration

Edit `strategies.json` to configure strategies (no code changes needed):

```json
[
  {
    "name": "SPY_SMA_RSI",
    "symbol": "SPY",
    "type": "sma_rsi",
    "enabled": true,
    "params": {
      "sma_fast": 10,
      "sma_slow": 30,
      "rsi_period": 14,
      "rsi_oversold": 30,
      "rsi_overbought": 70
    }
  }
]
```

Available strategy types: `sma_rsi`, `ema_crossover`, `macd`, `bollinger`

---

## Running Tests

```bash
pytest                          # all tests
pytest tests/unit/              # unit tests only
pytest tests/integration/       # integration (paper simulation) tests
pytest --cov=app tests/         # with coverage report
```

---

## Manual Strategy Cycle (Script)

```bash
python scripts/run_cycle.py
```

---

## Project Structure

```
trading-system/
  app/
    main.py                  — FastAPI app entry point
    config.py                — Pydantic settings (loaded from .env)
    db.py                    — SQLAlchemy engine + session factory
    models/                  — SQLAlchemy ORM models
    schemas/                 — Pydantic request/response schemas
    services/
      brokers/               — BrokerBase, PaperBroker, SchwabBroker, WebullBroker
      indicators/            — SMA, EMA, RSI, MACD, Bollinger
      strategy/              — Rule engine, strategy configs, scheduler
      risk/                  — Risk engine with all pre-flight checks
      execution/             — Execution pipeline (signal → order → fill)
      market_data/           — yfinance historical price provider
      audit/                 — Audit event + error log persistence
    api/routes/              — FastAPI route handlers
    utils/                   — Logging, time utilities
  scripts/
    init_db.py               — Create all tables
    run_cycle.py             — Manually trigger one strategy cycle
  tests/
    unit/                    — Indicator, strategy, risk, payload tests
    integration/             — End-to-end paper trading simulation
  strategies.json            — Strategy configuration (edit this)
  .env.example               — Environment variable template
  requirements.txt
```

---

## Key Risk Limits (configurable in .env)

| Setting | Default | Description |
|---|---|---|
| `MAX_POSITION_SIZE_USD` | 1000 | Max single order value |
| `MAX_DAILY_LOSS_USD` | 200 | Max daily loss before halt |
| `MAX_ORDERS_PER_DAY` | 10 | Max orders per trading day |
| `ORDER_COOLDOWN_SECONDS` | 60 | Min seconds between orders per symbol |
| `TRADING_START_TIME` | 09:30 | Market open (Eastern) |
| `TRADING_END_TIME` | 16:00 | Market close (Eastern) |
| `SCHEDULER_INTERVAL_SECONDS` | 60 | How often the cycle runs |

---

## Webull Adapter

The Webull adapter (`app/services/brokers/webull.py`) is a fully scaffolded placeholder. All methods raise `NotImplementedError` with clear TODO comments. Once Webull OpenAPI credentials and documentation are available:

1. Fill in `WEBULL_APP_KEY` and `WEBULL_APP_SECRET` in `.env`
2. Implement each method in `webull.py` following the same pattern as `schwab.py`
3. Switch `ACTIVE_BROKER=webull`

---

## Live Trading Checklist

Before enabling live trading, verify each item:

- [ ] Paper mode tested for at least several days
- [ ] All strategy signals reviewed and look reasonable
- [ ] Risk limits set conservatively
- [ ] Schwab OAuth tokens are fresh and working
- [ ] `LIVE_TRADING_ENABLED=true` set in `.env`
- [ ] `LIVE_TRADING_CONFIRMED=true` set in `.env`
- [ ] Kill switch mechanism tested
- [ ] Server bound to localhost only
- [ ] `.env` not committed to version control
