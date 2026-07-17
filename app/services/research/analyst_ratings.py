"""Wall-Street analyst ratings for holdings + assigned symbols (yfinance).

Free-tier data source: yf.Ticker(...).info (recommendationKey, price targets,
analyst count), .recommendations_summary (strong_buy..strong_sell counts for
the current month), and .upgrades_downgrades (recent firm actions). Each
accessor is individually try/excepted — yfinance is flaky and NSE coverage is
thin, so a symbol with no data still yields a row with a `note` instead of an
exception.

Results are cached in the analyst_ratings table (one row per symbol); the
dashboard only ever reads the cache. Refresh runs on the scheduler (every
analyst_ratings_refresh_hours) and on demand via POST /ratings/recompute.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.analyst_ratings import AnalystRating

logger = logging.getLogger(__name__)

_MAX_UPGRADE_ROWS = 15


def fetch_rating(symbol: str) -> dict:
    """Fetch the analyst view of one symbol from yfinance. Never raises."""
    import yfinance as yf

    from app.services.markets import yf_symbol

    out: dict = {"symbol": symbol.upper()}
    notes: list[str] = []
    try:
        t = yf.Ticker(yf_symbol(symbol))
    except Exception as exc:
        logger.warning("[ratings] Ticker init failed for %s: %s", symbol, exc)
        out["note"] = f"ticker init failed: {exc}"
        return out

    # ── info: consensus key, price targets, analyst count, price ────────────
    try:
        info = t.info or {}
        out["recommendation_key"] = info.get("recommendationKey")
        out["analyst_count"] = info.get("numberOfAnalystOpinions")
        out["target_mean"] = info.get("targetMeanPrice")
        out["target_high"] = info.get("targetHighPrice")
        out["target_low"] = info.get("targetLowPrice")
        out["target_median"] = info.get("targetMedianPrice")
        out["current_price"] = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
    except Exception as exc:
        logger.debug("[ratings] info fetch failed for %s: %s", symbol, exc)
        notes.append("info unavailable")

    price = out.get("current_price")
    mean = out.get("target_mean")
    if price and mean:
        try:
            out["upside_pct"] = round((float(mean) / float(price) - 1.0) * 100, 2)
        except Exception:
            pass

    # ── recommendations_summary: current-month analyst counts ───────────────
    try:
        rs = t.recommendations_summary
        if rs is not None and not rs.empty:
            cur = rs[rs["period"] == "0m"] if "period" in rs.columns else rs.iloc[:1]
            if not cur.empty:
                row = cur.iloc[0]
                for col, key in (
                    ("strongBuy", "strong_buy"), ("buy", "buy"), ("hold", "hold"),
                    ("sell", "sell"), ("strongSell", "strong_sell"),
                ):
                    if col in cur.columns:
                        out[key] = int(row[col])
    except Exception as exc:
        logger.debug("[ratings] recommendations_summary failed for %s: %s", symbol, exc)
        notes.append("counts unavailable")

    # ── upgrades_downgrades: most recent firm actions ────────────────────────
    try:
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            recent = ud.sort_index(ascending=False).head(_MAX_UPGRADE_ROWS)
            rows = []
            for idx, r in recent.iterrows():
                rows.append({
                    "date": str(getattr(idx, "date", lambda: idx)())[:10]
                    if hasattr(idx, "date") else str(idx)[:10],
                    "firm": r.get("Firm"),
                    "action": r.get("Action"),
                    "from_grade": r.get("FromGrade"),
                    "to_grade": r.get("ToGrade"),
                })
            out["upgrades_json"] = json.dumps(rows)
    except Exception as exc:
        logger.debug("[ratings] upgrades_downgrades failed for %s: %s", symbol, exc)
        notes.append("upgrades unavailable")

    has_any = any(
        out.get(k) is not None
        for k in ("recommendation_key", "target_mean", "strong_buy", "analyst_count")
    )
    if not has_any:
        out["note"] = "no analyst coverage"
    elif notes:
        out["note"] = "; ".join(notes)
    return out


def rating_universe(db: Session) -> dict[str, dict]:
    """{symbol: {"is_holding": bool, "is_assigned": bool}} — the symbols the
    ratings page cares about: broker positions ∪ enabled assignments."""
    universe: dict[str, dict] = {}

    from app.models.assignments import SymbolStrategyAssignment
    try:
        for a in db.query(SymbolStrategyAssignment).filter_by(enabled=True).all():
            sym = (a.symbol or "").upper()
            if sym:
                universe.setdefault(sym, {"is_holding": False, "is_assigned": False})
                universe[sym]["is_assigned"] = True
    except Exception as exc:
        logger.warning("[ratings] assignment universe failed: %s", exc)

    try:
        import asyncio as _aio

        from app.services.brokers.factory import get_position_brokers
        loop = _aio.new_event_loop()
        try:
            for b in get_position_brokers():
                try:
                    loop.run_until_complete(b.authenticate())
                    accts = loop.run_until_complete(b.get_accounts())
                    acct = accts[0].account_id if accts else ""
                    for p in loop.run_until_complete(b.get_positions(acct)):
                        if not p.quantity:
                            continue
                        sym = p.symbol.upper()
                        universe.setdefault(sym, {"is_holding": False, "is_assigned": False})
                        universe[sym]["is_holding"] = True
                except Exception as exc:
                    logger.warning("[ratings] %s positions fetch failed: %s",
                                   getattr(b, "name", "?"), exc)
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("[ratings] holdings universe failed: %s", exc)

    return universe


def upsert_rating(db: Session, data: dict, *, is_holding: bool, is_assigned: bool) -> AnalystRating:
    sym = data["symbol"]
    row = db.query(AnalystRating).filter_by(symbol=sym).first()
    if row is None:
        row = AnalystRating(symbol=sym)
        db.add(row)
    for field in (
        "current_price", "recommendation_key", "strong_buy", "buy", "hold",
        "sell", "strong_sell", "analyst_count", "target_mean", "target_high",
        "target_low", "target_median", "upside_pct", "upgrades_json", "note",
    ):
        setattr(row, field, data.get(field))
    row.is_holding = is_holding
    row.is_assigned = is_assigned
    row.computed_at = datetime.utcnow()
    return row


def refresh_all(db: Session, symbols: list[str] | None = None) -> dict:
    """Refresh the ratings cache for the given symbols (default: full universe).

    Per-symbol failures are recorded, never raised — one bad ticker must not
    kill the batch (or the scheduler job that calls this).
    """
    universe = rating_universe(db)
    if symbols:
        want = {s.upper().strip() for s in symbols if s and s.strip()}
        targets = {
            s: universe.get(s, {"is_holding": False, "is_assigned": False})
            for s in want
        }
    else:
        targets = universe

    refreshed = 0
    errors: list[str] = []
    for sym, flags in sorted(targets.items()):
        try:
            data = fetch_rating(sym)
            upsert_rating(
                db, data,
                is_holding=flags["is_holding"],
                is_assigned=flags["is_assigned"],
            )
            db.commit()
            refreshed += 1
        except Exception as exc:
            db.rollback()
            logger.warning("[ratings] refresh failed for %s: %s", sym, exc)
            errors.append(f"{sym}: {exc}")
    return {"refreshed": refreshed, "errors": errors}
