from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.schemas.scanner import ScanConfig, ScanResultOut, ScanSummary
from app.services.scanner.scanner_service import (
    get_last_persisted_scan,
    get_latest_results,
    run_scan,
)
from app.utils.time_utils import to_et

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scanner", tags=["scanner"])

# Last scan summary kept in memory for instant GET without DB query
_last_summary: ScanSummary | None = None
_scan_running: bool = False
_scan_lock = threading.Lock()  # protects the check-and-set of _scan_running


@router.post("/run", response_model=ScanSummary)
async def trigger_scan(config: ScanConfig, background_tasks: BackgroundTasks):
    """
    Start a scan immediately using the given config.
    For large universes (sp500, nasdaq100) the scan runs in the background
    and results are retrievable via GET /scanner/results.
    For small universes (watchlist, custom ≤20 symbols) it runs synchronously.
    """
    global _scan_running, _last_summary
    # Atomic check-and-set so two concurrent POSTs can't both start a scan
    with _scan_lock:
        if _scan_running:
            raise HTTPException(status_code=409, detail="A scan is already running. Check /scanner/results.")
        _scan_running = True
    started = True
    try:
        is_large = config.universe in (
            "sp500", "nasdaq100", "sp400", "sp600", "sp1500",
            "nifty50", "nifty100", "nifty200", "nifty500", "nse_all",
        ) or len(config.custom_symbols) > 20

        if is_large:
            # Hand off to background — _run_scan_bg owns the flag from here.
            background_tasks.add_task(_run_scan_bg, config)
            started = False  # background task will clear it
            return ScanSummary(
                scan_run_id="pending",
                scanned_at=datetime.now(timezone.utc),
                universe=config.universe,
                total_scanned=0,
                total_passed_filters=0,
                total_matches=0,
                top_candidates=[],
                duration_seconds=0.0,
            )
        try:
            summary = run_scan(config)
        except Exception as exc:
            logger.exception("[scanner] run_scan failed")
            raise HTTPException(status_code=500, detail="Scan failed. See server logs for details.")
        _last_summary = summary
        return summary
    finally:
        if started:
            with _scan_lock:
                _scan_running = False


def _run_scan_bg(config: ScanConfig) -> None:
    global _scan_running, _last_summary
    try:
        _last_summary = run_scan(config)
        logger.info("[scanner] Background scan complete — %d matches", _last_summary.total_matches)
    except Exception:
        logger.exception("[scanner] Background scan failed")
    finally:
        with _scan_lock:
            _scan_running = False


@router.get("/results", response_model=List[ScanResultOut])
async def get_scan_results(
    limit: int = Query(default=50, le=200),
    universe: Optional[str] = Query(default=None),
):
    """Return the most recent scan results from the database.

    `universe` narrows to one universe so manual large-universe scans stay
    findable after the auto-scheduler's frequent scans push them out of the
    unfiltered last-N window.
    """
    return get_latest_results(limit=limit, universe=universe)


@router.get("/latest", response_model=Optional[ScanSummary])
async def get_latest_scan():
    """Return the summary of the most recently completed scan (in-memory, fast)."""
    return _last_summary


@router.get("/status")
async def get_scan_status():
    """Return whether a scan is currently running + the last scan's metadata.

    `last_scan`/`last_matches` come from the DB (source of truth) so they
    reflect BACKGROUND auto-scans and survive server restarts — the in-memory
    `_last_summary` only ever captured manual scans and was lost on restart.
    Falls back to the in-memory summary only if the DB read fails.
    """
    last_scan_iso = None
    last_matches = 0
    last_universe = None
    try:
        persisted = get_last_persisted_scan()
        if persisted:
            last_scan_iso = to_et(persisted["scanned_at"]).isoformat()
            last_matches = persisted["matches"]
            last_universe = persisted["universe"]
    except Exception:
        if _last_summary:
            last_scan_iso = to_et(_last_summary.scanned_at).isoformat()
            last_matches = _last_summary.total_matches

    return {
        "running": _scan_running,
        "last_scan": last_scan_iso,
        "last_universe": last_universe,
        "last_run_id": _last_summary.scan_run_id if _last_summary else None,
        "last_matches": last_matches,
    }


@router.get("/config/defaults", response_model=ScanConfig)
async def get_default_config():
    """Return the default scan configuration."""
    return ScanConfig()


@router.get("/strategies")
async def list_scan_strategies():
    """Selectable strategies for a signal-mode scan.

    Returns the picker identifiers + friendly labels. `generic` are the 5
    regime-aware strategy TYPES the live scanner runs on every symbol; `perplexity`
    are the perplexity strategies (identified by the prefixed label the scan stores
    in its votes). The dashboard builds the strategy multiselect from this so the
    list can never drift from the engine.
    """
    from app.services.scanner.scanner_service import _make_generic_configs
    from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES

    def _friendly(text: str) -> str:
        return text.replace("_", " ")

    generic = [
        {"id": cfg.type, "label": _friendly(
            cfg.name[len("AAPL_"):] if cfg.name.startswith("AAPL_") else cfg.name
        )}
        for cfg in _make_generic_configs("AAPL")
    ]
    perplexity = [
        {"id": f"perplexity:{s.name}", "label": _friendly(s.name)}
        for s in PERPLEXITY_STRATEGIES
    ]
    return {"generic": generic, "perplexity": perplexity}
