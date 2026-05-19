from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.schemas.scanner import ScanConfig, ScanResultOut, ScanSummary
from app.services.scanner.scanner_service import get_latest_results, run_scan
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
        is_large = config.universe in ("sp500", "nasdaq100") or len(config.custom_symbols) > 20

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
async def get_scan_results(limit: int = Query(default=50, le=200)):
    """Return the most recent scan results from the database."""
    return get_latest_results(limit=limit)


@router.get("/latest", response_model=Optional[ScanSummary])
async def get_latest_scan():
    """Return the summary of the most recently completed scan (in-memory, fast)."""
    return _last_summary


@router.get("/status")
async def get_scan_status():
    """Return whether a scan is currently running."""
    return {
        "running": _scan_running,
        "last_scan": to_et(_last_summary.scanned_at).isoformat() if _last_summary else None,
        "last_run_id": _last_summary.scan_run_id if _last_summary else None,
        "last_matches": _last_summary.total_matches if _last_summary else 0,
    }


@router.get("/config/defaults", response_model=ScanConfig)
async def get_default_config():
    """Return the default scan configuration."""
    return ScanConfig()
