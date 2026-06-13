from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.services.m1.analyzer import analyze_portfolio, analyze_pies
from app.services.m1.targets import compute_targets

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/m1", tags=["m1"])


@router.get("/analyze")
def analyze(
    contribution: float = Query(0.0, ge=0, description="New money to allocate ($)."),
    tilt_mode: str = Query("aggressive", pattern="^(aggressive|moderate|gentle|dip)$"),
    period: str = Query("1y", description="History window for the strategy panel."),
):
    """Analyze the M1 holdings and return per-stock signals + a funding tilt.

    Advisory only — places no orders.
    """
    try:
        result = analyze_portfolio(
            contribution=contribution, tilt_mode=tilt_mode, period=period
        )
        return result.to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("[m1] analyze failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/analyze_pies")
def analyze_pies_route(
    contribution: float = Query(0.0, ge=0, description="New money to allocate ($)."),
    tilt_mode: str = Query("aggressive", pattern="^(aggressive|moderate|gentle|dip)$"),
    pie_split_mode: str = Query("conviction", pattern="^(conviction|equal|value)$"),
    period: str = Query("1y"),
):
    """Two-level M1 plan: split the contribution across pies, then tilt within
    each pie. Advisory only — places no orders.
    """
    try:
        return analyze_pies(
            contribution=contribution, tilt_mode=tilt_mode,
            pie_split_mode=pie_split_mode, period=period,
        ).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("[m1] analyze_pies failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/targets")
def targets(
    contribution: float = Query(0.0, ge=0, description="New money to allocate ($)."),
    period: str = Query("1y"),
    max_stock_weight: float = Query(0.10, gt=0, le=1.0),
    max_pie_weight: float = Query(0.35, gt=0, le=1.0),
):
    """Recommended TARGET weights per pie and per stock (signal x momentum x
    inverse-vol, capped), with current-vs-target drift and gap-closing funding.
    Advisory only.
    """
    try:
        return compute_targets(
            contribution=contribution, period=period,
            max_stock_weight=max_stock_weight, max_pie_weight=max_pie_weight,
        ).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("[m1] targets failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/scan_now")
def scan_now(
    session_label: str = Query("manual"),
    contribution: float = Query(None, ge=0),
):
    """Run the daily M1 SIP scan immediately and emit the notification digest.

    Same job the scheduler fires pre-open/post-close. Advisory only.
    """
    from app.services.m1.daily_scan import run_daily_m1_scan
    try:
        return run_daily_m1_scan(session_label=session_label, contribution=contribution)
    except Exception as exc:
        logger.exception("[m1] scan_now failed")
        raise HTTPException(status_code=500, detail=str(exc))
