from __future__ import annotations

"""
Daily M1 portfolio scan — the scheduled job behind the SIP advisor.

Runs the pie analyzer once (dip-weighted), then emits a notification digest:
  - which pies/names are in the buy-the-dip zone today,
  - any laggards worth reviewing (add-only — never auto-sells),
  - a one-line funding suggestion for the configured contribution.

Advisory only. Writes to the notifications table via notify_m1 (ungated) and
best-effort Windows toast. Registered in the strategy scheduler to fire
pre-open and post-close.
"""

import logging
import os

from app.services.m1.analyzer import analyze_pies
from app.services.notifications.bus import notify_m1

logger = logging.getLogger(__name__)

# Default per-scan contribution used to size the digest's $ suggestion.
# Override with env M1_SIP_CONTRIBUTION.
_DEFAULT_CONTRIBUTION = float(os.getenv("M1_SIP_CONTRIBUTION", "500") or 500)

# Only alert on dip-buy names at/above this dip_score.
_DIP_ALERT_THRESHOLD = float(os.getenv("M1_DIP_THRESHOLD", "60") or 60)


def run_daily_m1_scan(session_label: str = "scan", contribution: float | None = None) -> dict:
    """Run the dip-weighted pie analysis and emit a digest notification.

    session_label: "pre-open" | "post-close" | "scan" — shown in the alert.
    Returns a small summary dict (also useful for the API/manual trigger).
    """
    contribution = _DEFAULT_CONTRIBUTION if contribution is None else contribution

    try:
        res = analyze_pies(
            contribution=contribution, tilt_mode="dip", pie_split_mode="conviction"
        )
    except FileNotFoundError as exc:
        logger.warning("[m1-daily] pies.json missing: %s", exc)
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("[m1-daily] scan failed")
        return {"ok": False, "error": str(exc)}

    # Collect dip-buy opportunities and laggards across all pies (dedup by symbol).
    dip_names: dict[str, dict] = {}
    laggards: dict[str, dict] = {}
    for pie in res.pies:
        for s in pie.slices:
            if s.error is not None:
                continue
            # Pull the richer per-symbol metrics off the slice's signal — the
            # slice carries direction/conviction; dip metrics live on the unique
            # holding signal, which analyze_pies already computed. We re-read the
            # dip-relevant flags from the slice's note instead of re-fetching.
            if getattr(s, "dip_score", 0) >= _DIP_ALERT_THRESHOLD:
                dip_names.setdefault(s.symbol, {"symbol": s.symbol, "dip_score": s.dip_score})
            if getattr(s, "laggard", False):
                laggards.setdefault(s.symbol, {"symbol": s.symbol})

    top_pie = res.pies[0] if res.pies else None
    funded = [
        (pie.name, s.symbol, s.suggested_dollars)
        for pie in res.pies for s in pie.slices if s.suggested_dollars > 0
    ]
    funded.sort(key=lambda t: t[2], reverse=True)

    # Build digest body.
    lines = []
    if funded:
        top = funded[:5]
        lines.append("Top buys: " + ", ".join(f"{sym} ${amt:,.0f}" for _, sym, amt in top))
    else:
        lines.append("No buy-the-dip names today — consider holding the contribution.")
    if dip_names:
        lines.append("Dip zone: " + ", ".join(sorted(dip_names)))
    if laggards:
        lines.append("Review (laggards): " + ", ".join(sorted(laggards)))
    if top_pie:
        lines.append(f"Top pie: {top_pie.name} (${top_pie.suggested_dollars:,.0f})")

    title = f"M1 {session_label}: {len(funded)} buys, {len(dip_names)} dips"
    body = " · ".join(lines)

    notify_m1(title=title, body=body, toast=True)
    logger.info("[m1-daily] %s — %s", session_label, body)

    return {
        "ok": True,
        "session": session_label,
        "contribution": contribution,
        "funded_count": len(funded),
        "dip_count": len(dip_names),
        "laggard_count": len(laggards),
        "top_funded": funded[:10],
        "dip_names": sorted(dip_names),
        "laggards": sorted(laggards),
    }
