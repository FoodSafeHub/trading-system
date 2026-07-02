"""Runtime-mutable settings endpoints.

Currently exposes the trade_routing toggle so the UI can switch between
Webull / Schwab / Both without restarting the server. The change is
persisted to .env so it survives a restart.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])

_ENV_PATH = Path(".env")


class TradeRoutingResponse(BaseModel):
    trade_routing: Literal["auto", "paper", "schwab", "webull", "zerodha", "both", "cash_aware"]
    active_broker: Literal["paper", "schwab", "webull", "zerodha"]
    effective_brokers: list[str]


class TradeRoutingUpdate(BaseModel):
    trade_routing: Literal["auto", "paper", "schwab", "webull", "zerodha", "both", "cash_aware"]


def _effective_brokers(trade_routing: str, active_broker: str) -> list[str]:
    if trade_routing == "auto":
        return [active_broker]
    # cash_aware picks ONE US broker per BUY by cash, but both are eligible.
    if trade_routing in ("both", "cash_aware"):
        return ["schwab", "webull"]
    return [trade_routing]


def _upsert_env(key: str, value: str) -> None:
    """Set KEY=value in .env, replacing the existing line or appending."""
    line = f"{key}={value}"
    if not _ENV_PATH.exists():
        _ENV_PATH.write_text(line + "\n", encoding="utf-8")
        return
    text = _ENV_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(line, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += line + "\n"
    _ENV_PATH.write_text(text, encoding="utf-8")


@router.get("/trade-routing", response_model=TradeRoutingResponse)
def get_trade_routing() -> TradeRoutingResponse:
    s = get_settings()
    return TradeRoutingResponse(
        trade_routing=s.trade_routing,
        active_broker=s.active_broker,
        effective_brokers=_effective_brokers(s.trade_routing, s.active_broker),
    )


@router.post("/trade-routing", response_model=TradeRoutingResponse)
def set_trade_routing(payload: TradeRoutingUpdate) -> TradeRoutingResponse:
    s = get_settings()
    new_value = payload.trade_routing

    if new_value == "webull":
        logger.info(
            "[settings] trade_routing -> webull: ALL default-route orders now go to "
            "Webull only. Ensure the Webull session is valid (Webull page) or orders "
            "will be rejected at submit."
        )
    if new_value == "both":
        logger.warning(
            "[settings] trade_routing -> both: every order fans out to Schwab AND "
            "Webull (duplicate fills by design). Use cash_aware to pick ONE broker "
            "per BUY instead."
        )
    if new_value == "zerodha":
        logger.warning(
            "[settings] trade_routing -> zerodha: ALL orders now route to the India "
            "(Zerodha) broker. US symbols won't trade. Prefer per-assignment broker "
            "routing if you only want some symbols on Zerodha."
        )
    if new_value == "cash_aware":
        logger.info(
            "[settings] trade_routing -> cash_aware: each default-broker US BUY routes "
            "to whichever US broker (Schwab/Webull) has the most fundable cash. SELLs "
            "route to the broker holding the position; reads see both."
        )

    _upsert_env("TRADE_ROUTING", new_value)
    object.__setattr__(s, "trade_routing", new_value)

    logger.info("[settings] trade_routing set to %s", new_value)
    return TradeRoutingResponse(
        trade_routing=s.trade_routing,
        active_broker=s.active_broker,
        effective_brokers=_effective_brokers(s.trade_routing, s.active_broker),
    )
