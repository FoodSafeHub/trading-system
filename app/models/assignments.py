from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SymbolStrategyAssignment(Base):
    """
    Stores which strategy system + strategy name to use for auto-trading a symbol.
    Each symbol has at most one active assignment.
    """
    __tablename__ = "symbol_strategy_assignments"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    system: Mapped[str] = mapped_column(String(32), nullable=False)   # "bollinger" | "perplexity"
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Max dollars to allocate to this symbol. None = use global account settings.
    max_capital_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Max shares cap. Used ONLY when max_capital_usd is empty — dollar cap wins
    # whenever both are set. None = no shares-based cap.
    max_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Per-assignment broker override. "default" means follow the global
    # active_broker / trade_routing toggle; otherwise route this symbol's
    # orders to a specific broker (e.g. "schwab", "webull", "paper").
    broker: Mapped[str] = mapped_column(String(32), nullable=False, default="default")
    notes: Mapped[str] = mapped_column(String(256), nullable=True)
    # Per-assignment tight trailing stop % for Approach C.
    # When a strategy SELL signal fires, the scheduler places a trailing stop
    # at this % distance from the signal price instead of a market sell.
    # None means use the system default (2.0%). Range 1.0–10.0.
    tight_trail_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
