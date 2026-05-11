from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class BrokerToken(Base):
    __tablename__ = "broker_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    broker: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    access_token: Mapped[str] = mapped_column(String(2048), nullable=False)
    refresh_token: Mapped[str] = mapped_column(String(2048), nullable=True)
    token_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scope: Mapped[str | None] = mapped_column(String(512), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<BrokerToken broker={self.broker} expiry={self.token_expiry}>"
