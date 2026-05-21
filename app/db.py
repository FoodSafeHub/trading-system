from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _get_engine():
    settings = get_settings()
    db_url = settings.database_url

    # Ensure data directory exists for SQLite
    if db_url.startswith("sqlite"):
        db_path = db_url.replace("sqlite:///", "").lstrip("./")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        echo=False,
    )

    # Enable WAL mode for SQLite for better concurrent read performance
    if "sqlite" in db_url:
        @event.listens_for(engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


engine = _get_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency — yields a database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called once at startup."""
    from app.models import (  # noqa: F401 — import models so metadata is populated
        assignments,
        audit,
        broker_tokens,
        error_logs,
        executions,
        notifications,
        orders,
        positions,
        scan_results,
        settings as settings_model,
        signals,
        strategy_runs,
    )
    Base.metadata.create_all(bind=engine)
    _migrate_add_orders_source_column()


def _migrate_add_orders_source_column() -> None:
    """One-shot, idempotent ALTER TABLE to add orders.source on existing DBs.

    We don't use Alembic; SQLAlchemy's create_all only creates missing tables,
    not missing columns. This runs at every startup but is a no-op once the
    column exists.
    """
    with engine.connect() as conn:
        # Detect column presence — works on SQLite.
        try:
            rows = conn.exec_driver_sql("PRAGMA table_info(orders)").fetchall()
        except Exception:
            return  # Non-SQLite engine — skip the migration.
        cols = {r[1] for r in rows}
        if "source" in cols:
            return
        try:
            conn.exec_driver_sql(
                "ALTER TABLE orders ADD COLUMN source TEXT DEFAULT 'manual'"
            )
            conn.exec_driver_sql(
                "UPDATE orders SET source = 'unknown_pre_migration' "
                "WHERE source IS NULL OR source = 'manual'"
            )
            conn.commit()
        except Exception:
            # If the ALTER raced with another worker, the column will exist; safe to ignore.
            pass
