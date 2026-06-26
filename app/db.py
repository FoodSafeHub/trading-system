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
        realized_trades,
        scan_results,
        settings as settings_model,
        signals,
        strategy_positions,
        strategy_recommendations,
        strategy_runs,
        trail_peaks,
    )
    # Rebuild the assignments PK to the composite (symbol, system, strategy_name)
    # BEFORE create_all, so create_all doesn't try (and fail) to reconcile the
    # old single-column-PK table against the new model. The rebuild is a no-op
    # once the composite PK is in place.
    _migrate_assignments_composite_pk()
    Base.metadata.create_all(bind=engine)
    _migrate_add_orders_source_column()
    _migrate_add_assignments_max_shares_column()
    _migrate_add_assignments_broker_column()
    _migrate_add_orders_trail_columns()
    _migrate_add_assignments_tight_trail_pct()


def _migrate_assignments_composite_pk() -> None:
    """Idempotent rebuild of symbol_strategy_assignments to a composite PK.

    Originally the table's primary key was `symbol` alone, so a symbol could have
    only one assignment. We now allow several strategies per symbol, keyed by
    (symbol, system, strategy_name). SQLite can't ALTER a primary key, so when the
    old single-column PK is detected we rebuild: rename old -> recreate with the
    composite PK -> copy rows -> drop old. No-op once the composite PK exists, on a
    fresh DB (table absent — create_all will make it correctly), or on non-SQLite.
    """
    with engine.connect() as conn:
        try:
            info = conn.exec_driver_sql(
                "PRAGMA table_info(symbol_strategy_assignments)"
            ).fetchall()
        except Exception:
            return  # Non-SQLite or no driver support — skip.
        if not info:
            return  # Table doesn't exist yet — create_all builds it fresh.
        # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk).
        # pk > 0 marks a primary-key column; count how many participate.
        pk_cols = [r[1] for r in info if r[5]]
        if len(pk_cols) >= 2:
            return  # Already composite — nothing to do.

        existing = {r[1] for r in info}
        # Copy only columns present in BOTH the old table and the new schema, so a
        # DB missing a later-added column (broker/tight_trail_pct/max_shares) still
        # migrates; the column migrations below backfill the rest.
        candidate_cols = [
            "symbol", "system", "strategy_name", "enabled", "max_capital_usd",
            "max_shares", "broker", "notes", "tight_trail_pct", "assigned_at",
        ]
        copy_cols = [c for c in candidate_cols if c in existing]
        col_list = ", ".join(copy_cols)
        try:
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            conn.exec_driver_sql(
                "ALTER TABLE symbol_strategy_assignments "
                "RENAME TO symbol_strategy_assignments_old"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE symbol_strategy_assignments (
                    symbol VARCHAR(16) NOT NULL,
                    system VARCHAR(32) NOT NULL,
                    strategy_name VARCHAR(128) NOT NULL,
                    enabled BOOLEAN,
                    max_capital_usd FLOAT,
                    max_shares FLOAT,
                    broker VARCHAR(32) NOT NULL DEFAULT 'default',
                    notes VARCHAR(256),
                    tight_trail_pct FLOAT,
                    assigned_at DATETIME,
                    PRIMARY KEY (symbol, system, strategy_name)
                )
                """
            )
            conn.exec_driver_sql(
                f"INSERT INTO symbol_strategy_assignments ({col_list}) "
                f"SELECT {col_list} FROM symbol_strategy_assignments_old"
            )
            conn.exec_driver_sql("DROP TABLE symbol_strategy_assignments_old")
            conn.commit()
        except Exception:
            # If another worker raced the rebuild, the composite PK will exist;
            # try to leave the DB consistent by dropping the temp table if present.
            try:
                conn.exec_driver_sql(
                    "DROP TABLE IF EXISTS symbol_strategy_assignments_old"
                )
                conn.commit()
            except Exception:
                pass


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


def _migrate_add_assignments_max_shares_column() -> None:
    """Idempotent ALTER TABLE to add symbol_strategy_assignments.max_shares."""
    with engine.connect() as conn:
        try:
            rows = conn.exec_driver_sql(
                "PRAGMA table_info(symbol_strategy_assignments)"
            ).fetchall()
        except Exception:
            return
        cols = {r[1] for r in rows}
        if "max_shares" in cols:
            return
        try:
            conn.exec_driver_sql(
                "ALTER TABLE symbol_strategy_assignments ADD COLUMN max_shares REAL"
            )
            conn.commit()
        except Exception:
            pass


def _migrate_add_orders_trail_columns() -> None:
    """Idempotent ALTER TABLE to add trail_type and trail_value to orders.

    Needed for TRAILING_STOP orders placed after the volatility-aware trailing
    stop feature was introduced. Existing rows keep NULL values (non-trail orders).
    """
    with engine.connect() as conn:
        try:
            rows = conn.exec_driver_sql("PRAGMA table_info(orders)").fetchall()
        except Exception:
            return
        cols = {r[1] for r in rows}
        for col, typedef in [("trail_type", "TEXT"), ("trail_value", "REAL")]:
            if col not in cols:
                try:
                    conn.exec_driver_sql(f"ALTER TABLE orders ADD COLUMN {col} {typedef}")
                    conn.commit()
                except Exception:
                    pass


def _migrate_add_assignments_tight_trail_pct() -> None:
    """Idempotent ALTER TABLE to add symbol_strategy_assignments.tight_trail_pct.

    Stores the per-assignment Approach C tight trail % selected during backtesting.
    NULL means use the system default (2.0%). Existing assignments get NULL so
    they continue using the default without any change.
    """
    with engine.connect() as conn:
        try:
            rows = conn.exec_driver_sql(
                "PRAGMA table_info(symbol_strategy_assignments)"
            ).fetchall()
        except Exception:
            return
        cols = {r[1] for r in rows}
        if "tight_trail_pct" not in cols:
            try:
                conn.exec_driver_sql(
                    "ALTER TABLE symbol_strategy_assignments ADD COLUMN tight_trail_pct REAL"
                )
                conn.commit()
            except Exception:
                pass


def _migrate_add_assignments_broker_column() -> None:
    """Idempotent ALTER TABLE to add symbol_strategy_assignments.broker.

    Default is 'default' so existing rows keep using the global active_broker /
    trade_routing toggle until the user assigns a specific broker.
    """
    with engine.connect() as conn:
        try:
            rows = conn.exec_driver_sql(
                "PRAGMA table_info(symbol_strategy_assignments)"
            ).fetchall()
        except Exception:
            return
        cols = {r[1] for r in rows}
        if "broker" in cols:
            return
        try:
            conn.exec_driver_sql(
                "ALTER TABLE symbol_strategy_assignments "
                "ADD COLUMN broker TEXT NOT NULL DEFAULT 'default'"
            )
            conn.commit()
        except Exception:
            pass
