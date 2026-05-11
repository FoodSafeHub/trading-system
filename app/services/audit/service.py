from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.db import SessionLocal
from app.models.audit import AuditEvent
from app.models.error_logs import ErrorLog

logger = logging.getLogger(__name__)


class AuditService:

    def log(
        self,
        event_type: str,
        description: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            with SessionLocal() as db:
                event = AuditEvent(
                    event_type=event_type,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    description=description,
                    metadata_json=json.dumps(metadata) if metadata else None,
                    occurred_at=datetime.now(tz=timezone.utc),
                )
                db.add(event)
                db.commit()
        except Exception as exc:
            logger.error("[audit] Failed to persist audit event: %s", exc)

    def log_error(
        self,
        source: str,
        error_type: str,
        message: str,
        traceback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            with SessionLocal() as db:
                err = ErrorLog(
                    source=source,
                    error_type=error_type,
                    message=message,
                    traceback=traceback,
                    context_json=json.dumps(context) if context else None,
                    occurred_at=datetime.now(tz=timezone.utc),
                )
                db.add(err)
                db.commit()
        except Exception as exc:
            logger.error("[audit] Failed to persist error log: %s", exc)
