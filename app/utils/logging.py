from __future__ import annotations

import io
import logging
import re
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.logging import RichHandler

_ET = ZoneInfo("America/New_York")


class _ETFormatter(logging.Formatter):
    """Renders log timestamps in America/New_York (ET) regardless of host TZ."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=_ET)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="milliseconds")


def _utf8_stream(stream):
    """Wrap a text stream so Unicode chars (— → ✓ etc.) don't crash on cp1252 Windows consoles."""
    try:
        buf = getattr(stream, "buffer", None)
        if buf is not None:
            return io.TextIOWrapper(buf, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    return stream

_MASK_PATTERNS = [
    re.compile(r"(Bearer\s)\S+", re.IGNORECASE),
    re.compile(r"(access_token[\"']?\s*[:=]\s*[\"']?)\S+[\"']?", re.IGNORECASE),
    re.compile(r"(refresh_token[\"']?\s*[:=]\s*[\"']?)\S+[\"']?", re.IGNORECASE),
    re.compile(r"(client_secret[\"']?\s*[:=]\s*[\"']?)\S+[\"']?", re.IGNORECASE),
    re.compile(r"\b(\d{4})\d{4,}\b"),  # account numbers: keep first 4, mask rest
]


class MaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _mask(str(record.msg))
        record.args = tuple(_mask(str(a)) for a in record.args) if record.args else record.args
        return True


def _mask(text: str) -> str:
    text = _MASK_PATTERNS[0].sub(r"\1***MASKED***", text)
    text = _MASK_PATTERNS[1].sub(r"\1***MASKED***", text)
    text = _MASK_PATTERNS[2].sub(r"\1***MASKED***", text)
    text = _MASK_PATTERNS[3].sub(r"\1***MASKED***", text)
    text = _MASK_PATTERNS[4].sub(r"\g<1>****", text)
    return text


def configure_logging(log_level: str = "INFO", log_dir: Path = Path("logs")) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(log_level)

    masking_filter = MaskingFilter()

    # Rich console handler — force a UTF-8-safe stream so Unicode log lines
    # (em-dashes, arrows, check marks) don't crash on Windows cp1252 consoles.
    rich_console = Console(
        file=_utf8_stream(sys.stdout),
        force_terminal=True,
        legacy_windows=False,
    )
    console = RichHandler(
        console=rich_console,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        log_time_format=lambda dt: dt.astimezone(_ET).strftime("[%H:%M:%S ET]"),
    )
    console.setLevel(log_level)
    console.addFilter(masking_filter)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_dir / "trading.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_fmt = _ETFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S %Z",
    )
    file_handler.setFormatter(file_fmt)
    file_handler.addFilter(masking_filter)

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
