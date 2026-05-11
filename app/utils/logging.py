from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

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

    # Rich console handler
    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
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
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    file_handler.addFilter(masking_filter)

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
