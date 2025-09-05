from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..core.config_loader import Settings


class JsonFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON.

    Includes: timestamp (ISO8601 UTC), level, name, message.
    """

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def _coerce_log_level(level_value: str | int) -> int:
    if isinstance(level_value, int):
        return level_value
    # Accept string names like "INFO", "debug", etc.
    name = str(level_value).upper()
    return logging._nameToLevel.get(name, logging.INFO)


def setup_logger(settings: Settings) -> logging.Logger:
    """Configure and return the UnicornWealth logger.

    - Level from settings.log_level
    - File rotation: daily at midnight, keep 30 backups
    - Console output to stdout
    - Structured JSON formatting for both handlers
    - Creates the log directory if missing
    """
    logger = logging.getLogger("UnicornWealth")

    # Set level from settings
    logger.setLevel(_coerce_log_level(settings.log_level))

    # Ensure log directory exists
    log_path = Path(settings.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        # Clear existing handlers to re-apply current configuration
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    # Handlers
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_path), when="midnight", backupCount=30, encoding="utf-8"
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)

    # Formatter
    formatter = JsonFormatter()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
