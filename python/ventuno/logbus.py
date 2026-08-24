"""Central logging setup for every VENTUNO service.

One place configures the root logger so all modules share a consistent format and
level. Level comes from ``LOG_LEVEL`` (default ``INFO``); set ``LOG_LEVEL=DEBUG``
to see every MQTT publish/receive and per-window inference detail.

Usage:
    from ..logbus import get_logger
    log = get_logger("sim")
    log.info("started")
"""
from __future__ import annotations

import logging
import os
import sys

_configured = False


def setup_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, name, logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
