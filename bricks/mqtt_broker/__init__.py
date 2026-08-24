"""Companion Python API for the Mosquitto broker Custom Brick.

This code runs inside the App's main container (per App Lab's brick model), not
inside the broker container. It only exposes where the broker lives on the
Compose network and a small readiness helper; the broker itself is the
container defined in ``brick_compose.yaml``.
"""
from __future__ import annotations

import socket
import time

# Reachable from the App container by the compose service name.
BROKER_HOST = "mqtt_broker"
BROKER_PORT = 1883


def wait_until_ready(timeout: float = 30.0, host: str = BROKER_HOST,
                     port: int = BROKER_PORT) -> bool:
    """Block until the broker accepts TCP connections (or timeout). Returns True
    if the broker became reachable. Dependency-free (plain socket)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False
