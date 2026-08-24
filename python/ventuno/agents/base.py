"""Shared agent base: MQTT connection, rolling in-memory session memory, and
reasoning-trace publishing for dashboard Panel C.

Memory is intentionally in-RAM only (resets on restart), per the demo design.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

from ..config import CONFIG
from ..logbus import get_logger
from ..mqtt_client import MqttClient, now_ts
from .. import uns

log = get_logger("agent")

# A heartbeat callback returns (reasoning, decision). Returning a falsy
# reasoning means "stay quiet this tick" (e.g. the agent is mid-incident).
# Spelled with Optional rather than ``X | None``: this alias is evaluated at
# runtime (``from __future__ import annotations`` only defers real annotations),
# and the PEP 604 form would need Python 3.10+.
HeartbeatFn = Callable[[], tuple[Optional[str], Optional[dict]]]


class AgentBase:
    def __init__(self, name: str, memory_size: int = 20):
        self.name = name
        self.mqtt = MqttClient(client_id=f"ventuno-agent-{name}")
        self.memory: deque[dict] = deque(maxlen=memory_size)
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

    def connect(self) -> None:
        self.mqtt.connect()

    def start_heartbeat(self, status_fn: HeartbeatFn, interval: float | None = None) -> None:
        """Post a periodic 'all clear' trace to Panel C while the line is healthy.

        ``status_fn`` is polled every ``interval`` seconds (default
        ``CONFIG.heartbeat_s``); it returns ``(reasoning, decision)``. A falsy
        reasoning means the agent is busy (mid-incident) and skips this tick, so
        heartbeats never talk over an active investigation.
        """
        period = CONFIG.heartbeat_s if interval is None else interval
        if period <= 0:
            log.info("%s heartbeat disabled (interval=%s)", self.name, period)
            return
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, args=(status_fn, period),
            name=f"agent-{self.name}-heartbeat", daemon=True)
        self._hb_thread.start()
        log.info("%s heartbeat started (every %ss)", self.name, period)

    def _heartbeat_loop(self, status_fn: HeartbeatFn, period: float) -> None:
        while not self._hb_stop.wait(period):
            try:
                reasoning, decision = status_fn()
            except Exception as exc:  # a bad status probe must not kill the loop
                log.exception("%s heartbeat status error: %s", self.name, exc)
                continue
            if not reasoning:
                continue  # agent chose to stay quiet (busy with an incident)
            self.publish_trace("heartbeat", {"interval_s": period}, reasoning, decision)

    def stop_heartbeat(self) -> None:
        self._hb_stop.set()

    def remember(self, event: dict) -> None:
        event = {"ts": now_ts(), **event}
        self.memory.append(event)

    def publish_trace(self, event: str, context: dict, reasoning: str,
                      decision: dict | None = None, prompt: str | None = None) -> None:
        self.mqtt.publish(
            uns.agent_trace(self.name),
            {
                "ts": now_ts(),
                "agent": self.name,
                "event": event,
                "context": context,
                "reasoning": reasoning,
                "decision": decision or {},
                "prompt": prompt or "",
                "memory": list(self.memory),
            },
        )
