"""Corporate / business agent (mock CMMS/ERP).

A2A server exposing a ``schedule_inspection`` skill. On request it books an
inspection slot + maintenance window, publishes the window to the UNS, and
drives the machine into the MAINTENANCE state. A timer closes the window after
``maint_duration_s`` and returns the machine to HEALTHY (closing the demo loop).
"""
from __future__ import annotations

import json
import threading
import time
import uuid

from ..config import CONFIG
from ..mqtt_client import now_ts
from ..logbus import get_logger
from .. import uns
from .base import AgentBase
from .a2a import A2AServer

log = get_logger("corporate")


class CorporateAgent(AgentBase):
    def __init__(self, config=CONFIG):
        super().__init__("corporate")
        self.cfg = config
        self._open_window = False
        self.server = A2AServer(
            name="Corporate Scheduling Agent",
            description="Mock CMMS/ERP that books maintenance inspections.",
            url=config.corp_url,
            skills=[{
                "id": "schedule_inspection",
                "name": "Schedule inspection",
                "description": "Book an inspection slot and open a maintenance window.",
                "tags": ["maintenance", "cmms", "erp"],
            }],
            handler=self._handle,
        )

    def start(self) -> None:
        self.connect()
        self.server.start(self.cfg.a2a_host, self.cfg.corp_port)
        self.start_heartbeat(self._heartbeat_status)
        print(f"[corporate] A2A card at {self.cfg.corp_url}/.well-known/agent-card.json")
        log.info("A2A card at %s/.well-known/agent-card.json", self.cfg.corp_url)

    def _heartbeat_status(self) -> tuple[str | None, dict | None]:
        if self._open_window:
            return None, None  # busy managing a maintenance window
        return (
            "Schedule clear: no maintenance windows open and no work orders pending. "
            "Conveyor running normally.",
            {"windows_open": 0, "status": "nominal"},
        )

    def _handle(self, text: str) -> str:
        log.info("A2A request recv (%d chars)", len(text or ""))
        try:
            wo = json.loads(text)
        except ValueError:
            wo = {"raw": text}

        window_id = f"MW-{uuid.uuid4().hex[:6]}"
        self._open_window = True
        start = now_ts()
        end = start + self.cfg.maint_duration_s
        window = {"id": window_id, "start": start, "end": end, "status": "scheduled",
                  "workorder": wo.get("id"), "cause": wo.get("cause_hypothesis")}

        reasoning = (
            f"Received work order {wo.get('id')} (cause: {wo.get('cause_hypothesis')}, "
            f"severity: {wo.get('severity')}). Next technician slot available now; "
            f"opening a {int(self.cfg.maint_duration_s)}s maintenance window {window_id}."
        )
        self.remember({"event": "scheduled", "window_id": window_id, "workorder": wo.get("id")})
        self.publish_trace("schedule_inspection", {"workorder": wo}, reasoning, window)

        # Publish window + drive machine into maintenance.
        self.mqtt.publish(uns.WINDOW, window, retain=True)
        self.mqtt.publish(uns.STATE, {"state": uns.STATE_MAINTENANCE, "ts": now_ts()}, retain=True)
        log.info("opened window %s for WO %s -> state=maintenance (%ds)",
                 window_id, wo.get("id"), int(self.cfg.maint_duration_s))

        threading.Thread(target=self._close_later, args=(window,), daemon=True).start()
        return json.dumps(window)

    def _close_later(self, window: dict) -> None:
        time.sleep(self.cfg.maint_duration_s)
        # Brief "repairing" phase: the technician is finishing the fix before the
        # machine is handed back to production.
        self.mqtt.publish(uns.STATE, {"state": uns.STATE_REPAIRING, "ts": now_ts()}, retain=True)
        self.remember({"event": "repairing", "window_id": window["id"]})
        self.publish_trace("repairing", {"window_id": window["id"]},
                           "Maintenance window elapsed; technician finishing the repair before restart.",
                           window)
        log.info("window %s elapsed -> state=repairing", window["id"])
        time.sleep(3.0)
        window = {**window, "status": "closed", "end": now_ts()}
        self.mqtt.publish(uns.WINDOW, window, retain=True)
        self.mqtt.publish(uns.STATE, {"state": uns.STATE_HEALTHY, "ts": now_ts()}, retain=True)
        self.remember({"event": "closed", "window_id": window["id"]})
        self.publish_trace("close_window", {"window_id": window["id"]},
                           "Repair complete; returning machine to service.",
                           window)
        log.info("window %s closed -> state=healthy", window["id"])
        self._open_window = False


def main() -> None:
    agent = CorporateAgent()
    agent.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
