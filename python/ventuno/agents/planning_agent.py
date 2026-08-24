"""Planning agent.

Subscribes to ``maintenance/window`` and ``kpi/oee``. When a new maintenance
window is scheduled, it asks the local LLM to re-plan production for the
downtime (reroute/throttle/notify) and publishes the plan to ``production/plan``.
Keeps rolling memory of recent plans.
"""
from __future__ import annotations

import json
import time

from ..config import CONFIG
from ..mqtt_client import now_ts
from ..logbus import get_logger
from .. import uns
from .base import AgentBase
from .a2a import A2AServer
from .llm import build_llm

log = get_logger("planning")

SYSTEM = (
    "You are a production planning agent. A machine is going into a maintenance "
    "window. Re-plan production to minimise disruption. Respond ONLY as JSON with "
    "keys: plan_id, actions (list of strings), affected_orders (list of strings)."
)


class PlanningAgent(AgentBase):
    def __init__(self, config=CONFIG):
        super().__init__("planning")
        self.cfg = config
        self.llm = build_llm()
        self._latest_oee = 1.0
        self._handled: set[str] = set()
        self._active_window = False
        self.server = A2AServer(
            name="Planning Agent",
            description="Re-plans production around maintenance windows.",
            url=config.plan_url,
            skills=[{
                "id": "replan_production",
                "name": "Re-plan production",
                "description": "Adjust production plan for a maintenance window.",
                "tags": ["planning", "scheduling"],
            }],
            handler=lambda text: json.dumps({"note": "planning agent is event-driven"}),
        )

    def start(self) -> None:
        self.connect()
        self.mqtt.subscribe(uns.WINDOW, self._on_window)
        self.mqtt.subscribe(uns.KPI_OEE, self._on_oee)
        self.server.start(self.cfg.a2a_host, self.cfg.plan_port)
        self.start_heartbeat(self._heartbeat_status)
        print(f"[planning] A2A card at {self.cfg.plan_url}/.well-known/agent-card.json")
        log.info("A2A card at %s/.well-known/agent-card.json", self.cfg.plan_url)

    def _heartbeat_status(self) -> tuple[str | None, dict | None]:
        if self._active_window:
            return None, None  # busy re-planning around an open window
        return (
            f"Production plan steady: OEE {self._latest_oee:.0%}, no maintenance "
            f"windows pending.",
            {"oee": round(self._latest_oee, 4), "windows_pending": 0},
        )

    def _on_oee(self, topic: str, payload: dict) -> None:
        self._latest_oee = float(payload.get("oee", self._latest_oee))

    def _on_window(self, topic: str, payload: dict) -> None:
        if payload.get("status") == "closed":
            self._active_window = False
            return
        if payload.get("status") != "scheduled":
            return
        window_id = payload.get("id")
        if not window_id or window_id in self._handled:
            return
        self._handled.add(window_id)
        self._active_window = True
        log.info("new window %s -> re-planning with LLM (oee=%.2f)", window_id, self._latest_oee)

        context = {
            "maintenance_window": {"id": window_id, "start": payload.get("start"), "end": payload.get("end")},
            "window_id": window_id,
            "oee": self._latest_oee,
            "cause": payload.get("cause"),
            "affected_orders": ["WO-1001", "WO-1002"],
        }
        t0 = time.monotonic()
        raw, reasoning = self.llm.complete(SYSTEM, json.dumps(context))
        log.info("LLM replan returned in %.1fs (%d chars)", time.monotonic() - t0, len(raw or ""))
        try:
            plan = json.loads(raw)
        except ValueError:
            log.warning("LLM replan response not valid JSON; using fallback plan")
            plan = {"plan_id": f"plan-{window_id}", "actions": [raw[:200]], "affected_orders": []}
        plan["ts"] = now_ts()

        self.remember({"event": "replan", "window_id": window_id, "actions": plan.get("actions")})
        self.publish_trace("replan_production", context, reasoning, plan,
                           prompt=self.llm.last_prompt)
        self.mqtt.publish(uns.PLAN, plan)
        print(f"[planning] published plan {plan.get('plan_id')} for window {window_id}")
        log.info("published plan %s for window %s", plan.get("plan_id"), window_id)


def main() -> None:
    agent = PlanningAgent()
    agent.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
