"""Maintenance agent.

Subscribes to ``health/anomaly``. On a persistent anomaly (verdict=True) and no
open work order, it asks the local LLM to triage: repair vs false alarm, with a
cause hypothesis and rationale. On 'repair' it publishes a work order and
delegates scheduling to the corporate agent over A2A. Keeps rolling memory of
recent anomalies and decisions.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque

from ..config import CONFIG
from ..mqtt_client import now_ts
from ..logbus import get_logger
from .. import uns
from .base import AgentBase
from .a2a import A2AServer, a2a_send
from .llm import build_llm

log = get_logger("maintenance")

SYSTEM = (
    "You are a maintenance reliability engineer for a factory conveyor. "
    "Given a vibration anomaly, decide whether it needs repair or is a false alarm. "
    "Respond ONLY as JSON with keys: decision ('repair'|'false_alarm'), "
    "cause_hypothesis, severity ('low'|'medium'|'high'), rationale."
)


class MaintenanceAgent(AgentBase):
    def __init__(self, config=CONFIG):
        super().__init__("maintenance")
        self.cfg = config
        self.llm = build_llm()
        self._open_workorder = False
        self._recent_scores: deque[float] = deque(maxlen=10)
        self._suspected_fault = "unknown"
        self.server = A2AServer(
            name="Maintenance Agent",
            description="Triages vibration anomalies and orders inspections.",
            url=config.maint_url,
            skills=[{
                "id": "triage_anomaly",
                "name": "Triage anomaly",
                "description": "Assess a vibration anomaly and decide on repair.",
                "tags": ["maintenance", "diagnostics"],
            }],
            handler=lambda text: json.dumps({"note": "maintenance agent is event-driven"}),
        )

    def start(self) -> None:
        self.connect()
        self.mqtt.subscribe(uns.ANOMALY, self._on_anomaly)
        self.mqtt.subscribe(uns.STATE, self._on_state)
        self.mqtt.subscribe(uns.RAW, self._on_raw)  # peek ground-truth label (stand-in for a fault classifier)
        self.server.start(self.cfg.a2a_host, self.cfg.maint_port)
        self.start_heartbeat(self._heartbeat_status)
        print(f"[maintenance] A2A card at {self.cfg.maint_url}/.well-known/agent-card.json")
        log.info("A2A card at %s/.well-known/agent-card.json", self.cfg.maint_url)

    def _heartbeat_status(self) -> tuple[str | None, dict | None]:
        if self._open_workorder:
            return None, None  # busy with an active work order
        peak = max(self._recent_scores) if self._recent_scores else 0.0
        return (
            f"Conveyor nominal: peak anomaly score {peak:.3f} vs threshold "
            f"{self.cfg.anomaly_threshold}. No maintenance required.",
            {"peak_score": round(peak, 4), "threshold": self.cfg.anomaly_threshold,
             "open_workorder": False},
        )

    def _on_raw(self, topic: str, payload: dict) -> None:
        self._suspected_fault = payload.get("_label", self._suspected_fault)

    def _on_state(self, topic: str, payload: dict) -> None:
        if payload.get("state") == uns.STATE_HEALTHY:
            if self._open_workorder:
                log.info("state healthy -> clearing open work order flag")
            self._open_workorder = False  # ready for the next event

    def _on_anomaly(self, topic: str, payload: dict) -> None:
        self._recent_scores.append(float(payload.get("anomaly_score", 0)))
        log.debug("anomaly recv: score=%s verdict=%s open_wo=%s",
                  payload.get("anomaly_score"), payload.get("verdict"), self._open_workorder)
        if not payload.get("verdict") or self._open_workorder:
            return
        log.info("verdict=True -> triaging with LLM (score=%s consecutive=%s)",
                 payload.get("anomaly_score"), payload.get("consecutive"))

        context = {
            "anomaly_score": payload.get("anomaly_score"),
            "consecutive": payload.get("consecutive"),
            "threshold": payload.get("threshold"),
            "suspected_fault": self._suspected_fault,
            "recent_scores": list(self._recent_scores),
            "model_ver": payload.get("model_ver"),
        }
        t0 = time.monotonic()
        raw, reasoning = self.llm.complete(SYSTEM, json.dumps(context))
        log.info("LLM triage returned in %.1fs (%d chars)", time.monotonic() - t0, len(raw or ""))
        try:
            verdict = json.loads(raw)
        except ValueError:
            log.warning("LLM response was not valid JSON; defaulting to repair")
            verdict = {"decision": "repair", "cause_hypothesis": self._suspected_fault,
                       "severity": "medium", "rationale": raw[:300]}

        self.remember({"event": "triage", "score": context["anomaly_score"],
                       "decision": verdict.get("decision"), "cause": verdict.get("cause_hypothesis")})
        self.publish_trace("triage", context, verdict.get("rationale", reasoning), verdict,
                           prompt=self.llm.last_prompt)

        if verdict.get("decision") != "repair":
            print(f"[maintenance] false alarm (score={context['anomaly_score']})")
            log.info("decision=false_alarm (score=%s)", context["anomaly_score"])
            return

        wo = {
            "id": f"WO-{uuid.uuid4().hex[:6]}",
            "cause_hypothesis": verdict.get("cause_hypothesis"),
            "severity": verdict.get("severity"),
            "decision": "repair",
            "rationale": verdict.get("rationale"),
            "ts": now_ts(),
        }
        self.mqtt.publish(uns.WORKORDER, wo)
        self._open_workorder = True
        print(f"[maintenance] repair -> work order {wo['id']} ({wo['cause_hypothesis']})")
        log.info("decision=repair -> work order %s (%s, severity=%s)",
                 wo["id"], wo["cause_hypothesis"], wo["severity"])

        # Delegate scheduling to the corporate agent over A2A.
        try:
            log.info("A2A -> corporate %s to schedule %s", self.cfg.corp_url, wo["id"])
            result = a2a_send(self.cfg.corp_url, json.dumps(wo))
            window = json.loads(result) if result else {}
            self.remember({"event": "scheduled", "workorder": wo["id"], "window": window.get("id")})
            self.publish_trace("delegated_schedule", {"workorder": wo["id"]},
                               f"Corporate agent booked window {window.get('id')}.", window)
            log.info("A2A schedule ok -> window %s", window.get("id"))
        except Exception as exc:
            print(f"[maintenance] A2A schedule failed: {exc}")
            log.exception("A2A schedule failed: %s", exc)


def main() -> None:
    agent = MaintenanceAgent()
    agent.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
