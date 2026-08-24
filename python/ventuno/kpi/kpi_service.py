"""KPI service.

Tracks the machine state over time and publishes manufacturing KPIs to the UNS
so agents (and the dashboard) can reason over business metrics:

  * kpi/uptime      - operational vs downtime seconds
  * kpi/production  - real-time rate + cumulative units vs target
  * kpi/cost        - running / downtime / maintenance cost
  * kpi/oee         - Availability x Performance x Quality

State -> behaviour:
  healthy      -> full rate, full running cost
  anomaly      -> reduced rate + quality (machine limping), running cost
  maintenance  -> stopped (downtime), maintenance cost accrues
"""
from __future__ import annotations

import time

from ..config import CONFIG
from ..mqtt_client import MqttClient, now_ts
from ..logbus import get_logger
from .. import uns

log = get_logger("kpi")

TARGET_RATE = 60.0            # units / min at full performance
RUNNING_COST_PER_MIN = 2.0    # energy + labour while producing
DOWNTIME_COST_PER_MIN = 8.0   # lost-production cost while stopped
MAINT_COST_PER_MIN = 5.0      # technician + parts during maintenance
CURRENCY = "EUR"

# Performance / quality factors per state
FACTORS = {
    uns.STATE_HEALTHY: {"rate": 1.0, "quality": 0.99},
    uns.STATE_ANOMALY: {"rate": 0.6, "quality": 0.90},
    uns.STATE_MAINTENANCE: {"rate": 0.0, "quality": 1.0},
    uns.STATE_REPAIRING: {"rate": 0.0, "quality": 1.0},
}

# States where the machine is down (no production, incurs downtime + maint cost).
DOWN_STATES = (uns.STATE_MAINTENANCE, uns.STATE_REPAIRING)


class KpiService:
    def __init__(self, config=CONFIG):
        self.cfg = config
        self.mqtt = MqttClient(client_id="ventuno-kpi")
        self.state = uns.STATE_HEALTHY
        self._last = time.monotonic()
        self.op_time = 0.0
        self.down_time = 0.0
        self.units = 0.0
        self.good_units = 0.0
        self.running_cost = 0.0
        self.downtime_cost = 0.0
        self.maint_cost = 0.0

    def start(self) -> None:
        self.mqtt.connect()
        self.mqtt.subscribe(uns.STATE, self._on_state)
        self._publish()
        log.info("kpi service up (state=%s)", self.state)

    def _on_state(self, topic: str, payload: dict) -> None:
        self._accumulate()
        new = payload.get("state", self.state)
        if new != self.state:
            log.info("state %s -> %s", self.state, new)
        self.state = new

    def _accumulate(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        dt_min = dt / 60.0
        f = FACTORS.get(self.state, FACTORS[uns.STATE_HEALTHY])

        if self.state in DOWN_STATES:
            self.down_time += dt
            self.maint_cost += MAINT_COST_PER_MIN * dt_min
            self.downtime_cost += DOWNTIME_COST_PER_MIN * dt_min
        else:
            self.op_time += dt
            self.running_cost += RUNNING_COST_PER_MIN * dt_min
            produced = TARGET_RATE * f["rate"] * dt_min
            self.units += produced
            self.good_units += produced * f["quality"]

    def _oee(self) -> dict:
        total = self.op_time + self.down_time
        availability = (self.op_time / total) if total > 0 else 1.0
        f = FACTORS.get(self.state, FACTORS[uns.STATE_HEALTHY])
        performance = f["rate"] if self.state not in DOWN_STATES else 0.0
        quality = (self.good_units / self.units) if self.units > 0 else 1.0
        return {
            "availability": round(availability, 4),
            "performance": round(performance, 4),
            "quality": round(quality, 4),
            "oee": round(availability * performance * quality, 4),
        }

    def _publish(self) -> None:
        self._accumulate()
        f = FACTORS.get(self.state, FACTORS[uns.STATE_HEALTHY])
        ts = now_ts()
        self.mqtt.publish(uns.KPI_UPTIME, {
            "operational_time_s": round(self.op_time, 1),
            "downtime_s": round(self.down_time, 1),
            "state": self.state,
            "ts": ts,
        }, retain=True)
        self.mqtt.publish(uns.KPI_PRODUCTION, {
            "rate_units_min": round(TARGET_RATE * f["rate"], 2),
            "units_total": round(self.units, 1),
            "target_rate": TARGET_RATE,
            "ts": ts,
        }, retain=True)
        self.mqtt.publish(uns.KPI_COST, {
            "running_cost": round(self.running_cost, 2),
            "downtime_cost": round(self.downtime_cost, 2),
            "maintenance_cost": round(self.maint_cost, 2),
            "currency": CURRENCY,
            "ts": ts,
        }, retain=True)
        oee = self._oee()
        oee["ts"] = ts
        self.mqtt.publish(uns.KPI_OEE, oee, retain=True)

    def run(self, interval: float = 2.0) -> None:
        while True:
            time.sleep(interval)
            self._publish()

    def stop(self) -> None:
        self.mqtt.disconnect()


def main() -> None:
    svc = KpiService()
    svc.start()
    try:
        svc.run()
    except KeyboardInterrupt:
        svc.stop()


if __name__ == "__main__":
    main()
