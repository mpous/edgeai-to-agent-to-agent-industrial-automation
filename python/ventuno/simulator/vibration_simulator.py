"""Vibration simulator service.

Streams 3-axis conveyor vibration windows to the UNS ``vibration/raw`` topic in
real time. Injects a random fault on average every ``anomaly_period_s`` seconds.
Subscribes to ``health/state``: when the machine enters MAINTENANCE it clears
the active fault (the machine has been repaired), which closes the demo loop:

    healthy -> anomaly injected -> detected -> maintenance -> repaired -> healthy
"""
from __future__ import annotations

import threading
import time

from ..config import CONFIG
from ..mqtt_client import MqttClient, now_ts
from ..logbus import get_logger
from .. import uns
from .conveyor import ConveyorModel, FAULT_MODES

log = get_logger("sim")


class VibrationSimulator:
    def __init__(self, config=CONFIG):
        self.cfg = config
        self.model = ConveyorModel(fs=config.sim_fs, rpm=config.sim_rpm)
        self.mqtt = MqttClient(client_id="ventuno-simulator")
        self.rng = self.model.rng
        self._active_fault: str | None = None
        self._severity: float = 1.0
        self._under_maintenance = False
        self._next_anomaly_at = time.monotonic() + self._sample_interval()
        self._force = threading.Event()
        self._forced_fault: str | None = None
        self._stop = threading.Event()

    def _sample_interval(self) -> float:
        # Poisson-like arrival around the configured mean period.
        return max(20.0, self.rng.exponential(self.cfg.anomaly_period_s))

    def force_anomaly(self, fault: str | None = None) -> None:
        """Deterministic trigger for live demos (called by the dashboard)."""
        log.info("force_anomaly requested (fault=%s)", fault or "random")
        if self._under_maintenance or self._active_fault is not None:
            log.warning("force_anomaly ignored: under_maintenance=%s active_fault=%s",
                        self._under_maintenance, self._active_fault)
        self._forced_fault = fault
        self._force.set()

    def _on_state(self, topic: str, payload: dict) -> None:
        state = payload.get("state")
        log.info("state -> %s", state)
        if state in (uns.STATE_MAINTENANCE, uns.STATE_REPAIRING):
            self._under_maintenance = True
            if self._active_fault:
                print(f"[sim] {state} -> clearing fault '{self._active_fault}' (repaired)")
                log.info("%s -> clearing fault '%s' (repaired)", state, self._active_fault)
                self._active_fault = None
                self._next_anomaly_at = time.monotonic() + self._sample_interval()
        else:
            # Repair window closed (healthy) -> machine powers back up.
            if self._under_maintenance:
                print("[sim] maintenance closed -> machine running again")
                log.info("maintenance closed -> machine running again")
            self._under_maintenance = False

    def _maybe_trigger(self) -> None:
        if self._under_maintenance or self._active_fault is not None:
            return
        if self._force.is_set():
            self._force.clear()
            fault = self._forced_fault or self.rng.choice(FAULT_MODES)
            self._begin_fault(str(fault))
        elif time.monotonic() >= self._next_anomaly_at:
            self._begin_fault(str(self.rng.choice(FAULT_MODES)))

    def _begin_fault(self, fault: str) -> None:
        self._active_fault = fault
        self._severity = float(self.rng.uniform(0.7, 1.3))
        print(f"[sim] injecting fault '{fault}' severity={self._severity:.2f}")
        log.info("injecting fault '%s' severity=%.2f", fault, self._severity)

    def start(self) -> None:
        self.mqtt.connect()
        self.mqtt.subscribe(uns.STATE, self._on_state)
        self.mqtt.subscribe("agents/dashboard/command", self._on_command)
        self.mqtt.publish(uns.EDGE_STATUS, {"online": True, "app_ver": "0.1.0", "ts": now_ts()}, retain=True)
        thread = threading.Thread(target=self._run, name="simulator", daemon=True)
        thread.start()
        log.info("simulator started; publishing raw to %s every %.2fs",
                 uns.RAW, self.cfg.sim_window / self.cfg.sim_fs)

    def _on_command(self, topic: str, payload: dict) -> None:
        log.debug("command recv: %s", payload)
        if payload.get("cmd") == "force_anomaly":
            self.force_anomaly(payload.get("fault"))

    def _run(self) -> None:
        n = self.cfg.sim_window
        fs = self.cfg.sim_fs
        window_period = n / fs  # real-time cadence
        count = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            self._maybe_trigger()
            if self._under_maintenance:
                # Machine is powered down for repair: no vibration at all.
                zeros = [0.0] * n
                win = {"x": zeros, "y": zeros, "z": zeros}
                label = "stopped"
            else:
                w = self.model.generate_window(n, fault=self._active_fault, severity=self._severity)
                win = {k: [round(v, 5) for v in w[k].tolist()] for k in ("x", "y", "z")}
                label = self._active_fault or "normal"
            self.mqtt.publish(
                uns.RAW,
                {
                    "ts": now_ts(),
                    "fs_hz": fs,
                    "rpm": 0.0 if self._under_maintenance else self.cfg.sim_rpm,
                    "axis": win,
                    "_label": label,  # ground truth for dataset export
                },
            )
            count += 1
            # Heartbeat every ~10 windows so the log shows the stream is alive
            # without a line per second; DEBUG shows every publish via mqtt_client.
            if count == 1 or count % 10 == 0:
                log.info("published raw window #%d (label=%s, n=%d/axis)", count, label, n)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, window_period - elapsed))

    def stop(self) -> None:
        self._stop.set()
        self.mqtt.publish(uns.EDGE_STATUS, {"online": False, "app_ver": "0.1.0", "ts": now_ts()}, retain=True)
        self.mqtt.disconnect()


def main() -> None:
    sim = VibrationSimulator()
    sim.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
