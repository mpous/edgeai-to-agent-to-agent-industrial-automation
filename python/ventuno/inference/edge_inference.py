"""Edge inference service.

Subscribes to ``vibration/raw``, computes features (shared pipeline), scores the
window with the anomaly model, and publishes:
  * ``vibration/features``  - the feature summary
  * ``health/anomaly``      - score / threshold / verdict
  * ``health/state``        - healthy | anomaly  (never overrides 'maintenance')

Verdict requires the score to exceed the threshold for ``anomaly_persist``
consecutive windows, so a single noisy window does not raise an alarm.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import CONFIG
from ..mqtt_client import MqttClient, now_ts
from ..logbus import get_logger
from .. import uns
from ..features import compute_features
from .models import build_model

log = get_logger("edge")


class EdgeInference:
    def __init__(self, config=CONFIG):
        self.cfg = config
        self.mqtt = MqttClient(client_id="ventuno-inference")
        self.model = build_model(config.model_backend, config.eim_path)
        self._consecutive = 0
        self._current_state = uns.STATE_HEALTHY

    def start(self) -> None:
        self.mqtt.connect()
        self.mqtt.subscribe(uns.RAW, self._on_raw)
        self.mqtt.subscribe(uns.STATE, self._on_state)
        self.mqtt.publish(uns.STATE, {"state": uns.STATE_HEALTHY, "ts": now_ts()}, retain=True)
        self._publish_model_info()
        info = getattr(self.model, "info", None)
        info = info() if callable(info) else {"version": self.model.version}
        print(f"[edge] inference up: backend={self.cfg.model_backend} model={info} "
              f"threshold={self.cfg.anomaly_threshold} persist={self.cfg.anomaly_persist}")
        log.info("inference up: backend=%s threshold=%s persist=%s model=%s",
                 self.cfg.model_backend, self.cfg.anomaly_threshold, self.cfg.anomaly_persist, info)

    def _publish_model_info(self) -> None:
        info = getattr(self.model, "info", None)
        payload = info() if callable(info) else {"backend": self.cfg.model_backend, "version": self.model.version}
        payload["ts"] = now_ts()
        self.mqtt.publish(uns.MODEL, payload, retain=True)

    def _on_state(self, topic: str, payload: dict) -> None:
        self._current_state = payload.get("state", self._current_state)
        log.debug("state -> %s", self._current_state)

    def _on_raw(self, topic: str, payload: dict) -> None:
        axis = payload.get("axis", {})
        if not axis.get("x"):
            log.warning("raw window with no axis data (keys=%s); skipping", list(payload.keys()))
            return
        fs = int(payload.get("fs_hz", self.cfg.sim_fs))
        log.debug("raw window recv: n=%d/axis fs=%d state=%s label=%s",
                  len(axis.get("x", [])), fs, self._current_state, payload.get("_label"))
        summary, vec = compute_features(axis, fs)
        summary.update({"ts": now_ts()})
        self.mqtt.publish(uns.FEATURES, summary)

        # While the machine is stopped for repair it emits no vibration; don't
        # let the model read the flat window as an anomaly. Report a clean score.
        if self._current_state in (uns.STATE_MAINTENANCE, uns.STATE_REPAIRING):
            self._consecutive = 0
            log.debug("state=%s -> reporting clean score 0.0 (machine down)", self._current_state)
            self.mqtt.publish(
                uns.ANOMALY,
                {
                    "ts": now_ts(),
                    "anomaly_score": 0.0,
                    "threshold": self.cfg.anomaly_threshold,
                    "verdict": False,
                    "consecutive": 0,
                    "model_ver": self.model.version,
                },
            )
            return

        score = float(self.model.score(vec, raw_axis=axis, fs=fs))
        threshold = self.cfg.anomaly_threshold
        over = score > threshold
        self._consecutive = self._consecutive + 1 if over else 0
        verdict = self._consecutive >= self.cfg.anomaly_persist
        print(f"[edge] score={score:.4f} thr={threshold} over={over} "
              f"consecutive={self._consecutive}/{self.cfg.anomaly_persist} verdict={verdict}")
        log.info("score=%.4f thr=%s over=%s consecutive=%d/%d verdict=%s",
                 score, threshold, over, self._consecutive, self.cfg.anomaly_persist, verdict)

        self.mqtt.publish(
            uns.ANOMALY,
            {
                "ts": now_ts(),
                "anomaly_score": round(score, 4),
                "threshold": threshold,
                "verdict": verdict,
                "consecutive": self._consecutive,
                "model_ver": self.model.version,
            },
        )

        # Only the edge owns the healthy/anomaly transition. Never stomp a
        # maintenance/repair window — whoever opened it owns that state (nobody
        # in Part 1; the maintenance agent in Part 2; corporate in Part 3).
        if self._current_state not in (uns.STATE_MAINTENANCE, uns.STATE_REPAIRING):
            new_state = uns.STATE_ANOMALY if verdict else uns.STATE_HEALTHY
            if new_state != self._current_state:
                self._current_state = new_state
                log.info("state transition -> %s (verdict=%s)", new_state, verdict)
                self.mqtt.publish(uns.STATE, {"state": new_state, "ts": now_ts()}, retain=True)

    def stop(self) -> None:
        stop = getattr(self.model, "stop", None)
        if callable(stop):
            stop()
        self.mqtt.disconnect()


def main() -> None:
    svc = EdgeInference()
    svc.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        svc.stop()


if __name__ == "__main__":
    main()
