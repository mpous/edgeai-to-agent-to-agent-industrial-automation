"""Pluggable anomaly models.

* StatisticalModel  - dependency-free baseline. Learns feature mean/std during a
  healthy warm-up, then scores windows by mean z-distance mapped to 0..1.
  Stands in for the Edge Impulse model so the whole demo runs on a laptop.
* EIModel           - runs a real Edge Impulse ``.eim`` (Spectral Analysis +
  K-means anomaly detection) via ``edge_impulse_linux``. Used on the VENTUNO Q.
* BrickModel        - runs the App Lab ``vibration_anomaly_detection`` brick,
  which serves a deployed Edge Impulse model through the App Lab EI runner.
  Board-only; ingests the raw window we publish over the UNS.

All expose: score(vec, raw_axis=None, fs=None) -> float in [0, 1], and a
``version`` string. Statistical/EI use the computed feature ``vec``; the brick
uses the raw ``raw_axis`` samples (it does its own DSP internally).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..logbus import get_logger

log = get_logger("model")


class StatisticalModel:
    version = "statistical-0.1"

    def __init__(self, warmup: int = 20):
        self.warmup = warmup
        self._samples: list[np.ndarray] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.ready = False

    def score(self, vec: np.ndarray, raw_axis: dict | None = None, fs: int | None = None) -> float:
        if not self.ready:
            self._samples.append(vec)
            if len(self._samples) >= self.warmup:
                arr = np.asarray(self._samples)
                self._mean = arr.mean(axis=0)
                self._std = arr.std(axis=0) + 1e-6
                self.ready = True
            return 0.0
        z = np.abs((vec - self._mean) / self._std)
        return float(np.tanh(np.mean(z) / 3.0))

    def info(self) -> dict:
        return {
            "backend": "statistical",
            "version": self.version,
            "ready": self.ready,
            "note": "z-distance baseline (no Edge Impulse model deployed)",
        }


class EIModel:
    version = "eim"

    def __init__(self, model_path: str):
        from edge_impulse_linux.runner import ImpulseRunner  # imported lazily (board only)

        self._runner = ImpulseRunner(model_path)
        info = self._runner.init()
        self.version = f"eim:{info.get('project', {}).get('name', 'model')}"

    def score(self, vec: np.ndarray, raw_axis: dict | None = None, fs: int | None = None) -> float:
        res = self._runner.classify(vec.tolist())
        result = res.get("result", {})
        # K-means anomaly detection returns an "anomaly" score (higher = worse).
        anomaly = result.get("anomaly")
        if anomaly is not None:
            return float(np.tanh(float(anomaly) / 3.0))
        # Fallback: 1 - P(normal) if a classifier head is present.
        classification = result.get("classification", {})
        if classification:
            return float(1.0 - classification.get("normal", 0.0))
        return 0.0

    def stop(self) -> None:
        try:
            self._runner.stop()
        except Exception:
            pass

    def info(self) -> dict:
        return {
            "backend": "eim",
            "version": self.version,
            "ready": True,
            "note": "Edge Impulse .eim runner",
        }


class BrickModel:
    """App Lab ``vibration_anomaly_detection`` brick (board-only).

    The brick buffers raw accelerometer samples into a sliding window sized to
    the EI model's ``input_features_count``, runs inference, and fires
    ``on_anomaly`` when the raw anomaly score crosses a threshold. We set that
    threshold to 0 so the callback fires every window, capture the latest score,
    and let ``edge_inference`` apply its own threshold + persistence logic.

    We are **simulating** the accelerometer: there is no MCU sketch or Router
    Bridge here. In the stock App Lab example the microcontroller pushes one
    live reading at a time via ``Bridge.provide("record_sensor_movement", ...)``
    which calls ``vibration.accumulate_samples((x, y, z))``. Instead we replay
    the synthetic window we publish on ``vibration/raw``, feeding it to the same
    ``accumulate_samples()`` one ``(x, y, z)`` triple at a time. Units, axis
    order, and rate must still match the EI model's training data.
    """

    version = "brick-vibration"

    def __init__(self):
        from arduino.app_bricks.vibration_anomaly_detection import VibrationAnomalyDetection

        self._brick = VibrationAnomalyDetection(anomaly_detection_threshold=0.0)
        self._last_score = 0.0
        self._capture_count = 0
        self._fed_count = 0
        self._stop = threading.Event()
        # The brick's loop() only invokes the callback if inspect.isfunction()
        # is true for it — which is FALSE for a bound method. Register a plain
        # closure (isfunction=True) instead, or the callback silently never
        # fires and _last_score stays 0. Signature keeps a 'classification' arg
        # so the brick's 2-arg dispatch path matches.
        def _on_anomaly(anomaly_score, classification=None):
            self._capture(anomaly_score, classification)
        self._on_anomaly = _on_anomaly  # keep a ref; the brick stores it weakly-ish
        self._brick.on_anomaly(self._on_anomaly)
        start = getattr(self._brick, "start", None)
        if callable(start):
            start()
        info = self._brick.get_model_info()
        freq = int(getattr(info, "frequency", 0) or 0)
        self._freq = freq
        self._features = int(getattr(info, "input_features_count", 0) or 0)
        self.version = f"brick-vibration:{freq}Hz"
        print(f"[brick] vibration model ready: freq={freq}Hz "
              f"input_features={self._features}")
        log.info("vibration brick ready: freq=%dHz input_features=%d", freq, self._features)

        # The brick's loop() is a blocking event-pump (normally driven by
        # App.run()). This app runs Flask instead, so we pump it here in a
        # daemon thread: it parks inside loop() until samples fed via
        # accumulate_samples() complete a window, fires on_anomaly, then repeats.
        # Keeping it off the caller's thread is what stops score() from freezing
        # the MQTT network loop.
        self._pump = threading.Thread(target=self._run_loop, name="brick-loop", daemon=True)
        self._pump.start()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._brick.loop()
            except Exception as exc:  # keep pumping across transient brick errors
                log.exception("brick loop() error: %s", exc)
                time.sleep(0.5)
            else:
                time.sleep(0.01)  # avoid a hot spin if loop() returns immediately

    def _capture(self, anomaly_score: float, classification: dict | None = None) -> None:
        self._last_score = float(anomaly_score)
        self._capture_count += 1
        print(f"[brick] on_anomaly #{self._capture_count} raw_score={anomaly_score}")
        log.info("on_anomaly #%d raw_score=%s", self._capture_count, anomaly_score)

    def score(self, vec: np.ndarray, raw_axis: dict | None = None, fs: int | None = None) -> float:
        if not raw_axis:
            return float(np.tanh(self._last_score / 3.0))
        x, y, z = raw_axis.get("x", []), raw_axis.get("y", []), raw_axis.get("z", [])
        n = min(len(x), len(y), len(z))
        if n:
            # accumulate_samples() takes ONE (x, y, z) triple per call (the App
            # Lab example feeds one MCU reading at a time). We're simulating the
            # sensor, so we replay the whole window here, one triple per call,
            # rather than reading a physical accelerometer over the bridge.
            for i in range(n):
                self._brick.accumulate_samples((x[i], y[i], z[i]))
            self._fed_count += 1
            log.debug("fed %d triples (x,y,z); captured=%d", n, self._capture_count)
            # on_anomaly fires from the pump thread once enough triples span a
            # full window. Warn once if we've fed several windows and it still
            # hasn't fired — usually a feature-count/axis/rate mismatch.
            if self._capture_count == 0 and self._fed_count == 5:
                log.warning("fed %d windows (%d triples each) but on_anomaly "
                            "has not fired yet; brick needs input_features=%d "
                            "(check window size, axis order, and rate)",
                            self._fed_count, n, self._features)
        # Raw EI anomaly score is a distance (can exceed 1); squash to 0..1 to
        # match the threshold scale the rest of the pipeline expects.
        return float(np.tanh(self._last_score / 3.0))

    def stop(self) -> None:
        self._stop.set()
        stop = getattr(self._brick, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass

    def info(self) -> dict:
        return {
            "backend": "brick",
            "version": self.version,
            "ready": True,
            "frequency": self._freq,
            "input_features": self._features,
            "note": "App Lab vibration_anomaly_detection brick (live Edge Impulse inference)",
        }


def build_model(backend: str, eim_path: str):
    if backend == "brick":
        return BrickModel()
    if backend == "eim":
        return EIModel(eim_path)
    return StatisticalModel()
