"""Physics-based conveyor-belt vibration model.

Generates realistic 3-axis accelerometer windows for a belt conveyor driven by
an induction motor + gearbox + rolling-element bearings, with configurable
fault modes. Amplitudes are in g (approximate, for demo realism).

Signature content:
  * running speed 1x (fr) + harmonics 2x, 3x
  * rolling-element bearing tones: BPFO, BPFI, BSF, FTF (as multiples of fr)
  * belt / idler-roller pass frequency
  * broadband Gaussian noise (sensor + process floor)

Fault modes (injected):
  * imbalance      -> elevated 1x (radial)
  * misalignment   -> elevated 2x/3x + axial component
  * bearing_spall  -> BPFO tone + fr sidebands + high-frequency impact bursts
  * belt_slip      -> amplitude modulation at belt frequency + belt harmonics
  * degradation    -> gradually rising broadband RMS and noise floor
"""
from __future__ import annotations

import numpy as np

FAULT_MODES = ("imbalance", "misalignment", "bearing_spall", "belt_slip", "degradation")


class ConveyorModel:
    def __init__(
        self,
        fs: int = 2000,
        rpm: float = 1450.0,
        # 6205-style deep-groove ball bearing geometry
        n_balls: int = 9,
        ball_dia_mm: float = 7.94,
        pitch_dia_mm: float = 39.04,
        contact_angle_deg: float = 0.0,
        belt_freq_hz: float = 3.1,
        idler_freq_hz: float = 11.5,
        seed: int | None = None,
    ):
        self.fs = fs
        self.fr = rpm / 60.0
        self.belt_freq = belt_freq_hz
        self.idler_freq = idler_freq_hz
        self.rng = np.random.default_rng(seed)

        ratio = (ball_dia_mm / pitch_dia_mm) * np.cos(np.radians(contact_angle_deg))
        self.bpfo = (n_balls / 2.0) * (1.0 - ratio) * self.fr
        self.bpfi = (n_balls / 2.0) * (1.0 + ratio) * self.fr
        self.bsf = (pitch_dia_mm / (2.0 * ball_dia_mm)) * (1.0 - ratio ** 2) * self.fr
        self.ftf = 0.5 * (1.0 - ratio) * self.fr

    def fault_frequencies(self) -> dict:
        return {
            "fr": self.fr,
            "bpfo": self.bpfo,
            "bpfi": self.bpfi,
            "bsf": self.bsf,
            "ftf": self.ftf,
            "belt": self.belt_freq,
            "idler": self.idler_freq,
        }

    def _tone(self, t: np.ndarray, freq: float, amp: float, phase: float | None = None) -> np.ndarray:
        if phase is None:
            phase = self.rng.uniform(0, 2 * np.pi)
        return amp * np.sin(2 * np.pi * freq * t + phase)

    def _impacts(self, t: np.ndarray, freq: float, amp: float, resonance: float = 750.0) -> np.ndarray:
        """Periodic decaying impacts (bearing spall) at ``freq`` exciting a resonance."""
        n = len(t)
        out = np.zeros(n)
        period = int(self.fs / freq) if freq > 0 else n
        if period <= 0:
            return out
        decay = np.exp(-np.linspace(0, 1, period) * 12.0)
        ring = np.sin(2 * np.pi * resonance * np.arange(period) / self.fs) * decay
        for start in range(self.rng.integers(0, period), n, period):
            end = min(start + period, n)
            out[start:end] += amp * ring[: end - start]
        return out

    def generate_window(self, n: int, fault: str | None = None, severity: float = 1.0) -> dict:
        t = np.arange(n) / self.fs
        fr = self.fr

        # Healthy baseline, distributed across radial (x,y) and axial (z)
        x = self._tone(t, fr, 0.30) + self._tone(t, 2 * fr, 0.08) + self._tone(t, 3 * fr, 0.04)
        y = self._tone(t, fr, 0.26) + self._tone(t, 2 * fr, 0.07)
        z = self._tone(t, fr, 0.06) + self._tone(t, self.idler_freq, 0.05)

        # low-level bearing + belt content present even when healthy
        x += self._tone(t, self.bpfo, 0.02) + self._tone(t, self.belt_freq, 0.03)
        y += self._tone(t, self.bpfi, 0.02)

        noise = 0.05
        if fault == "imbalance":
            x += self._tone(t, fr, 0.55 * severity)
            y += self._tone(t, fr, 0.50 * severity)
        elif fault == "misalignment":
            x += self._tone(t, 2 * fr, 0.45 * severity) + self._tone(t, 3 * fr, 0.20 * severity)
            z += self._tone(t, fr, 0.35 * severity) + self._tone(t, 2 * fr, 0.25 * severity)
        elif fault == "bearing_spall":
            x += self._tone(t, self.bpfo, 0.20 * severity)
            # fr sidebands around BPFO
            x += self._tone(t, self.bpfo + fr, 0.10 * severity)
            x += self._tone(t, self.bpfo - fr, 0.10 * severity)
            burst = self._impacts(t, self.bpfo, 0.8 * severity)
            x += burst
            y += 0.6 * burst
        elif fault == "belt_slip":
            mod = 1.0 + 0.6 * severity * np.sin(2 * np.pi * self.belt_freq * t)
            x *= mod
            y *= mod
            x += self._tone(t, 2 * self.belt_freq, 0.15 * severity)
        elif fault == "degradation":
            gain = 1.0 + 0.8 * severity
            x *= gain
            y *= gain
            z *= gain
            noise = 0.05 + 0.15 * severity

        x = x + self.rng.normal(0, noise, n)
        y = y + self.rng.normal(0, noise, n)
        z = z + self.rng.normal(0, noise, n)
        return {"x": x, "y": y, "z": z}
