"""Shared feature extraction.

The SAME function feeds the anomaly model input vector and the
`vibration/features` UNS payload, guaranteeing the training pipeline (dataset
export) and the inference pipeline are identical.
"""
from __future__ import annotations

import numpy as np

N_BANDS = 8
AXES = ("x", "y", "z")


def _axis_stats(sig: np.ndarray) -> tuple[float, float, float]:
    sig = sig - sig.mean()
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    crest = float(peak / rms) if rms > 1e-9 else 0.0
    std = float(sig.std())
    kurt = float(np.mean((sig / std) ** 4)) if std > 1e-9 else 0.0
    return rms, crest, kurt


def compute_features(window: dict, fs: int, n_bands: int = N_BANDS) -> tuple[dict, np.ndarray]:
    """Return (summary_dict, feature_vector).

    ``window`` maps 'x'/'y'/'z' to equal-length sample sequences.
    ``summary_dict`` matches the UNS ``vibration/features`` payload.
    ``feature_vector`` is the flat input to the anomaly model.
    """
    vec: list[float] = []
    per_axis: dict[str, dict] = {}
    for ax in AXES:
        rms, crest, kurt = _axis_stats(np.asarray(window[ax], dtype=float))
        per_axis[ax] = {"rms": rms, "crest": crest, "kurtosis": kurt}
        vec += [rms, crest, kurt]

    mag = np.sqrt(
        np.asarray(window["x"], dtype=float) ** 2
        + np.asarray(window["y"], dtype=float) ** 2
        + np.asarray(window["z"], dtype=float) ** 2
    )
    mag = mag - mag.mean()
    n = len(mag)
    spec = np.abs(np.fft.rfft(mag * np.hanning(n))) if n else np.zeros(1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs) if n else np.zeros(1)

    edges = np.linspace(0.0, fs / 2.0, n_bands + 1)
    band_energy: list[float] = []
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        band_energy.append(float(np.sum(spec[mask] ** 2)))
    vec += band_energy

    overall_rms = float(np.sqrt(np.mean(mag ** 2))) if n else 0.0
    overall_kurt = max(per_axis[a]["kurtosis"] for a in AXES)
    overall_crest = max(per_axis[a]["crest"] for a in AXES)

    summary = {
        "rms": overall_rms,
        "kurtosis": overall_kurt,
        "crest": overall_crest,
        "band_energy": band_energy,
        "per_axis": per_axis,
    }
    return summary, np.asarray(vec, dtype=float)
