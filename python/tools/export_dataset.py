"""Generate a labeled dataset in Edge Impulse data-acquisition (CBOR/JSON) format.

Produces one JSON file per window under ``dataset/<label>/`` using the SAME
ConveyorModel as the live simulator, so what you train on matches what runs.
Upload the folder to Edge Impulse Studio (or use the Ingestion API) to train the
Spectral Analysis + K-means anomaly-detection impulse.

Usage:
    python -m tools.export_dataset --per-class 60 --out dataset
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ventuno.config import CONFIG
from ventuno.simulator.conveyor import ConveyorModel, FAULT_MODES


def _ei_payload(window: dict, fs: int, label: str) -> dict:
    n = len(window["x"])
    values = [[float(window["x"][i]), float(window["y"][i]), float(window["z"][i])] for i in range(n)]
    return {
        "protected": {"ver": "v1", "alg": "none", "iat": int(time.time())},
        "signature": "0" * 64,
        "payload": {
            "device_name": "ventuno-conveyor-sim",
            "device_type": "VENTUNO_Q_SIM",
            "interval_ms": 1000.0 / fs,
            "sensors": [
                {"name": "accX", "units": "g"},
                {"name": "accY", "units": "g"},
                {"name": "accZ", "units": "g"},
            ],
            "values": values,
        },
        "_label": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60, help="windows per label")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--fs", type=int, default=CONFIG.sim_fs)
    ap.add_argument("--window", type=int, default=CONFIG.sim_window)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = ConveyorModel(fs=args.fs, rpm=CONFIG.sim_rpm, seed=args.seed)
    labels = ["normal", *FAULT_MODES]
    for label in labels:
        d = os.path.join(args.out, label)
        os.makedirs(d, exist_ok=True)
        fault = None if label == "normal" else label
        for i in range(args.per_class):
            severity = 1.0 if fault is None else float(model.rng.uniform(0.7, 1.3))
            win = model.generate_window(args.window, fault=fault, severity=severity)
            payload = _ei_payload(win, args.fs, label)
            with open(os.path.join(d, f"{label}.{i:03d}.json"), "w") as f:
                json.dump(payload, f)
        print(f"[dataset] wrote {args.per_class} windows -> {d}")

    print(f"[dataset] done. Root: {os.path.abspath(args.out)}")
    print("Train in Edge Impulse: Spectral Analysis (DSP) + Anomaly Detection (K-means).")


if __name__ == "__main__":
    main()
