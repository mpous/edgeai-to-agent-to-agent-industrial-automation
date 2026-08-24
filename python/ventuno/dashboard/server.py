"""Operations dashboard (Flask + Server-Sent Events).

Subscribes to the whole UNS + agent traces, keeps the latest state, and streams
updates to the browser. Four panels:
  A - vibration & real-time inference (+ Edge Impulse record/ingest, UI-entered key)
  B - UNS live topic tree
  C - agents thinking (context, reasoning, memory) + A2A trace timeline
  D - manufacturing KPIs (OEE / time / cost / production)
"""
from __future__ import annotations

import json
import queue
import threading
import time

import requests
from flask import Flask, Response, jsonify, render_template, request

from ..config import CONFIG
from ..mqtt_client import MqttClient, now_ts
from ..logbus import get_logger
from .. import uns

log = get_logger("dashboard")

app = Flask(__name__)

# Latest payload per topic (for initial render + Panel B tree).
_state: dict[str, dict] = {}
# Ring buffer of agent trace events (Panel C timeline).
_agent_events: list[dict] = []
# Last raw window (for Panel A waveform + EI ingestion).
_last_raw: dict = {}
# Active EI recording session (full-res windows collected server-side).
_record: dict = {"active": False, "buf": [], "target": 0}
# Connected SSE clients.
_subscribers: list[queue.Queue] = []
_lock = threading.Lock()

_mqtt = MqttClient(client_id="ventuno-dashboard")


def _broadcast(kind: str, topic: str, payload: dict) -> None:
    msg = json.dumps({"kind": kind, "topic": topic, "payload": payload, "ts": now_ts()})
    with _lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def _on_uns(topic: str, payload: dict) -> None:
    global _last_raw
    if topic == uns.RAW:
        _last_raw = payload
        # If a recording session is active, keep the FULL-res window for EI.
        with _lock:
            if _record["active"] and len(_record["buf"]) < _record["target"]:
                _record["buf"].append(payload)
        # Downsample the waveform for the browser (don't ship 2000 pts/axis).
        axis = payload.get("axis", {})
        step = max(1, len(axis.get("x", [])) // 200)
        light = {
            "ts": payload.get("ts"),
            "fs_hz": payload.get("fs_hz"),
            "axis": {k: axis.get(k, [])[::step] for k in ("x", "y", "z")},
        }
        _broadcast("raw", topic, light)
        return

    # Agent reasoning traces now live INSIDE the UNS (at each agent's level).
    # Route the full event to Panel C, and keep a compact marker in the tree so
    # the hierarchy shows where each agent sits without the heavy memory blob.
    if topic.endswith("/trace") and payload.get("agent"):
        with _lock:
            _agent_events.append(payload)
            if len(_agent_events) > 200:
                del _agent_events[0]
        _broadcast("agent", topic, payload)
        marker = {
            "agent": payload.get("agent"),
            "event": payload.get("event"),
            "decision": payload.get("decision") or {},
            "ts": payload.get("ts"),
        }
        _state[topic] = marker
        _broadcast("uns", topic, marker)
        return

    _state[topic] = payload
    _broadcast("uns", topic, payload)


def _topics() -> dict:
    """Named UNS topic paths for the browser, so panel lookups don't have to
    reconstruct paths (topics live at different ISA-95 levels now)."""
    return {
        "raw": uns.RAW, "features": uns.FEATURES, "anomaly": uns.ANOMALY,
        "model": uns.MODEL, "state": uns.STATE, "edge_status": uns.EDGE_STATUS,
        "plan": uns.PLAN, "kpi_oee": uns.KPI_OEE, "kpi_production": uns.KPI_PRODUCTION,
        "kpi_uptime": uns.KPI_UPTIME, "workorder": uns.WORKORDER, "window": uns.WINDOW,
        "kpi_cost": uns.KPI_COST,
    }


def _llm_info() -> dict:
    """Which LLM backend/model the agents use (surfaced in Panel C)."""
    backend = CONFIG.llm_backend
    if backend == "ollama":
        return {"backend": backend, "label": "Ollama (local)", "model": CONFIG.ollama_model}
    # Brick path: App Lab picks the model, so use the name the brick resolved at
    # runtime (falls back to an explicit LLM_MODEL override, else "default").
    from ..agents.llm import RESOLVED_BRICK_MODEL
    model = CONFIG.llm_model or RESOLVED_BRICK_MODEL or "default"
    return {"backend": backend, "label": "App Lab LLM brick", "model": model}


@app.route("/")
def index():
    return render_template(
        "index.html",
        base=uns.BASE,
        enterprise=uns.ENTERPRISE,
        topics=_topics(),
        llm=_llm_info(),
    )


@app.route("/api/state")
def api_state():
    return jsonify({"uns": _state, "agents": _agent_events[-20:]})


@app.route("/api/force_anomaly", methods=["POST"])
def api_force_anomaly():
    """Panel A demo trigger: ask the simulator to inject a fault now."""
    fault = (request.json or {}).get("fault")
    log.info("force_anomaly requested via UI (fault=%s)", fault or "random")
    _mqtt.publish("agents/dashboard/command", {"cmd": "force_anomaly", "fault": fault, "ts": now_ts()})
    return jsonify({"ok": True, "fault": fault})


def _ei_upload(values: list[list[float]], fs: float, label: str, api_key: str) -> tuple[bool, int, str]:
    """POST one labeled multi-axis sample to the Edge Impulse Ingestion API.

    ``api_key`` is supplied by the browser per request (never stored server-side)
    and used only for this outbound call. ``values`` is a list of [x, y, z] rows.
    """
    body = {
        "protected": {"ver": "v1", "alg": "none", "iat": int(time.time())},
        "signature": "0" * 64,
        "payload": {
            "device_name": "ventuno-conveyor",
            "device_type": "VENTUNO_Q",
            "interval_ms": 1000.0 / fs,
            "sensors": [
                {"name": "accX", "units": "g"},
                {"name": "accY", "units": "g"},
                {"name": "accZ", "units": "g"},
            ],
            "values": values,
        },
    }
    resp = requests.post(
        f"{CONFIG.ei_ingestion_url}/api/training/data",
        headers={
            "x-api-key": api_key,
            "x-label": label,
            "x-file-name": f"{label}.{int(time.time())}.json",
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
        timeout=30,
    )
    return resp.ok, resp.status_code, resp.text[:200]


def _rows_from_window(win: dict) -> list[list[float]]:
    axis = win.get("axis", {})
    x, y, z = axis.get("x", []), axis.get("y", []), axis.get("z", [])
    n = min(len(x), len(y), len(z))
    return [[x[i], y[i], z[i]] for i in range(n)]


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Push the last single raw window to Edge Impulse (ingest only).

    The API key is entered in the UI and sent with the request (browser-held,
    per the project's chosen key handling); it is never stored on the server.
    """
    data = request.json or {}
    api_key = (data.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"ok": False, "error": "Edge Impulse API key required (enter it in the panel)"}), 400
    if not _last_raw.get("axis"):
        return jsonify({"ok": False, "error": "no vibration window captured yet"}), 400

    label = data.get("label", "normal")
    rows = _rows_from_window(_last_raw)
    fs = _last_raw.get("fs_hz", CONFIG.sim_fs)
    try:
        ok, status, text = _ei_upload(rows, fs, label, api_key)
        return jsonify({"ok": ok, "status": status, "label": label, "samples": len(rows), "body": text})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502


@app.route("/api/record", methods=["POST"])
def api_record():
    """Capture ~N seconds of live vibration and upload it as one labeled sample.

    Collects full-resolution raw windows server-side for the requested duration,
    stitches them into a single time series, and pushes to Edge Impulse with the
    UI-selected label and the UI-entered API key.
    """
    data = request.json or {}
    api_key = (data.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"ok": False, "error": "Edge Impulse API key required (enter it in the panel)"}), 400
    label = data.get("label", "normal")
    seconds = max(1, min(30, int(data.get("seconds", 10))))
    # Windows arrive at ~1/s (sim_window/sim_fs); target that many windows.
    per_s = max(1, round(CONFIG.sim_fs / CONFIG.sim_window))
    target = seconds * per_s

    with _lock:
        _record.update(active=True, buf=[], target=target)
    deadline = time.monotonic() + seconds + 5.0
    while time.monotonic() < deadline:
        with _lock:
            done = len(_record["buf"]) >= target
        if done:
            break
        time.sleep(0.1)
    with _lock:
        _record["active"] = False
        frames = list(_record["buf"])

    if not frames:
        return jsonify({"ok": False, "error": "no vibration captured (is the simulator running?)"}), 400

    rows: list[list[float]] = []
    for win in frames:
        rows.extend(_rows_from_window(win))
    fs = frames[0].get("fs_hz", CONFIG.sim_fs)
    try:
        ok, status, text = _ei_upload(rows, fs, label, api_key)
        return jsonify({"ok": ok, "status": status, "label": label,
                        "seconds": seconds, "windows": len(frames), "samples": len(rows), "body": text})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502


@app.route("/events")
def events():
    def stream():
        q: queue.Queue = queue.Queue(maxsize=200)
        with _lock:
            _subscribers.append(q)
        try:
            # Prime with current state so a fresh page isn't empty.
            for topic, payload in list(_state.items()):
                yield f"data: {json.dumps({'kind': 'uns', 'topic': topic, 'payload': payload})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            with _lock:
                if q in _subscribers:
                    _subscribers.remove(q)
    return Response(stream(), mimetype="text/event-stream")


def start_mqtt() -> None:
    _mqtt.connect()
    # One wildcard covers the whole enterprise: every ISA-95 level plus the
    # agent traces now published inside the UNS.
    _mqtt.subscribe(uns.ENTERPRISE_WILDCARD, _on_uns)
    log.info("dashboard subscribed to %s", uns.ENTERPRISE_WILDCARD)


def main() -> None:
    start_mqtt()
    app.run(host="0.0.0.0", port=CONFIG.dashboard_port, threaded=True)


if __name__ == "__main__":
    main()
