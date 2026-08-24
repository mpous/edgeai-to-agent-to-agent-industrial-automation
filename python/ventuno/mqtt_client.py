"""Thin wrapper around paho-mqtt (v1.x API) with JSON helpers and a Last-Will.

Pinned to paho-mqtt 1.6.x for stable v1 callback signatures across platforms.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Callable

import paho.mqtt.client as mqtt

from .config import CONFIG
from .logbus import get_logger

log = get_logger("mqtt")


def _default_gateway_ip() -> str | None:
    """Return the container's default-gateway IP by parsing ``/proc/net/route``.

    Under App Lab the app runs in a bridged container, so ``localhost`` is the
    container itself — the board's Mosquitto is not reachable there. On a Docker
    bridge network the default gateway *is* the host (the board), so this is the
    address where the broker actually lives. Returns None off-Linux or if the
    route can't be read (e.g. host networking, where localhost already works).
    """
    try:
        with open("/proc/net/route", encoding="ascii") as f:
            next(f)  # header
            for line in f:
                fields = line.strip().split()
                # Destination 0.0.0.0 with the RTF_GATEWAY flag (0x2) set.
                if fields[1] == "00000000" and int(fields[3], 16) & 0x2:
                    gw = int(fields[2], 16)  # little-endian hex
                    return socket.inet_ntoa(struct.pack("<L", gw))
    except (OSError, StopIteration, IndexError, ValueError):
        return None
    return None


def _candidate_hosts() -> list[str]:
    """Broker hosts to try, in order.

    1. The configured host (``MQTT_HOST``) — wins immediately on a laptop where
       it's ``localhost``.
    2. ``mqtt_broker`` — the compose service name of the bundled Mosquitto custom
       brick. Under App Lab the app runs on the same Docker network as the brick,
       so this hostname resolves to the broker container. Tried automatically so
       the demo works even if ``MQTT_HOST`` wasn't set to ``mqtt_broker`` in
       ``.env`` (off-board this name simply fails to resolve and is skipped).
    3. The container's default gateway and the Docker bridge (172.17.0.1) — the
       fallback for a broker published on the board host (direct/no-App-Lab path).
    """
    hosts = [CONFIG.mqtt_host]
    for extra in ("mqtt_broker", _default_gateway_ip(), "172.17.0.1"):
        if extra and extra not in hosts:
            hosts.append(extra)
    return hosts


class MqttClient:
    def __init__(self, client_id: str, lwt_topic: str | None = None, lwt_payload: dict | None = None):
        self._client_id = client_id
        self._client = mqtt.Client(client_id=client_id, clean_session=True)
        self._handlers: dict[str, Callable[[str, dict], None]] = {}
        self._connected = threading.Event()
        self._host: str | None = None
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        if lwt_topic is not None:
            self._client.will_set(lwt_topic, json.dumps(lwt_payload or {"online": False}), qos=1, retain=True)

    # --- lifecycle ---
    def connect(self, timeout: float = 10.0, retries: int = 30, retry_delay: float = 2.0) -> None:
        """Connect to the broker, retrying if it isn't up yet.

        On the board the app and Mosquitto can start in either order, so a single
        refused connection shouldn't kill the demo. We also try several candidate
        hosts per round (see ``_candidate_hosts``): under App Lab the app is
        containerized and ``localhost`` is the container, not the board, so the
        broker is reached via the default gateway instead.
        """
        port = CONFIG.mqtt_port
        hosts = _candidate_hosts()
        log.info("[%s] connecting; candidate hosts=%s port=%s", self._client_id, hosts, port)
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            for host in hosts:
                try:
                    self._client.connect(host, port, keepalive=30)
                    self._host = host
                    break
                except (ConnectionRefusedError, OSError) as exc:
                    last_exc = exc
                    log.debug("[%s] connect to %s:%s failed: %s", self._client_id, host, port, exc)
            else:
                print(f"[mqtt] broker unavailable on {hosts}:{port} ({last_exc}); "
                      f"retry {attempt}/{retries} in {retry_delay:.0f}s")
                log.warning("[%s] broker unavailable on %s:%s (%s); retry %d/%d in %.0fs",
                            self._client_id, hosts, port, last_exc, attempt, retries, retry_delay)
                time.sleep(retry_delay)
                continue
            break
        else:
            raise ConnectionError(
                f"MQTT broker never came up on any of {hosts}:{port}. Under App "
                f"Lab the broker ships as the 'mqtt_broker' custom brick — make "
                f"sure it's listed in app.yaml and started (arduino-app-cli app "
                f"logs). For the direct path, run a broker on 0.0.0.0:1883 (e.g. "
                f"the eclipse-mosquitto container) and disable any system "
                f"mosquitto that might hold the port."
            ) from last_exc
        self._client.loop_start()
        if not self._connected.wait(timeout):
            raise TimeoutError(f"MQTT connect timeout to {self._host}:{port}")
        log.info("[%s] connected to %s:%s", self._client_id, self._host, port)

    def disconnect(self) -> None:
        log.info("[%s] disconnecting", self._client_id)
        self._client.loop_stop()
        self._client.disconnect()

    # --- pub/sub ---
    def publish(self, topic: str, payload: dict, retain: bool = False, qos: int = 0) -> None:
        data = json.dumps(payload, default=str)
        log.debug("[%s] publish %s (%d bytes, retain=%s, qos=%d)",
                  self._client_id, topic, len(data), retain, qos)
        info = self._client.publish(topic, data, qos=qos, retain=retain)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning("[%s] publish to %s failed rc=%s", self._client_id, topic, info.rc)

    def subscribe(self, topic: str, handler: Callable[[str, dict], None], qos: int = 0) -> None:
        self._handlers[topic] = handler
        log.info("[%s] subscribe %s", self._client_id, topic)
        if self._connected.is_set():
            self._client.subscribe(topic, qos=qos)

    # --- callbacks ---
    def _on_connect(self, client, userdata, flags, rc):
        self._connected.set()
        log.info("[%s] on_connect rc=%s; (re)subscribing to %d topic(s)",
                 self._client_id, rc, len(self._handlers))
        for topic in self._handlers:
            client.subscribe(topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            log.debug("[%s] recv %s (%d bytes)", self._client_id, msg.topic, len(msg.payload))
        except (ValueError, UnicodeDecodeError) as exc:
            log.warning("[%s] recv %s: JSON decode failed (%d bytes): %s",
                        self._client_id, msg.topic, len(msg.payload), exc)
            payload = {"_raw": msg.payload.decode("utf-8", errors="replace")}
        for pattern, handler in self._handlers.items():
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    handler(msg.topic, payload)
                except Exception as exc:  # keep the loop alive on handler errors
                    log.exception("[%s] handler error on %s: %s", self._client_id, msg.topic, exc)


def now_ts() -> float:
    return time.time()
