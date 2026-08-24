# MQTT Broker (Mosquitto) — Custom Brick

An on-board **Eclipse Mosquitto** MQTT broker, packaged as an App Lab **Custom
Brick**. It hosts the **Unified Namespace (UNS)** — the single MQTT topic tree
that every service in this App publishes and subscribes to.

Under App Lab the orchestrator starts this brick as a container on the App's
virtual Docker network, alongside the main app container. The app's Python does
**not** run inside the broker container; it reaches the broker over the Compose
network using the service name as the hostname:

```
mqtt_broker:1883
```

## Files

| File | Role |
|---|---|
| `brick_config.yaml` | Brick metadata (`id`, name, category) and the `BIND_ADDRESS` variable. |
| `brick_compose.yaml` | The Mosquitto container: image, port publishing, volume mount, healthcheck. |
| `mosquitto.conf` | Broker config — listener, anonymous access, persistence. |
| `__init__.py` | Companion Python API (runs in the *app* container): `BROKER_HOST`, `BROKER_PORT`, and `wait_until_ready()`. |

## How the app connects

Set the broker hostname in `.env` on the board:

```bash
MQTT_HOST=mqtt_broker
```

The app's MQTT client also tries `mqtt_broker` automatically as a candidate
host, so the demo still connects even if `MQTT_HOST` is left at its default. Off
the board (laptop, no App Lab) the `mqtt_broker` name simply fails to resolve
and the client falls back to `localhost`.

Optionally block until the broker is accepting connections:

```python
from bricks.mqtt_broker import wait_until_ready
wait_until_ready(timeout=30)   # True once mqtt_broker:1883 is reachable
```

## Ports & external access

Port `1883` is also published to the **board host** (`BIND_ADDRESS`, default
`0.0.0.0`), so you can watch the UNS from the board or any LAN client:

```bash
mosquitto_sub -h localhost -t 'acme/#' -v
```

> **Port clash:** if a **system Mosquitto** is already enabled on the board it
> will hold port 1883 and clash with this brick. Disable it first:
> ```bash
> sudo systemctl disable --now mosquitto
> ```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BIND_ADDRESS` | `0.0.0.0` | Host interface the broker's `1883` port is published on. `0.0.0.0` = all interfaces (board + LAN); set to `127.0.0.1` to keep it board-local. |

**`mosquitto.conf`** opens the listener on `0.0.0.0:1883` with
`allow_anonymous true` and on-disk persistence at `/mosquitto/data/`. Anonymous
access is fine for an on-board demo; add a `password_file` / ACLs if the board
sits on an untrusted network.

## Health

`brick_compose.yaml` defines a healthcheck that subscribes to `$SYS/#` and exits
non-zero if the broker isn't answering, so App Lab can tell when the UNS is
ready before the rest of the App relies on it.

## Enabling the brick

Declared in the App's `app.yaml`:

```yaml
bricks:
  - mqtt_broker: {}
```
