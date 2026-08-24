# Part 3 — A2A Agents across the Shop Floor

Everything in [Part 2](../02-reasoning-agent/), plus **two more agents** and the
**Agent2Agent (A2A) protocol** connecting them — three agents reasoning at three
levels of the ISA-95 hierarchy, over the same single conveyor.

**MQTT** = broadcast state/events on the UNS. **A2A** = directed task delegation
between agents.

## What "shop floor" means here

One physical cell, three altitudes of concern. What differentiates the agents is
**scope and authority**, not data source — they all see the same conveyor:

```
acme/barcelona/                     ← corporate agent  (site)   maintenance windows, cost
  packaging/line1/                  ← planning agent   (line)   production plan, OEE, rate
    conveyor01/                     ← maintenance agent (cell)  anomaly triage, work orders
      vibration/raw
      health/anomaly
```

## What this part adds

```
Part 2: maintenance agent triages AND schedules its own window

Part 3: maintenance agent triages → A2A call → corporate agent schedules
        the window and owns the machine state → planning agent re-plans
        production → KPIs (OEE / cost / rate) react
```

The scheduling logic that lived inline in Part 2's maintenance agent is
**extracted** into the corporate agent, reached over A2A. That is the point of
this part: the same demo loop, but responsibilities split across agents that
negotiate.

## Architecture

| Layer | Component | Publishes / role |
|---|---|---|
| L0 Edge | `vibration_simulator` | `vibration/raw` |
| L0 Edge | `edge_inference` | `vibration/features`, `health/anomaly`, `health/state` |
| L1 UNS | Mosquitto (`mqtt_broker` brick) | single source of truth / event bus |
| L2 | `maintenance_agent` (A2A + LLM) | `maintenance/workorder`, delegates via A2A |
| L4 | `corporate_agent` (A2A, mock CMMS) | `maintenance/window`, drives `health/state` |
| L3 | `planning_agent` (A2A + LLM) | `production/plan` |
| KPI | `kpi_service` | `kpi/oee`, `kpi/production`, `kpi/cost`, `kpi/uptime` |
| UX | `dashboard` (Flask :5001) | Panels A, B, C + **D (manufacturing KPIs)** |

## Full UNS topic tree

Each datum is published at the enterprise level that owns it, not all under the
cell. Base cell path: `acme/barcelona/packaging/line1/conveyor01`.

```
acme/                                              (enterprise)
└─ barcelona/                                       (site)
   ├─ maintenance/workorder   {id, cause_hypothesis, severity, decision, rationale}
   ├─ maintenance/window (r)  {id, start, end, status}
   ├─ kpi/cost           (r)  {running_cost, downtime_cost, maintenance_cost, currency}
   ├─ agents/corporate/trace  (corporate/CMMS agent — site level)
   └─ packaging/                                    (area)
      └─ line1/                                      (line)
         ├─ production/plan       {plan_id, actions[], affected_orders[]}
         ├─ kpi/oee         (r)   {availability, performance, quality, oee}
         ├─ kpi/production  (r)   {rate_units_min, units_total, target_rate}
         ├─ kpi/uptime      (r)   {operational_time_s, downtime_s, state}
         ├─ agents/planning/trace (production-planning agent — line level)
         └─ conveyor01/                              (cell)
            ├─ vibration/raw          {ts, fs_hz, axis:{x,y,z}, rpm}
            ├─ vibration/features     {ts, rms, kurtosis, crest, band_energy[]}
            ├─ health/anomaly         {ts, anomaly_score, threshold, verdict, model_ver}
            ├─ health/model     (r)   {backend, version, ready, frequency, input_features, note}
            ├─ health/state     (r)   {state: healthy|anomaly|maintenance|repairing, ts}
            ├─ edge/status    (r,lwt) {online, app_ver}
            └─ agents/maintenance/trace (edge-maintenance agent — cell level)
```

Agent reasoning traces live **inside** the UNS at each agent's level (a compact
marker shows in the tree; the full context/reasoning/memory feeds Panel C). The
dashboard subscribes to the whole enterprise (`acme/#`).

## Current A2A implementation

`python/ventuno/agents/a2a.py` is a dependency-free implementation of just
enough of Google's A2A protocol to be authentic and interoperable:

* `GET /.well-known/agent-card.json` → Agent Card (name, url, skills)
* `POST /` → JSON-RPC 2.0 `message/send` returning a completed Task with a text artifact

It mirrors the spec shape and uses the stdlib `http.server`. See
**[Planned: LangChain + `a2a-sdk`](#planned-langchain--a2a-sdk)** below.

## Run locally (laptop)

Prereqs: **Python 3.10+**, an **MQTT broker** on `localhost:1883`, and **Ollama**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt

cp .env.example .env
#   MODEL_BACKEND=statistical   LLM_BACKEND=ollama   OLLAMA_MODEL=gemma3:4b
#   ANOMALY_PERIOD_S=45

cd python && python main.py
#   dashboard -> http://localhost:5001
```

### Run components individually

```bash
cd python
python -m ventuno.simulator.vibration_simulator
python -m ventuno.inference.edge_inference
python -m ventuno.kpi.kpi_service
python -m ventuno.agents.corporate_agent
python -m ventuno.agents.maintenance_agent
python -m ventuno.agents.planning_agent
python -m ventuno.dashboard.server
```

## Run on the VENTUNO Q (App Lab)

Same three bricks as Part 2 (`mqtt_broker`, `arduino:llm`,
`arduino:vibration_anomaly_detection`):

```bash
cp .env.example .env
#   MQTT_HOST=mqtt_broker   MODEL_BACKEND=brick   LLM_BACKEND=brick
arduino-app-cli app start .
```

## Verify

```bash
mosquitto_sub -h localhost -t 'acme/#' -v

# Fetch an Agent Card
curl http://localhost:8003/.well-known/agent-card.json
```

One anomaly should produce a maintenance trace → an A2A call → a corporate
trace → a planning trace, with `health/state` cycling
`anomaly → maintenance → repairing → healthy` and Panel D's KPIs reacting.

## Planned: LangChain + `a2a-sdk`

This part is the target for two framework upgrades, deliberately deferred until
Parts 1 and 2 are stable on the board:

- **LangChain over the `llm` brick** — subclass `BaseChatModel` wrapping the
  brick's `LargeLanguageModel.chat()`, keeping on-device inference exactly as it
  is. Prefer structured output (`PydanticOutputParser` + `OutputFixingParser`)
  over ReAct tool-calling: the brick exposes no function-calling channel, and
  local 4B models are unreliable in multi-step ReAct loops.
- **`a2a-sdk`** replacing `a2a.py` — typed `AgentCard`/`AgentSkill`, an
  `AgentExecutor` per agent, `DefaultRequestHandler` + `InMemoryTaskStore`,
  served by uvicorn. Unlocks streaming, task state and push notifications, which
  the current `capabilities: {"streaming": false}` forgoes.

Both carry real cost: the SDK is asyncio-native while this codebase is
thread-based (paho callbacks, `threading.Thread` timers, blocking Flask), so
every brick call must be wrapped in `asyncio.to_thread()` or a multi-second
inference stalls all three agents. Check aarch64 wheels for `pydantic-core` and
any pydantic version conflict with `arduino.app_bricks` before committing.

Inspired by [Kudzai Manditereza's A2A for industrial AI systems](https://github.com/kmanditereza/agent-to-agent-a2a-for-industrial-ai-systems),
which pairs LangChain agents with A2A over OPC UA and TimescaleDB.

## Configuration

Part 2's settings plus `A2A_HOST`, `MAINT_PORT`, `PLAN_PORT`, `CORP_PORT`.
See `.env.example`.
