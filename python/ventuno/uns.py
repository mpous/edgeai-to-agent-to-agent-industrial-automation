"""Unified Namespace topic contract (single source of truth).

Plain hierarchical **ISA-95** topics. Each datum is published at the level of the
enterprise that actually owns it, rather than dumping everything under the cell:

    enterprise / site / area / line / cell
    acme       / barcelona / packaging / line1 / conveyor01

* **Cell** (conveyor01)  - raw vibration, features, anomaly score, model info,
  machine state, edge status. These belong to the physical asset.
* **Line** (line1)       - production plan + the line's OEE / production-rate /
  uptime KPIs. A line is planned and measured as a unit.
* **Site** (barcelona)   - maintenance work orders + windows and the site-level
  cost KPI. Maintenance and cost are business functions above a single machine.

Agent reasoning traces live **inside the UNS** at the level each agent operates
on (edge maintenance -> cell, production planning -> line, corporate/CMMS ->
site), so the tree shows where each agent sits in the enterprise. Traces are
published under an ``agents/<name>/trace`` leaf at that level.
"""
from __future__ import annotations

from .config import CONFIG

BASE = CONFIG.uns_base

# --- ISA-95 levels (derived from the configured base path) ---
_parts = BASE.split("/")
ENTERPRISE = _parts[0]
SITE = "/".join(_parts[:2]) if len(_parts) >= 2 else BASE
AREA = "/".join(_parts[:3]) if len(_parts) >= 3 else SITE
LINE = "/".join(_parts[:4]) if len(_parts) >= 4 else AREA
CELL = BASE

# --- Cell level: the physical asset (conveyor01) ---
RAW = f"{CELL}/vibration/raw"                 # {ts, fs_hz, axis:{x[],y[],z[]}, rpm}
FEATURES = f"{CELL}/vibration/features"       # {ts, rms, kurtosis, crest, band_energy[]}
ANOMALY = f"{CELL}/health/anomaly"            # {ts, anomaly_score, threshold, verdict, model_ver}
MODEL = f"{CELL}/health/model"                # retained {backend, version, ready, frequency, input_features, note}
STATE = f"{CELL}/health/state"                # retained {state, ts}
EDGE_STATUS = f"{CELL}/edge/status"           # retained + LWT {online, app_ver}
VISION = f"{CELL}/vision/inspection"          # {ts, defect, confidence, bbox[]}

# --- Line level: how the line is planned and measured ---
PLAN = f"{LINE}/production/plan"              # {plan_id, actions[], affected_orders[], ts}
KPI_OEE = f"{LINE}/kpi/oee"                   # retained {availability, performance, quality, oee, ts}
KPI_PRODUCTION = f"{LINE}/kpi/production"     # retained {rate_units_min, units_total, target_rate, ts}
KPI_UPTIME = f"{LINE}/kpi/uptime"             # retained {operational_time_s, downtime_s, state, ts}

# --- Site level: business functions above a single machine ---
WORKORDER = f"{SITE}/maintenance/workorder"   # {id, cause_hypothesis, severity, decision, rationale, ts}
WINDOW = f"{SITE}/maintenance/window"         # retained {id, start, end, status}
KPI_COST = f"{SITE}/kpi/cost"                 # retained {running_cost, downtime_cost, maintenance_cost, currency, ts}

# --- Agent reasoning traces (inside the UNS, at each agent's level) ---
_AGENT_LEVEL = {
    "maintenance": CELL,   # edge maintenance triage sits on the asset
    "planning": LINE,      # production planning sits on the line
    "corporate": SITE,     # corporate/CMMS sits at the site
}


def agent_trace(name: str) -> str:
    level = _AGENT_LEVEL.get(name, CELL)
    return f"{level}/agents/{name}/trace"


# Dashboard subscribes to the whole enterprise so it sees every level.
ENTERPRISE_WILDCARD = f"{ENTERPRISE}/#"
UNS_WILDCARD = ENTERPRISE_WILDCARD  # back-compat alias

# States used on STATE topic
STATE_HEALTHY = "healthy"
STATE_ANOMALY = "anomaly"
STATE_MAINTENANCE = "maintenance"
STATE_REPAIRING = "repairing"
