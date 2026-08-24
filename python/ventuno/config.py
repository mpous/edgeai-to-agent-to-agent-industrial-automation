"""Central configuration, loaded from environment variables with sane defaults.

Defaults target the VENTUNO Q under App Lab: the ``llm`` brick (Gemma/Qwen/Qwen3)
and the ``vibration_anomaly_detection`` brick serving the deployed Edge Impulse
model. For the direct (non-App-Lab) path, override with Ollama and an ``.eim``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv() -> None:
    """Minimal .env loader (no python-dotenv dependency). Existing env wins.

    Looks in the current working directory first, then the app root (the parent
    of ``python/``) so it works whether launched from the repo root, from
    ``python/``, or by App Lab.
    """
    here = os.path.dirname(os.path.abspath(__file__))  # python/ventuno
    app_root = os.path.dirname(os.path.dirname(here))   # repo root (parent of python/)
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(app_root, ".env"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.split("#", 1)[0].strip()
            os.environ.setdefault(key, value)


_load_dotenv()


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- MQTT / Unified Namespace ---
    mqtt_host: str = field(default_factory=lambda: _get("MQTT_HOST", "localhost"))
    mqtt_port: int = field(default_factory=lambda: _get_int("MQTT_PORT", 1883))
    # ISA-95 base path: enterprise/site/area/line/cell
    uns_base: str = field(default_factory=lambda: _get("UNS_BASE", "acme/barcelona/packaging/line1/conveyor01"))

    # --- Simulator ---
    sim_fs: int = field(default_factory=lambda: _get_int("SIM_FS", 2000))          # sampling rate (Hz)
    sim_window: int = field(default_factory=lambda: _get_int("SIM_WINDOW", 2000))   # samples per published window
    sim_rpm: float = field(default_factory=lambda: _get_float("SIM_RPM", 1450.0))   # driven shaft rpm
    anomaly_period_s: float = field(default_factory=lambda: _get_float("ANOMALY_PERIOD_S", 300.0))

    # --- Inference / anomaly model ---
    model_backend: str = field(default_factory=lambda: _get("MODEL_BACKEND", "brick"))  # brick | eim | statistical
    eim_path: str = field(default_factory=lambda: _get("EIM_PATH", "models/conveyor-anomaly.eim"))
    anomaly_threshold: float = field(default_factory=lambda: _get_float("ANOMALY_THRESHOLD", 0.5))
    anomaly_persist: int = field(default_factory=lambda: _get_int("ANOMALY_PERSIST", 3))  # consecutive windows

    # --- LLM (agents) ---
    llm_backend: str = field(default_factory=lambda: _get("LLM_BACKEND", "brick"))  # brick | ollama
    ollama_host: str = field(default_factory=lambda: _get("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "gemma3:4b"))
    # Optional label override for the brick path. Empty = auto-detect the model
    # the App Lab llm brick resolves at runtime (shown on the dashboard).
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", ""))

    # --- Agent A2A endpoints ---
    maint_port: int = field(default_factory=lambda: _get_int("MAINT_PORT", 8001))
    plan_port: int = field(default_factory=lambda: _get_int("PLAN_PORT", 8002))
    corp_port: int = field(default_factory=lambda: _get_int("CORP_PORT", 8003))
    a2a_host: str = field(default_factory=lambda: _get("A2A_HOST", "localhost"))
    maint_duration_s: float = field(default_factory=lambda: _get_float("MAINT_DURATION_S", 60.0))
    # How often each agent posts a "nominal / all clear" heartbeat trace when the
    # line is healthy (0 disables). Keeps Panel C alive during quiet periods.
    heartbeat_s: float = field(default_factory=lambda: _get_float("HEARTBEAT_S", 60.0))

    # --- Dashboard ---
    dashboard_port: int = field(default_factory=lambda: _get_int("DASHBOARD_PORT", 5001))

    # --- Logging ---
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))  # DEBUG for per-message pub/sub

    # --- Edge Impulse ingestion (Panel A re-training, ingest only) ---
    ei_api_key: str = field(default_factory=lambda: _get("EI_API_KEY", ""))
    ei_ingestion_url: str = field(default_factory=lambda: _get("EI_INGESTION_URL", "https://ingestion.edgeimpulse.com"))

    @property
    def corp_url(self) -> str:
        return f"http://{self.a2a_host}:{self.corp_port}"

    @property
    def maint_url(self) -> str:
        return f"http://{self.a2a_host}:{self.maint_port}"

    @property
    def plan_url(self) -> str:
        return f"http://{self.a2a_host}:{self.plan_port}"


CONFIG = Config()
