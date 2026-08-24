"""Supervisor entry point (App Lab app).

Starts every component in one process (App Lab runs one app at a time):
  simulator -> inference -> KPI service -> corporate/maintenance/planning agents
  -> dashboard (foreground, port 5001).

On the VENTUNO Q, Mosquitto and Ollama run as system services; this app assumes
a broker at MQTT_HOST:MQTT_PORT and (if LLM_BACKEND=ollama) an Ollama server.
"""
from __future__ import annotations

import time

from ventuno.config import CONFIG
from ventuno.logbus import get_logger
from ventuno.simulator.vibration_simulator import VibrationSimulator
from ventuno.inference.edge_inference import EdgeInference
from ventuno.kpi.kpi_service import KpiService
from ventuno.agents.corporate_agent import CorporateAgent
from ventuno.agents.maintenance_agent import MaintenanceAgent
from ventuno.agents.planning_agent import PlanningAgent
from ventuno.dashboard import server as dashboard

log = get_logger("main")


def main() -> None:
    print(f"[main] UNS base: {CONFIG.uns_base}  broker: {CONFIG.mqtt_host}:{CONFIG.mqtt_port}")
    print(f"[main] model={CONFIG.model_backend}  llm={CONFIG.llm_backend}")
    log.info("UNS base=%s broker=%s:%s model=%s llm=%s log_level=%s",
             CONFIG.uns_base, CONFIG.mqtt_host, CONFIG.mqtt_port,
             CONFIG.model_backend, CONFIG.llm_backend, CONFIG.log_level)

    # Agents first (A2A servers up before events flow), then producers.
    corporate = CorporateAgent(); corporate.start()
    maintenance = MaintenanceAgent(); maintenance.start()
    planning = PlanningAgent(); planning.start()

    inference = EdgeInference(); inference.start()

    kpi = KpiService(); kpi.start()
    import threading
    threading.Thread(target=kpi.run, name="kpi", daemon=True).start()

    simulator = VibrationSimulator(); simulator.start()

    time.sleep(0.5)
    dashboard.start_mqtt()
    print(f"[main] dashboard -> http://0.0.0.0:{CONFIG.dashboard_port}")
    log.info("dashboard -> http://0.0.0.0:%s", CONFIG.dashboard_port)
    dashboard.app.run(host="0.0.0.0", port=CONFIG.dashboard_port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
