"""Pluggable LLM backend for the agents.

* BrickLLM  - uses the Arduino App Lab ``llm`` brick (LargeLanguageModel) to run
  a local model (Gemma/Qwen/Qwen3) on the VENTUNO Q. This is the default: the
  model is selected and downloaded in App Lab; the brick's backing service is
  started by App Lab. Real on-device inference, never mocked.
* OllamaLLM - talks to a local Ollama server running Gemma (or Qwen) on the
  VENTUNO Q. Uses the /api/chat endpoint with format=json. Alternative to the
  brick for the direct (non-App-Lab) path.

All expose: complete(system, user, want_json=True) -> (text, reasoning).
"""
from __future__ import annotations

import json

import requests

from ..config import CONFIG

# Set once the App Lab llm brick resolves its model (chosen in App Lab, not by
# us), so the dashboard can label Panel C with the real model instead of the
# Ollama default. None until a BrickLLM is built.
RESOLVED_BRICK_MODEL: str | None = None


def _extract_json(text: str) -> str:
    """Best-effort: pull the first {...} object out of a chatty LLM reply.

    Local models sometimes wrap JSON in prose or a ```json fence; the agents need
    a bare object to parse.
    """
    if not text:
        return "{}"
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return t[start:end + 1]
    return t


class OllamaLLM:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.last_prompt = ""  # full prompt from the most recent complete() (Panel C)

    def complete(self, system: str, user: str, want_json: bool = True) -> tuple[str, str]:
        self.last_prompt = f"{system}\n\n{user}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if want_json:
            body["format"] = "json"
        resp = requests.post(f"{self.host}/api/chat", json=body, timeout=120)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        reasoning = ""
        try:
            reasoning = json.loads(content).get("rationale", "")
        except ValueError:
            reasoning = content[:400]
        return content, reasoning


class BrickLLM:
    """Local LLM via the App Lab ``llm`` brick (LargeLanguageModel).

    Board-only. The brick wraps LangChain over a locally-hosted model (default
    depends on the board; Gemma/Qwen selectable in App Lab). We combine the
    system + user text into one prompt because ``complete()`` is stateless per
    call and the agents already carry their own rolling memory.
    """

    def __init__(self, model: str | None = None):
        from arduino.app_bricks.llm import LargeLanguageModel  # board-only import

        self._llm = LargeLanguageModel()
        self.last_prompt = ""  # full prompt from the most recent complete() (Panel C)
        # The brick picks its model from App Lab config, ignoring anything we
        # pass. Read the resolved name back so the dashboard shows the truth.
        resolved = CONFIG.llm_model or self._resolve_model_name() or model or "default"
        self.model = resolved
        global RESOLVED_BRICK_MODEL
        RESOLVED_BRICK_MODEL = resolved
        self.name = f"brick-llm:{resolved}"

    def _resolve_model_name(self) -> str | None:
        for attr in ("model", "model_name", "_model", "_model_name"):
            v = getattr(self._llm, attr, None)
            if isinstance(v, str) and v:
                return v
        info = getattr(self._llm, "get_model_info", None)
        if callable(info):
            try:
                data = info()
            except Exception:
                return None
            for key in ("model", "name"):
                v = getattr(data, key, None) or (data.get(key) if isinstance(data, dict) else None)
                if isinstance(v, str) and v:
                    return v
        return None

    def complete(self, system: str, user: str, want_json: bool = True) -> tuple[str, str]:
        prompt = f"{system}\n\n{user}"
        if want_json:
            prompt += "\n\nRespond with ONLY a single JSON object, no prose."
        self.last_prompt = prompt
        text = self._llm.chat(prompt)
        content = _extract_json(text) if want_json else text
        try:
            reasoning = json.loads(content).get("rationale", "")
        except ValueError:
            reasoning = (text or "")[:400]
        return content, reasoning


def build_llm():
    # BrickLLM is the default: real on-device inference via the App Lab llm
    # brick (Gemma/Qwen/Qwen3). Ollama is the alternative for the direct path.
    if CONFIG.llm_backend == "ollama":
        return OllamaLLM(CONFIG.ollama_host, CONFIG.ollama_model)
    return BrickLLM(CONFIG.ollama_model)
