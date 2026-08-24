"""Minimal Agent2Agent (A2A) compatible transport.

Implements just enough of Google's A2A protocol to be authentic and runnable
without extra dependencies:

  * GET  /.well-known/agent-card.json   -> Agent Card (name, url, skills)
  * POST /                              -> JSON-RPC 2.0 ``message/send`` returning
                                           a completed Task with a text artifact

This mirrors the A2A spec shape and can be replaced by the official ``a2a-sdk``
without changing the agents. Uses the stdlib http.server (one thread per agent).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import requests


class A2AServer:
    def __init__(self, name: str, description: str, url: str, skills: list[dict],
                 handler: Callable[[str], str]):
        self.card = {
            "protocolVersion": "0.2.0",
            "name": name,
            "description": description,
            "url": url,
            "version": "0.1.0",
            "capabilities": {"streaming": False},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": skills,
        }
        self._handler = handler
        self._httpd: ThreadingHTTPServer | None = None

    def _make_request_handler(self):
        card = self.card
        handler = self._handler

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence default logging
                pass

            def _send(self, code: int, body: dict):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/") in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
                    self._send(200, card)
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    req = json.loads(raw)
                except ValueError:
                    self._send(400, {"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}})
                    return

                if req.get("method") != "message/send":
                    self._send(200, {"jsonrpc": "2.0", "id": req.get("id"),
                                     "error": {"code": -32601, "message": "method not found"}})
                    return

                parts = req.get("params", {}).get("message", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if p.get("kind", "text") == "text")
                try:
                    result_text = handler(text)
                except Exception as exc:  # surface agent errors as JSON-RPC error
                    self._send(200, {"jsonrpc": "2.0", "id": req.get("id"),
                                     "error": {"code": -32000, "message": str(exc)}})
                    return

                task = {
                    "id": req.get("id", "task"),
                    "status": {"state": "completed"},
                    "artifacts": [{"parts": [{"kind": "text", "text": result_text}]}],
                }
                self._send(200, {"jsonrpc": "2.0", "id": req.get("id"), "result": task})

        return _Handler

    def start(self, host: str, port: int) -> None:
        self._httpd = ThreadingHTTPServer((host, port), self._make_request_handler())
        threading.Thread(target=self._httpd.serve_forever, name=f"a2a-{self.card['name']}", daemon=True).start()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()


def a2a_send(url: str, text: str, timeout: float = 130.0) -> str:
    """A2A client: send a text message, return the text of the first artifact."""
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": text}], "messageId": "m1"}},
    }
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "a2a error"))
    artifacts = data.get("result", {}).get("artifacts", [])
    if artifacts and artifacts[0].get("parts"):
        return artifacts[0]["parts"][0].get("text", "")
    return ""
