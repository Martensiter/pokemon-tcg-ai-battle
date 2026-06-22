"""Optional local control API so an on-device agent can drive the collector.

On the SwitchBot AI Hub, OpenClaw runs on the same box as the collector. This
exposes a tiny, stdlib-only HTTP API OpenClaw can call as a tool — so from a chat
app while you're out you can ask it the collection status or trigger a pass,
without giving the agent raw shell access.

Security: binds to ``127.0.0.1`` by default (reachable by other local containers
like OpenClaw, not the public internet). A bearer token (``COLLECTOR_API_TOKEN``)
guards the non-health routes; binding to a non-loopback host without a token is
refused.

Routes:
  * ``GET  /health``  -> ``{"ok": true}``                  (no auth; liveness)
  * ``GET  /status``  -> manifest counts + last-pass summary
  * ``POST /collect`` -> run ONE discovery+collection pass; 409 if one is running
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from .collector import Collector
from .config import CollectorConfig
from .logutil import get_logger, log_kv


class ControlServer:
    """Wraps a :class:`Collector` and serializes on-demand passes under a lock."""

    def __init__(self, config: CollectorConfig, collector: Collector | None = None,
                 token: str | None = None, logger: Any | None = None):
        self.cfg = config
        self.log = logger or get_logger("collector", config.state_dir / "collector.log")
        self.collector = collector or Collector(config, logger=self.log)
        self.token = token or config.api_token or None
        self._lock = threading.Lock()
        self.pass_in_progress = False

    def status(self) -> dict[str, Any]:
        m = self.collector.manifest
        return {
            "competition": self.cfg.competition,
            "sink": self.cfg.sink,
            "seen": m.seen_count,
            "counts": m.counts(),
            "pass_in_progress": self.pass_in_progress,
            "has_credentials": self.cfg.has_credentials(),
        }

    def collect_once(self) -> tuple[int, dict[str, Any]]:
        """Run one pass. Returns (http_status, body). 409 if already running."""
        if not self._lock.acquire(blocking=False):
            return 409, {"error": "busy", "pass_in_progress": True}
        try:
            self.pass_in_progress = True
            stats = self.collector.run_once()
            return 200, {"ok": True, **stats.as_kv(),
                         "seen_total": self.collector.manifest.seen_count}
        finally:
            self.pass_in_progress = False
            self._lock.release()


def _make_handler(ctrl: ControlServer):
    token = ctrl.token

    class Handler(BaseHTTPRequestHandler):
        server_version = "ptcg-collector/0.1"

        def _send(self, code: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authed(self) -> bool:
            if not token:
                return True
            hdr = self.headers.get("Authorization", "")
            if hdr.startswith("Bearer "):
                hdr = hdr[len("Bearer "):]
            return hdr == token or self.headers.get("X-Auth-Token", "") == token

        def log_message(self, fmt: str, *args: Any) -> None:  # route through our logger
            log_kv(ctrl.log, "api_request", line=(fmt % args))

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True})
                return
            if not self._authed():
                self._send(401, {"error": "unauthorized"})
                return
            if self.path == "/status":
                self._send(200, ctrl.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authed():
                self._send(401, {"error": "unauthorized"})
                return
            if self.path == "/collect":
                code, body = ctrl.collect_once()
                self._send(code, body)
            else:
                self._send(404, {"error": "not found"})

    return Handler


def serve(config: CollectorConfig, host: str | None = None, port: int | None = None,
          token: str | None = None, collector: Collector | None = None,
          logger: Any | None = None) -> None:
    """Start the blocking control server (Ctrl-C / SIGTERM to stop)."""
    log = logger or get_logger("collector", config.state_dir / "collector.log")
    host = host or config.api_host
    port = port if port is not None else config.api_port
    ctrl = ControlServer(config, collector=collector, token=token, logger=log)

    if host not in ("127.0.0.1", "localhost", "::1") and not ctrl.token:
        raise SystemExit("refusing to bind a non-loopback host without "
                         "COLLECTOR_API_TOKEN set (set a token or use 127.0.0.1)")

    httpd = ThreadingHTTPServer((host, port), _make_handler(ctrl))
    log_kv(log, "api_serving", host=host, port=port, auth=bool(ctrl.token))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        log_kv(log, "api_stopped")
