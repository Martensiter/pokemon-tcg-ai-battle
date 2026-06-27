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
  * ``GET  /status``  -> manifest counts + ``pass_in_progress`` + last-pass result
  * ``POST /collect`` -> start ONE pass in the background; returns ``202
        {"status": "started"}`` immediately (so a chat tool call never blocks),
        or ``409`` if a pass is already running. Poll ``/status`` for progress.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from .collector import Collector
from .config import CollectorConfig
from .logutil import get_logger, log_kv


class ControlServer:
    """Wraps a :class:`Collector`; runs passes in a background thread (one at a time)."""

    def __init__(self, config: CollectorConfig, collector: Collector | None = None,
                 token: str | None = None, logger: Any | None = None):
        self.cfg = config
        self.log = logger or get_logger("collector", config.state_dir / "collector.log")
        self.collector = collector or Collector(config, logger=self.log)
        self.token = token or config.api_token or None
        self._guard = threading.Lock()      # protects pass_in_progress
        self.pass_in_progress = False
        self._last_result: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        m = self.collector.manifest
        now = int(time.time())
        out: dict[str, Any] = {
            "now": now,
            "competition": self.cfg.competition,
            "sink": self.cfg.sink,
            "seen": m.seen_count,
            "counts": m.counts(),
            "pass_in_progress": self.pass_in_progress,
            "has_credentials": self.cfg.has_credentials(),
        }
        # Surface the on-disk last-success heartbeat. The collecting passes may be
        # SEPARATE processes (cron `--once`) from this server, so the in-memory
        # manifest is not the source of truth for "when did a pass last succeed";
        # status.json (written at the end of every pass) is. `age_seconds` lets an
        # external monitor alert on staleness with a single field.
        disk = self._read_status_file()
        if disk is not None:
            ts = disk.get("ts")
            if isinstance(ts, (int, float)):
                out["last_success_ts"] = int(ts)
                out["age_seconds"] = now - int(ts)
            if "last_pass" in disk:
                out["last_pass"] = disk["last_pass"]
        if self._last_result is not None:
            out["last_pass"] = self._last_result
        return out

    def _read_status_file(self) -> dict[str, Any] | None:
        """Best-effort read of the on-disk status.json (None if absent/corrupt)."""
        try:
            p = Path(self.cfg.state_dir) / "status.json"
            if not p.exists():
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001  (status must never break on a torn file)
            return None

    def _run_pass(self) -> dict[str, Any]:
        """Run one pass synchronously and record the result. Never raises."""
        try:
            stats = self.collector.run_once()
            result = {"ok": True, **stats.as_kv(),
                      "seen_total": self.collector.manifest.seen_count}
        except Exception as e:  # noqa: BLE001  (keep the server alive)
            log_kv(self.log, "collect_error", level=40, err=f"{type(e).__name__}: {e}")
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            with self._guard:
                self.pass_in_progress = False
        self._last_result = result
        return result

    def collect_once(self) -> dict[str, Any]:
        """Run one pass synchronously (programmatic / tests). Returns the result."""
        with self._guard:
            if self.pass_in_progress:
                return {"ok": False, "error": "busy", "pass_in_progress": True}
            self.pass_in_progress = True
        return self._run_pass()

    def start_collect(self) -> tuple[int, dict[str, Any]]:
        """Kick off a pass in the background. Returns (202, started) or (409, busy)."""
        with self._guard:
            if self.pass_in_progress:
                return 409, {"status": "busy", "pass_in_progress": True}
            self.pass_in_progress = True
        threading.Thread(target=self._run_pass, name="collector-pass", daemon=True).start()
        return 202, {"status": "started"}


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
                code, body = ctrl.start_collect()
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
