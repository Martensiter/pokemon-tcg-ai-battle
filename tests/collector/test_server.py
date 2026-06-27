"""Control API tests: status / collect / auth, over real loopback HTTP.

Uses the package's offline client (no network, no engine, no torch) and a
throwaway data/state dir, so the whole thing runs in CI.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer

from collector.collector import Collector
from collector.config import CollectorConfig
from collector.manifest import Manifest
from collector.selftest import _OfflineClient
from collector.server import ControlServer, _make_handler
from collector.sink import LocalSink


def _ctrl(tmp_path, token=None):
    cfg = CollectorConfig(data_dir=tmp_path / "data", state_dir=tmp_path / "state",
                          rps=0.0, chunk_size=100, sink="local", api_token=token or "")
    col = Collector(cfg, client=_OfflineClient(n_subs=2, n_eps=2),
                    sink=LocalSink(cfg.data_dir),
                    manifest=Manifest(cfg.state_dir / "manifest.jsonl"))
    return ControlServer(cfg, collector=col, token=token)


def _serve(ctrl):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(ctrl))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _get(port, path, token=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())


def _post(port, path, token=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"", method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


def _wait_idle(port, token, timeout=10.0):
    """Poll /status until a background pass finishes (or time out)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, st = _get(port, "/status", token=token)
        if not st["pass_in_progress"] and st.get("last_pass") is not None:
            return st
        time.sleep(0.05)
    raise AssertionError("pass did not finish in time")


def test_control_server_status_and_collect_units(tmp_path):
    ctrl = _ctrl(tmp_path)
    st = ctrl.status()
    assert st["seen"] == 0 and st["pass_in_progress"] is False
    # synchronous helper (used by programmatic callers / this unit test)
    body = ctrl.collect_once()
    assert body["ok"] is True
    assert body["converted_rows"] == 2 * 2 * 4
    assert ctrl.status()["seen"] == 4


def test_status_exposes_heartbeat_after_pass(tmp_path):
    """/status surfaces `now` plus the on-disk last-success ts + age_seconds, so an
    external monitor can alert on staleness without parsing logs."""
    ctrl = _ctrl(tmp_path)
    st0 = ctrl.status()
    assert "now" in st0
    assert "last_success_ts" not in st0          # no pass has run yet
    ctrl.collect_once()                           # writes state/status.json with ts
    st1 = ctrl.status()
    assert st1["last_success_ts"] <= st1["now"]
    assert st1["age_seconds"] >= 0
    assert st1["age_seconds"] == st1["now"] - st1["last_success_ts"]


def test_start_collect_is_async(tmp_path):
    ctrl = _ctrl(tmp_path)
    code, body = ctrl.start_collect()
    assert code == 202 and body["status"] == "started"
    # eventually completes and records the result
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and ctrl.pass_in_progress:
        time.sleep(0.02)
    assert ctrl._last_result and ctrl._last_result["converted_rows"] == 16
    # a second concurrent start while busy would 409 (simulate by holding the flag)
    with ctrl._guard:
        ctrl.pass_in_progress = True
    try:
        code2, body2 = ctrl.start_collect()
        assert code2 == 409 and body2["status"] == "busy"
    finally:
        with ctrl._guard:
            ctrl.pass_in_progress = False


def test_health_needs_no_auth(tmp_path):
    httpd, port = _serve(_ctrl(tmp_path, token="secret"))
    try:
        code, body = _get(port, "/health")
        assert code == 200 and body["ok"] is True
    finally:
        httpd.shutdown()


def test_status_requires_token(tmp_path):
    httpd, port = _serve(_ctrl(tmp_path, token="secret"))
    try:
        # no token -> 401
        try:
            _get(port, "/status")
            assert False, "expected 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        # with token -> 200
        code, body = _get(port, "/status", token="secret")
        assert code == 200 and "seen" in body
    finally:
        httpd.shutdown()


def test_collect_over_http(tmp_path):
    httpd, port = _serve(_ctrl(tmp_path, token="secret"))
    try:
        code, body = _post(port, "/collect", token="secret")
        assert code == 202 and body["status"] == "started"   # async: returns immediately
        st = _wait_idle(port, "secret")                       # poll until done
        assert st["seen"] == 4
        assert st["last_pass"]["converted_rows"] == 16
    finally:
        httpd.shutdown()


def test_unknown_route_404(tmp_path):
    httpd, port = _serve(_ctrl(tmp_path))
    try:
        try:
            _get(port, "/nope")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
