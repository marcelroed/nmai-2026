"""
API call log viewer with persistent storage (local or GCS).

Captures all LoggedHTTPClient calls per /solve request, saves them as JSON,
and serves an HTML dashboard at /logs.

Storage backend:
  - Set GCS_LOG_BUCKET env var → saves to GCS (for Cloud Run)
  - Otherwise → saves to data/api_logs/ (for local dev)
"""

from __future__ import annotations

import dataclasses
import html as html_mod
import json
import logging
import os
import tempfile
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------

_GCS_BUCKET = os.environ.get("GCS_LOG_BUCKET", "")
_LOCAL_DIR = Path("data/api_logs")


def _save_json(name: str, data: dict) -> None:
    blob = json.dumps(data, default=str, ensure_ascii=False, indent=2)
    if _GCS_BUCKET:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(_GCS_BUCKET)
        bucket.blob(f"api_logs/{name}").upload_from_string(blob, content_type="application/json")
    else:
        _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        (_LOCAL_DIR / name).write_text(blob)


def _list_logs() -> list[dict]:
    records = []
    if _GCS_BUCKET:
        from google.cloud import storage
        client = storage.Client()
        blobs = client.bucket(_GCS_BUCKET).list_blobs(prefix="api_logs/")
        for blob in blobs:
            if blob.name.endswith(".json"):
                try:
                    records.append(json.loads(blob.download_as_text()))
                except (json.JSONDecodeError, KeyError):
                    continue
    else:
        if _LOCAL_DIR.exists():
            for f in sorted(_LOCAL_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
                try:
                    records.append(json.loads(f.read_text()))
                except (json.JSONDecodeError, KeyError):
                    continue
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records


def _load_log(request_id: str) -> dict | None:
    name = f"{request_id}.json"
    if _GCS_BUCKET:
        from google.cloud import storage
        client = storage.Client()
        blob = client.bucket(_GCS_BUCKET).blob(f"api_logs/{name}")
        if blob.exists():
            return json.loads(blob.download_as_text())
        return None
    else:
        path = _LOCAL_DIR / name
        if path.exists():
            return json.loads(path.read_text())
        return None


# ---------------------------------------------------------------------------
# APICall + LogCollector
# ---------------------------------------------------------------------------


@dataclass
class APICall:
    call_id: str = ""
    request_id: str = ""
    unique_request_id: str = ""
    method: str = ""
    url: str = ""
    endpoint: str = ""
    request_params: Any = None
    request_body: Any = None
    request_headers: dict = field(default_factory=dict)
    request_timestamp: str = ""
    status_code: int | None = None
    status_reason: str = ""
    response_body: Any = None
    response_headers: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
    error_type: str | None = None
    level: str = "INFO"


class LogCollector(logging.Handler):
    """Captures LoggedHTTPClient calls in-memory, then saves to storage."""

    def __init__(self, request_id: str) -> None:
        super().__init__()
        self.request_id = request_id
        self._pending: dict[str, APICall] = {}
        self.calls: list[APICall] = []

    def install(self) -> LogCollector:
        lgr = logging.getLogger("logged_client")
        lgr.addHandler(self)
        lgr.setLevel(logging.DEBUG)
        return self

    def uninstall(self) -> None:
        logging.getLogger("logged_client").removeHandler(self)

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "api_event", None)
        if not event:
            return
        if getattr(record, "api_unique_request_id", "") != self.request_id:
            return
        call_id = getattr(record, "api_call_id", "")

        if event == "request":
            self._pending[call_id] = APICall(
                call_id=call_id,
                request_id=getattr(record, "api_request_id", ""),
                unique_request_id=getattr(record, "api_unique_request_id", ""),
                method=getattr(record, "api_method", ""),
                url=getattr(record, "api_url", ""),
                endpoint=getattr(record, "api_endpoint", ""),
                request_params=getattr(record, "api_request_params", None),
                request_body=getattr(record, "api_request_body", None),
                request_headers=getattr(record, "api_request_headers", {}),
                request_timestamp=self._format_time(record),
            )
        elif event == "response":
            call = self._pending.pop(call_id, APICall(call_id=call_id))
            call.request_id = getattr(record, "api_request_id", call.request_id)
            call.unique_request_id = getattr(
                record, "api_unique_request_id", call.unique_request_id
            )
            call.status_code = getattr(record, "api_status_code", None)
            call.status_reason = getattr(record, "api_status_reason", "")
            call.response_body = getattr(record, "api_response_body", None)
            call.response_headers = getattr(record, "api_response_headers", {})
            call.duration_ms = getattr(record, "api_duration_ms", 0.0)
            call.level = record.levelname
            self.calls.append(call)
        elif event == "error":
            call = self._pending.pop(call_id, APICall(call_id=call_id))
            call.request_id = getattr(record, "api_request_id", call.request_id)
            call.unique_request_id = getattr(
                record, "api_unique_request_id", call.unique_request_id
            )
            call.error = getattr(record, "api_error", "")
            call.error_type = getattr(record, "api_error_type", "")
            call.duration_ms = getattr(record, "api_duration_ms", 0.0)
            call.level = "ERROR"
            self.calls.append(call)

    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%H:%M:%S.%f")[:-3]

    def save_to_disk(self, prompt: str = "", files: list[str] | None = None, request_id: str = "") -> None:
        rid = request_id or str(uuid.uuid4())[:8]
        record = {
            "request_id": rid,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt": prompt,
            "files": files or [],
            "call_count": len(self.calls),
            "total_duration_ms": round(sum(c.duration_ms for c in self.calls), 1),
            "calls": [dataclasses.asdict(c) for c in self.calls],
        }
        _save_json(f"{rid}.json", record)

    def open_in_browser(self, title: str = "API Call Timeline") -> Path:
        html = render_timeline(self.calls, title=title)
        path = Path(tempfile.gettempdir()) / "api_timeline.html"
        path.write_text(html)
        webbrowser.open(f"file://{path}")
        return path


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def _esc(s: Any) -> str:
    return html_mod.escape(str(s)) if s is not None else ""


def _format_json(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return html_mod.escape(obj)
    return html_mod.escape(json.dumps(obj, indent=2, ensure_ascii=False))


def _status_class(code: int | None) -> str:
    if code is None:
        return "error"
    if code < 400:
        return "success"
    if code < 500:
        return "warning"
    return "error"


def _detail_section(label: str, content: str, css_class: str = "") -> str:
    pre_class = f' class="{css_class}"' if css_class else ""
    return f'<div class="detail-section"><div class="detail-label">{label}</div><pre{pre_class}>{content}</pre></div>'


def _icon_for(call: APICall) -> tuple[str, str]:
    if call.error:
        return "&#10007;", "icon-error"
    if call.status_code and call.status_code >= 500:
        return "&#9888;", "icon-error"
    if call.status_code and call.status_code >= 400:
        return "&#9888;", "icon-warning"
    return "&#10003;", "icon-success"


def _render_call(i: int, c: APICall) -> str:
    sc = _status_class(c.status_code)
    data_id = f"data-{i}"
    icon, icon_class = _icon_for(c)

    method_badge = f'<span class="badge method-{c.method.lower()}">{c.method}</span>'
    if c.error:
        status_badge = f'<span class="badge error-badge">{_esc(c.error_type)}</span>'
    elif c.status_code is not None:
        status_badge = f'<span class="badge status-{sc}">{c.status_code} {_esc(c.status_reason)}</span>'
    else:
        status_badge = ""
    params_badge = f'<span class="badge params">{_esc(json.dumps(c.request_params))}</span>' if c.request_params else ""
    dur = f'<span class="duration">{c.duration_ms:.1f}ms</span>'

    sections = []
    if c.request_headers:
        sections.append(_detail_section("Request Headers", _format_json(c.request_headers)))
    if c.request_body is not None:
        sections.append(_detail_section("Request Body", _format_json(c.request_body)))
    if c.response_headers:
        sections.append(_detail_section("Response Headers", _format_json(c.response_headers)))
    if c.response_body is not None:
        sections.append(_detail_section("Response Body", _format_json(c.response_body)))
    if c.error:
        sections.append(_detail_section("Error", _esc(c.error), css_class="error-pre"))

    details_html = f'<div class="call-details" id="{data_id}" style="display:none;">{"".join(sections)}</div>' if sections else ""

    return f"""
        <div class="phase call {sc}" data-status="{sc}">
          <div class="phase-icon {icon_class}">{icon}</div>
          <div class="phase-content">
            <div class="call-header" onclick="toggle('{data_id}')" style="cursor:pointer">
              {method_badge}
              <span class="call-url">{_esc(c.url)}</span>
              {params_badge}
              {status_badge}
              {dur}
              <span class="call-id">{_esc(c.call_id)}</span>
            </div>
            {details_html}
          </div>
        </div>"""


# ---------------------------------------------------------------------------
# Timeline page
# ---------------------------------------------------------------------------

_TIMELINE_CSS = """
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 14px; background: #0d1117; color: #c9d1d9; padding: 24px 40px; max-width: 1200px; margin: 0 auto; }
  code, pre, .call-url, .badge, .call-id, .duration { font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; }
  a { color: #58a6ff; text-decoration: none; } a:hover { text-decoration: underline; }
  .header { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; }
  .header h1 { font-size: 18px; color: #f0f6fc; margin-bottom: 8px; }
  .header-meta { font-size: 13px; color: #8b949e; }
  .stats { display: flex; gap: 16px; margin-bottom: 20px; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; text-align: center; min-width: 100px; }
  .stat-num { font-size: 24px; font-weight: bold; }
  .stat-label { font-size: 11px; color: #8b949e; margin-top: 2px; }
  .stat.green .stat-num { color: #3fb950; } .stat.yellow .stat-num { color: #d29922; }
  .stat.red .stat-num { color: #f85149; } .stat.blue .stat-num { color: #58a6ff; }
  .controls { margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
  .controls button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  .controls button:hover { background: #30363d; } .controls button.active { border-color: #58a6ff; color: #58a6ff; }
  .timeline { position: relative; padding-left: 32px; }
  .timeline::before { content: ''; position: absolute; left: 11px; top: 0; bottom: 0; width: 2px; background: #21262d; }
  .phase { position: relative; margin-bottom: 4px; display: flex; align-items: flex-start; gap: 12px; }
  .phase-icon { width: 24px; min-width: 24px; text-align: center; font-size: 14px; position: relative; z-index: 1; line-height: 1.6; }
  .icon-success { color: #3fb950; } .icon-warning { color: #d29922; } .icon-error { color: #f85149; }
  .phase.call { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 8px 12px; }
  .phase.call:hover { border-color: #30363d; } .phase.call.hidden { display: none !important; }
  .call-header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .call-url { font-size: 13px; font-weight: 600; color: #c9d1d9; }
  .phase.success .call-url { color: #3fb950; } .phase.warning .call-url { color: #d29922; } .phase.error .call-url { color: #f85149; }
  .call-id { font-size: 10px; color: #484f58; margin-left: auto; }
  .duration { font-size: 11px; color: #8b949e; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }
  .badge.method-get { background: #1f6feb33; color: #58a6ff; } .badge.method-post { background: #3fb95033; color: #3fb950; }
  .badge.method-put { background: #d2992233; color: #d29922; } .badge.method-patch { background: #d2992233; color: #d29922; }
  .badge.method-delete { background: #f8514933; color: #f85149; }
  .badge.params { background: #d2992233; color: #d29922; max-width: 300px; overflow: hidden; text-overflow: ellipsis; }
  .badge.status-success { background: #3fb95033; color: #3fb950; } .badge.status-warning { background: #d2992233; color: #d29922; }
  .badge.status-error { background: #f8514933; color: #f85149; } .badge.error-badge { background: #f8514933; color: #f85149; }
  .call-details { margin-top: 10px; border-top: 1px solid #21262d; padding-top: 10px; }
  .detail-section { margin-bottom: 10px; }
  .detail-label { font-size: 11px; color: #8b949e; font-weight: 600; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .detail-section pre { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; font-size: 12px; color: #79c0ff; max-height: 400px; overflow: auto; white-space: pre-wrap; word-break: break-all; }
  .error-pre { color: #f85149 !important; }
"""

_TIMELINE_JS = """
function toggle(id) { var el = document.getElementById(id); if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none'; }
function filterCalls(status, btn) {
  document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.phase.call').forEach(el => {
    if (status === 'all') el.classList.remove('hidden');
    else el.classList.toggle('hidden', el.dataset.status !== status);
  });
}
function expandAll() { document.querySelectorAll('.call-details').forEach(el => el.style.display = 'block'); }
function collapseAll() { document.querySelectorAll('.call-details').forEach(el => el.style.display = 'none'); }
"""


def render_timeline(calls: list[APICall], title: str = "API Call Timeline") -> str:
    successes = [c for c in calls if c.status_code is not None and c.status_code < 400]
    warnings = [c for c in calls if c.status_code is not None and 400 <= c.status_code < 500]
    errors = [c for c in calls if c.error is not None or (c.status_code is not None and c.status_code >= 500)]
    total_duration = sum(c.duration_ms for c in calls)
    phases = [_render_call(i, c) for i, c in enumerate(calls)]

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{_esc(title)}</title><style>{_TIMELINE_CSS}</style></head>
<body>
<div class="header">
  <h1>{_esc(title)}</h1>
  <div class="header-meta"><a href="/logs">&larr; All requests</a> &middot; {len(calls)} API calls &middot; {total_duration:.0f}ms total</div>
</div>
<div class="stats">
  <div class="stat green"><div class="stat-num">{len(successes)}</div><div class="stat-label">Success</div></div>
  <div class="stat yellow"><div class="stat-num">{len(warnings)}</div><div class="stat-label">4xx</div></div>
  <div class="stat red"><div class="stat-num">{len(errors)}</div><div class="stat-label">Error</div></div>
  <div class="stat blue"><div class="stat-num">{total_duration:.0f}</div><div class="stat-label">Total ms</div></div>
</div>
<div class="controls">
  <button class="active" onclick="filterCalls('all', this)">All</button>
  <button onclick="filterCalls('success', this)">Success</button>
  <button onclick="filterCalls('warning', this)">4xx</button>
  <button onclick="filterCalls('error', this)">Errors</button>
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
</div>
<div class="timeline">{''.join(phases)}</div>
<script>{_TIMELINE_JS}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

_INDEX_CSS = """
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 14px; background: #0d1117; color: #c9d1d9; padding: 24px 40px; max-width: 1200px; margin: 0 auto; }
  .header { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; }
  .header h1 { font-size: 18px; color: #f0f6fc; margin-bottom: 8px; }
  .header-meta { font-size: 13px; color: #8b949e; }
  .request-list { display: flex; flex-direction: column; gap: 4px; }
  .request-row { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 12px 16px; cursor: pointer; display: flex; align-items: center; gap: 16px; }
  .request-row:hover { border-color: #30363d; background: #1c2128; }
  .req-time { font-size: 12px; color: #484f58; min-width: 160px; font-family: 'SF Mono', monospace; }
  .req-prompt { color: #f0883e; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .req-badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-family: 'SF Mono', monospace; }
  .req-badge.calls { background: #1f6feb33; color: #58a6ff; }
  .req-badge.duration { background: #21262d; color: #8b949e; }
"""


def render_index() -> str:
    records = _list_logs()
    rows = []
    for r in records:
        rid = r.get("request_id", "?")
        ts = _esc(r.get("timestamp", "?"))
        prompt = _esc(r.get("prompt", "")[:120]) or "<em>no prompt</em>"
        call_count = r.get("call_count", 0)
        dur = r.get("total_duration_ms", 0)
        rows.append(f"""
        <div class="request-row" onclick="window.location='/logs?request={rid}'">
          <span class="req-time">{ts}</span>
          <span class="req-prompt">{prompt}</span>
          <span class="req-badge calls">{call_count} calls</span>
          <span class="req-badge duration">{dur:.0f}ms</span>
        </div>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>API Log Viewer</title><style>{_INDEX_CSS}</style></head>
<body>
<div class="header">
  <h1>API Log Viewer</h1>
  <div class="header-meta">{len(records)} requests logged &middot; storage: {'GCS (' + _GCS_BUCKET + ')' if _GCS_BUCKET else 'local (data/api_logs/)'}</div>
</div>
<div class="request-list">
{''.join(rows) if rows else '<div style="color:#8b949e;padding:20px;">No logs yet. Send a request to /solve to start logging.</div>'}
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Entry point for FastAPI
# ---------------------------------------------------------------------------


def viewer_html(request_id: str | None = None) -> str:
    if request_id:
        record = _load_log(request_id)
        if record:
            calls = [APICall(**c) for c in record.get("calls", [])]
            prompt = record.get("prompt", "")
            title = f"{prompt[:60]}..." if len(prompt) > 60 else (prompt or request_id)
            return render_timeline(calls, title=title)
        return f"<html><body style='background:#0d1117;color:#f85149;padding:40px'>Request {_esc(request_id)} not found.</body></html>"
    return render_index()
