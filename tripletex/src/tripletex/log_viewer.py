"""Generate HTML timeline viewer for parsed logs."""

import html
import json
import re
import webbrowser
from pathlib import Path


TASK_NAMES = {
    1: "Create employee", 2: "Create customer", 3: "Create product", 4: "Register supplier",
    5: "Create departments", 6: "Create & send invoice", 7: "Register payment",
    8: "Create project", 9: "Multi-line invoice", 10: "Order to invoice",
    11: "Register supplier invoice", 12: "Payroll with bonus", 13: "Travel expense",
    14: "Credit note", 15: "Fixed price project", 16: "Time logging invoice",
    17: "Custom dimension", 18: "Reverse payment", 19: "Employee from PDF",
    20: "Supplier invoice PDF", 21: "Onboarding from PDF", 22: "Expense from receipt PDF",
    23: "Bank reconciliation CSV", 24: "Ledger error correction", 25: "Overdue reminder",
    26: "Monthly closing", 27: "Currency invoice", 28: "Cost increase analysis",
    29: "Complete project lifecycle", 30: "Year-end closing",
}


def classify_prompt(p: str) -> int:
    """Classify a prompt into task 1-30 by structural patterns."""
    t = p.lower()
    if any(k in t for k in ['vedlagt pdf', 'pdf ci-joint', 'pdf anexo', 'pdf adjunto', 'attached pdf', 'beigefugte pdf']):
        if any(k in t for k in ['arbeidskontrakt', 'contrat de travail', 'contrato de trabalho', 'employment contract', 'arbeitsvertrag']): return 19
        if any(k in t for k in ['leverandorfaktura', 'leverandørfaktura', 'facture fournisseur', 'fatura de fornecedor']): return 20
        if any(k in t for k in ['tilbudsbrev', 'tilbodsbrev', "lettre d'offre", 'carta de oferta', 'angebotsschreiben']): return 21
    if any(k in t for k in ['kvittering', 'this receipt', 'ce recu', 'este recibo', 'dieser quittung', 'deste recibo']): return 22
    if any(k in t for k in ['bankutskrift', 'bank statement', 'kontoauszug', 'extracto bancario', 'extrato bancario', 'releve bancaire']) and 'csv' in t: return 23
    if any(k in t for k in ['årsoppgjør', 'year-end closing', 'jahresabschluss', 'clôture annuelle', 'cierre anual', 'encerramento anual']): return 30
    if any(k in t for k in ['kreditnota', 'credit note', 'avoir complet', 'nota de crédito', 'nota de credito']): return 14
    if '1)' in p and '2)' in p and '3)' in p and '4)' in p: return 29
    if any(k in t for k in ['monatsabschluss', 'encerramento mensal', 'cierre mensual']) or (any(k in t for k in ['rechnungsabgrenzung', 'acréscimos', 'gehaltsrückstellung', 'provisão salarial', 'periodificación', 'provisión salarial']) and any(k in t for k in ['abschreibung', 'depreciação', 'depreciación'])): return 26
    if any(k in t for k in ['hovudboka', 'hovedbok', 'general ledger', 'hauptbuch', 'libro mayor', 'grand livre', 'livro razão']) and ('4 feil' in t or '4 error' in t or '4 fehler' in t or '4 err' in t): return 24
    if any(k in t for k in ['purregebyr', 'mahngebuhr', 'reminder fee', 'cargo por recordatorio', 'frais de rappel', 'taxa de lembrete']): return 25
    if any(k in t for k in ['zurückgebucht', 'returned by the bank', 'retourné par la banque', 'devuelto por el banco', 'reverser betalingen', 'reverse the payment', 'stornieren sie die zahlung', 'annulez le paiement']): return 18
    if any(k in t for k in ['kostnadsøkning', 'kostnadskontoane', 'cost increase', 'costs increased', 'costos totales aumentaron', 'coûts totaux', 'custos totais', 'gesamtkosten', 'totalkostnadene']): return 28
    if 'eur' in t and any(k in t for k in ['agio', 'disagio', 'wechselkurs', 'câmbio', 'cambio', 'change']): return 27
    if any(k in t for k in ['rekneskapsdimensjon', 'dimensión contable', 'buchhaltungsdimension']) or ('dimensjon' in t and 'bokfør' in t): return 17
    if re.search(r'\d+\s*(hours?|timer?|timar|stunden|horas|heures)', t) and any(k in t for k in ['nok/h', 'hourly rate', 'stundensatz', 'taxa horária', 'horária']): return 16
    if any(k in t for k in ['reiseregning', 'reiserekning', 'travel expense', 'reisekostenabrechnung']) and any(k in t for k in ['flybillett', 'flight ticket', 'taxi', 'flugticket']): return 13
    if any(k in t for k in ['grunnlønn', 'grunnløn', 'base salary', 'grundgehalt', 'salaire de base', 'salario base', 'salário base']): return 12
    if any(k in t for k in ['fastpris', 'fixed price', 'festpreis', 'prix forfaitaire', 'precio fijo']): return 15
    if any(k in t for k in ['tre produktlinjer', 'drei produkt', 'trois lignes', 'tres líneas', 'três linhas']): return 9
    if re.search(r'INV-\d{4}-\d+', p): return 11
    if any(k in t for k in ['konverter ordren til faktura', 'convierte el pedido en factura', 'converta o pedido em fatura', 'wandeln sie den auftrag']): return 10
    if any(k in t for k in ['opprett prosjektet', 'crie o projeto', 'erstellen sie das projekt', 'create the project', 'créez le projet']): return 8
    if any(k in t for k in ['uteståande faktura', 'outstanding invoice', 'offene rechnung', 'factura pendiente', 'facture impayée', 'paiement intégral']): return 7
    if any(k in t for k in ['opprett og send', 'crea y envía', 'create and send', 'crie e envie', 'erstellen und senden sie eine rechnung']): return 6
    if any(k in t for k in ['avdelinger', 'departamentos', 'départements', 'avdelingar', 'departments']): return 5
    if any(k in t for k in ['opprett produktet', 'crie o produto', 'create the product']): return 3
    if any(k in t for k in ['registrer leverandøren', 'registe o fornecedor', 'register the supplier', 'registrieren sie den lieferanten']): return 4
    if any(k in t for k in ['opprett kunden', 'crie o cliente', 'créez le client', 'crea el cliente', 'create the customer']): return 2
    if any(k in t for k in ['ny ansatt', 'ny tilsett', 'nuevo empleado', 'nouvel employé', 'new employee', 'opprett en ansatt']): return 1
    return 0


def build_flow(entries: list[dict]) -> dict:
    """Parse log entries into a structured request flow."""
    entries.sort(key=lambda e: e.get("timestamp", ""))

    flow = {
        "request_id": "",
        "prompt": "",
        "files": [],
        "base_url": "",
        "session_token": "",
        "timestamp_start": "",
        "timestamp_end": "",
        "status_code": None,
        "duration_ms": None,
        "phases": [],  # ordered list of flow steps
    }

    # Pending GET requests (endpoint → timestamp)
    pending = {}

    for e in entries:
        msg = e["message"].split("] ", 1)[-1] if "] " in e["message"] else e["message"]
        extra = e.get("extra", {})
        ts = e.get("timestamp", "")
        level = e["level"]

        # --- Lifecycle ---
        if msg == "request.start":
            flow["timestamp_start"] = ts
            flow["phases"].append({
                "type": "lifecycle",
                "label": "Request received",
                "timestamp": ts,
                "details": {
                    "client": extra.get("client"),
                    "method": extra.get("request_line"),
                },
            })

        elif msg == "serve.handler.enter":
            flow["prompt"] = extra.get("prompt", "")
            flow["base_url"] = extra.get("base_url", "")
            payload = extra.get("payload", "")
            if isinstance(payload, str):
                flow["files"] = re.findall(r"filename='([^']+)'", payload)
                m = re.search(r"session_token='([^']+)'", payload)
                if m:
                    flow["session_token"] = m.group(1)
            flow["phases"].append({
                "type": "lifecycle",
                "label": "Handler started",
                "timestamp": ts,
                "details": {
                    "prompt": flow["prompt"],
                    "files": flow["files"],
                    "file_count": extra.get("file_count"),
                },
            })

        elif msg == "request.end":
            flow["timestamp_end"] = ts
            flow["status_code"] = extra.get("status_code")
            flow["duration_ms"] = extra.get("duration_ms")
            flow["phases"].append({
                "type": "lifecycle",
                "label": f"Request completed — {extra.get('status_code')} ({extra.get('duration_ms', 0):.0f}ms)",
                "timestamp": ts,
            })

        elif msg == "serve.handler.before_return":
            flow["phases"].append({
                "type": "lifecycle",
                "label": "Handler returning",
                "timestamp": ts,
            })

        # --- Submission status ---
        elif "submission" in msg:
            pass  # skip meta noise

        # --- Endpoint: request sent ---
        elif msg.startswith("getting data for "):
            endpoint = msg.replace("getting data for ", "")
            pending[endpoint] = ts

        elif msg.startswith("w/params data for "):
            endpoint = msg.replace("w/params data for ", "")
            pending[endpoint] = ts

        # --- Endpoint: success ---
        elif msg.startswith("got tripletex data for ") or msg.startswith("w/params got tripletex data for "):
            endpoint = extra.get("endpoint", "")
            req_ts = pending.pop(endpoint, "")
            data = extra.get("data")
            params = extra.get("params")

            record_count = None
            records = []
            meta = {}
            if isinstance(data, dict):
                records = data.get("values", [])
                record_count = len(records) if isinstance(records, list) else None
                meta = {k: v for k, v in data.items() if k != "values"}

            flow["phases"].append({
                "type": "endpoint_success",
                "label": endpoint,
                "timestamp_request": req_ts,
                "timestamp_response": ts,
                "params": params,
                "record_count": record_count,
                "records": records,
                "meta": meta,
            })

        # --- Endpoint: failure ---
        elif msg.startswith("failed to get tripletex data for ") or msg.startswith("w/params failed"):
            endpoint = extra.get("endpoint", "")
            req_ts = pending.pop(endpoint, "")

            flow["phases"].append({
                "type": "endpoint_error",
                "label": endpoint,
                "timestamp_request": req_ts,
                "timestamp_response": ts,
                "params": extra.get("params"),
                "error": extra.get("error", "unknown"),
            })

    return flow


def build_flow_v3(data: dict) -> dict:
    """Parse pre-structured v3 log format into the same flow structure."""
    flow = {
        "request_id": data.get("request_id", ""),
        "prompt": "",
        "files": [],
        "base_url": "",
        "session_token": "",
        "timestamp_start": "",
        "timestamp_end": "",
        "status_code": None,
        "duration_ms": None,
        "phases": [],
    }

    # Request start
    req = data.get("request", {})
    if req:
        flow["timestamp_start"] = req.get("timestamp", "")
        flow["phases"].append({
            "type": "lifecycle",
            "label": "Request received",
            "timestamp": req.get("timestamp", ""),
            "details": {"client": req.get("extra", {}).get("client")},
        })

    # Handler enter
    handler = data.get("handler", {})
    enter = handler.get("enter", {}) if isinstance(handler, dict) else {}
    if enter:
        extra = enter.get("extra", {})
        flow["prompt"] = extra.get("prompt", "")
        flow["base_url"] = extra.get("base_url", "")
        payload = extra.get("payload", "")
        if isinstance(payload, str):
            flow["files"] = re.findall(r"filename='([^']+)'", payload)
            m = re.search(r"session_token='([^']+)'", payload)
            if m:
                flow["session_token"] = m.group(1)
        flow["phases"].append({
            "type": "lifecycle",
            "label": "Handler started",
            "timestamp": enter.get("timestamp", ""),
            "details": {
                "prompt": flow["prompt"],
                "files": flow["files"],
                "file_count": extra.get("file_count"),
            },
        })

    # Endpoints
    for ep in data.get("tripletex_endpoints", []):
        endpoint = ep.get("endpoint", "")
        getting = ep.get("getting", {})
        result = ep.get("result", {})
        kind = result.get("kind", "")

        if kind == "got_tripletex_data" or kind == "w_params_got_tripletex_data":
            result_extra = result.get("extra", {})
            result_data = result_extra.get("data")
            params = result_extra.get("params")

            record_count = None
            records = []
            meta = {}
            if isinstance(result_data, dict):
                records = result_data.get("values", [])
                record_count = len(records) if isinstance(records, list) else None
                meta = {k: v for k, v in result_data.items() if k != "values"}

            flow["phases"].append({
                "type": "endpoint_success",
                "label": endpoint,
                "timestamp_request": getting.get("timestamp", ""),
                "timestamp_response": result.get("timestamp", ""),
                "params": params,
                "record_count": record_count,
                "records": records,
                "meta": meta,
            })
        else:
            result_extra = result.get("extra", {})
            flow["phases"].append({
                "type": "endpoint_error",
                "label": endpoint,
                "timestamp_request": getting.get("timestamp", ""),
                "timestamp_response": result.get("timestamp", ""),
                "params": result_extra.get("params"),
                "error": result_extra.get("error", result.get("message", "unknown")),
            })

    # Request end
    req_end = data.get("request_end", {})
    if req_end:
        end_extra = req_end.get("extra", {})
        flow["timestamp_end"] = req_end.get("timestamp", "")
        flow["status_code"] = end_extra.get("status_code")
        flow["duration_ms"] = end_extra.get("duration_ms")
        flow["phases"].append({
            "type": "lifecycle",
            "label": f"Request completed — {end_extra.get('status_code')} ({end_extra.get('duration_ms', 0):.0f}ms)",
            "timestamp": req_end.get("timestamp", ""),
        })

    return flow


def render_log(path: str) -> str:
    with open(path) as f:
        data = json.load(f)

    # Detect format: v3 is a dict with 'request_id', v1/v2 is a list
    if isinstance(data, dict) and "request_id" in data:
        flow = build_flow_v3(data)
    else:
        flow = build_flow(data)

    request_id = Path(path).stem

    # Count endpoints
    successes = [p for p in flow["phases"] if p["type"] == "endpoint_success"]
    errors = [p for p in flow["phases"] if p["type"] == "endpoint_error"]
    with_data = [p for p in successes if p.get("record_count", 0) and p["record_count"] > 0]
    empty = [p for p in successes if not p.get("record_count", 0) or p["record_count"] == 0]

    # Build phase HTML
    phase_html = []
    for i, p in enumerate(flow["phases"]):
        if p["type"] == "lifecycle":
            phase_html.append(f"""
            <div class="phase lifecycle">
              <div class="phase-icon">&#9679;</div>
              <div class="phase-content">
                <div class="phase-label">{html.escape(p['label'])}</div>
                <div class="phase-time">{_time(p.get('timestamp', ''))}</div>
              </div>
            </div>""")

            # Show prompt details inline after "Handler started"
            if "Handler started" in p["label"] and p.get("details"):
                det = p["details"]
                prompt_html = html.escape(det.get("prompt", ""))
                files_html = ""
                if det.get("files"):
                    files_html = "<br>".join(f"📎 {html.escape(f)}" for f in det["files"])
                    files_html = f'<div class="prompt-files">{files_html}</div>'
                phase_html.append(f"""
                <div class="phase prompt-box">
                  <div class="phase-icon">💬</div>
                  <div class="phase-content">
                    <div class="prompt-text">{prompt_html}</div>
                    {files_html}
                  </div>
                </div>""")

        elif p["type"] == "endpoint_success":
            count = p.get("record_count")
            count_str = f"{count} records" if count is not None else "ok"
            params_badge = ""
            if p.get("params"):
                params_badge = f'<span class="badge params">params: {html.escape(json.dumps(p["params"]))}</span>'

            empty_class = " empty" if count == 0 else " has-data"

            # Build records preview
            records_html = ""
            if p.get("records") and len(p["records"]) > 0:
                records_json = json.dumps(p["records"], indent=2, ensure_ascii=False)
                records_escaped = html.escape(records_json)
                records_html = f"""
                <div class="endpoint-data" id="data-{i}" style="display:none;">
                  <pre>{records_escaped}</pre>
                </div>"""

            phase_html.append(f"""
            <div class="phase endpoint success{empty_class}">
              <div class="phase-icon">{'✓' if count != 0 else '○'}</div>
              <div class="phase-content">
                <div class="endpoint-header" {'onclick=toggle(\"data-' + str(i) + '\")' if records_html else ''} {'style=cursor:pointer' if records_html else ''}>
                  <span class="endpoint-name">GET {html.escape(p['label'])}</span>
                  {params_badge}
                  <span class="badge count{'-zero' if count == 0 else ''}">{count_str}</span>
                  <span class="phase-time">{_time(p.get('timestamp_request', ''))} → {_time(p.get('timestamp_response', ''))}</span>
                </div>
                {records_html}
              </div>
            </div>""")

        elif p["type"] == "endpoint_error":
            params_badge = ""
            if p.get("params"):
                params_badge = f'<span class="badge params">params: {html.escape(json.dumps(p["params"]))}</span>'

            err_short = html.escape((p.get("error", "")[:100]))

            phase_html.append(f"""
            <div class="phase endpoint error">
              <div class="phase-icon">✗</div>
              <div class="phase-content">
                <div class="endpoint-header">
                  <span class="endpoint-name">GET {html.escape(p['label'])}</span>
                  {params_badge}
                  <span class="badge error-badge">FAILED</span>
                  <span class="phase-time">{_time(p.get('timestamp_request', ''))} → {_time(p.get('timestamp_response', ''))}</span>
                </div>
                <div class="error-msg">{err_short}</div>
              </div>
            </div>""")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Log: {request_id[:8]}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 14px; background: #0d1117; color: #c9d1d9; padding: 24px 40px; max-width: 1200px; margin: 0 auto; }}
  code, pre, .endpoint-name, .badge {{ font-family: 'SF Mono', 'Menlo', monospace; }}

  .header {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 18px; color: #f0f6fc; margin-bottom: 12px; }}
  .header-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 16px; font-size: 13px; }}
  .header-grid dt {{ color: #8b949e; }}
  .header-grid dd {{ color: #c9d1d9; }}
  .header-grid .prompt {{ color: #f0883e; }}

  .stats {{ display: flex; gap: 16px; margin-bottom: 20px; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; text-align: center; }}
  .stat-num {{ font-size: 24px; font-weight: bold; }}
  .stat-label {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
  .stat.green .stat-num {{ color: #3fb950; }}
  .stat.grey .stat-num {{ color: #484f58; }}
  .stat.red .stat-num {{ color: #f85149; }}

  .controls {{ margin-bottom: 16px; display: flex; gap: 8px; }}
  .controls button {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
  .controls button:hover {{ background: #30363d; }}

  .timeline {{ position: relative; padding-left: 32px; }}
  .timeline::before {{ content: ''; position: absolute; left: 11px; top: 0; bottom: 0; width: 2px; background: #21262d; }}

  .phase {{ position: relative; margin-bottom: 4px; display: flex; align-items: flex-start; gap: 12px; }}
  .phase-icon {{ width: 24px; min-width: 24px; text-align: center; font-size: 14px; position: relative; z-index: 1; }}

  .phase.lifecycle .phase-icon {{ color: #58a6ff; }}
  .phase.lifecycle .phase-label {{ color: #58a6ff; font-weight: 600; font-size: 13px; }}

  .phase.prompt-box {{ background: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin: 4px 0 12px 0; }}
  .prompt-text {{ color: #f0883e; line-height: 1.5; }}
  .prompt-files {{ margin-top: 8px; color: #8b949e; font-size: 13px; }}

  .phase.endpoint {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 8px 12px; }}
  .phase.endpoint:hover {{ border-color: #30363d; }}
  .endpoint-header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .endpoint-name {{ font-size: 13px; font-weight: 600; }}
  .phase.success .phase-icon {{ color: #3fb950; }}
  .phase.success .endpoint-name {{ color: #3fb950; }}
  .phase.success.empty .phase-icon {{ color: #484f58; }}
  .phase.success.empty .endpoint-name {{ color: #8b949e; }}
  .phase.error .phase-icon {{ color: #f85149; }}
  .phase.error .endpoint-name {{ color: #f85149; }}

  .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 10px; }}
  .badge.count {{ background: #1f6feb33; color: #58a6ff; }}
  .badge.count-zero {{ background: #21262d; color: #484f58; }}
  .badge.params {{ background: #d2992233; color: #d29922; }}
  .badge.error-badge {{ background: #f8514933; color: #f85149; }}

  .phase-time {{ font-size: 11px; color: #484f58; margin-left: auto; font-family: 'SF Mono', monospace; }}
  .error-msg {{ font-size: 12px; color: #f85149; margin-top: 4px; opacity: 0.8; }}

  .endpoint-data {{ margin-top: 8px; }}
  .endpoint-data pre {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; font-size: 12px; color: #79c0ff; max-height: 500px; overflow: auto; white-space: pre-wrap; word-break: break-all; }}

  .hidden {{ display: none !important; }}
</style>
</head>
<body>
<div class="header">
  <h1>Request {request_id}</h1>
  <dl class="header-grid">
    <dt>Prompt</dt><dd class="prompt">{html.escape(flow['prompt'])}</dd>
    {'<dt>Files</dt><dd>' + ', '.join(flow['files']) + '</dd>' if flow['files'] else ''}
    <dt>Base URL</dt><dd>{html.escape(flow['base_url'])}</dd>
    <dt>Duration</dt><dd>{flow['duration_ms']:.0f}ms</dd>
    <dt>Status</dt><dd>{flow['status_code']}</dd>
    <dt>Time</dt><dd>{flow['timestamp_start']} → {flow['timestamp_end']}</dd>
  </dl>
</div>

<div class="stats">
  <div class="stat green"><div class="stat-num">{len(with_data)}</div><div class="stat-label">Returned Data</div></div>
  <div class="stat grey"><div class="stat-num">{len(empty)}</div><div class="stat-label">Empty</div></div>
  <div class="stat red"><div class="stat-num">{len(errors)}</div><div class="stat-label">Failed</div></div>
</div>

<div class="controls">
  <button onclick="showAll()">Show All</button>
  <button onclick="showOnly('has-data')">Returned Data</button>
  <button onclick="showOnly('empty')">Empty</button>
  <button onclick="showOnly('error')">Failed</button>
  <button onclick="expandAllData()">Expand All</button>
  <button onclick="collapseAllData()">Collapse All</button>
</div>

<div class="timeline" id="timeline">
{''.join(phase_html)}
</div>

<script>
function toggle(id) {{
  var el = document.getElementById(id);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}
function showAll() {{
  document.querySelectorAll('.phase.endpoint').forEach(el => el.classList.remove('hidden'));
}}
function showOnly(cls) {{
  document.querySelectorAll('.phase.endpoint').forEach(el => {{
    el.classList.toggle('hidden', !el.classList.contains(cls));
  }});
}}
function expandAllData() {{
  document.querySelectorAll('.endpoint-data').forEach(el => el.style.display = 'block');
}}
function collapseAllData() {{
  document.querySelectorAll('.endpoint-data').forEach(el => el.style.display = 'none');
}}
</script>
</body>
</html>"""


def _time(ts: str) -> str:
    if "T" in ts:
        return ts.split("T")[1]
    return ts


def _summarize_log_file(f: Path) -> dict:
    """Extract summary from a single log file (v1/v2 list or v3 dict)."""
    with open(f) as fh:
        data = json.load(fh)

    prompt = ""
    full_prompt = ""
    file_list = []
    status = ""
    duration = ""

    if isinstance(data, dict) and "request_id" in data:
        # v3 format
        log_version = "3"
        handler = data.get("handler", {})
        enter = handler.get("enter", {}) if isinstance(handler, dict) else {}
        extra = enter.get("extra", {}) if enter else {}
        full_prompt = extra.get("prompt", "")
        prompt = full_prompt[:140]
        payload = extra.get("payload", "")
        if isinstance(payload, str):
            file_list = re.findall(r"filename='([^']+)'", payload)
        req_end = data.get("request_end", {})
        if req_end:
            end_extra = req_end.get("extra", {})
            status = str(end_extra.get("status_code", ""))
            duration = f"{end_extra.get('duration_ms', 0):.0f}ms"
        entries_count = data.get("log_count", 0)
    else:
        # v1/v2 format
        entries = data if isinstance(data, list) else []
        log_version = entries[0].get("log_version", "?") if entries else "?"
        for e in entries:
            if "serve.handler.enter" in e.get("message", ""):
                full_prompt = e.get("extra", {}).get("prompt", "")
                prompt = full_prompt[:140]
                payload = e.get("extra", {}).get("payload", "")
                if isinstance(payload, str):
                    file_list = re.findall(r"filename='([^']+)'", payload)
            if "request.end" in e.get("message", ""):
                status = str(e.get("extra", {}).get("status_code", ""))
                duration = f"{e.get('extra', {}).get('duration_ms', 0):.0f}ms"
        entries_count = len(entries)

    task_num = classify_prompt(full_prompt) if full_prompt else 0
    return {
        "id": f.stem,
        "prompt": prompt,
        "files": file_list,
        "entries": entries_count,
        "status": status,
        "duration": duration,
        "log_version": log_version,
        "task": task_num,
    }


def render_index(log_dirs: list[str]) -> str:
    files = []
    for d in log_dirs:
        files.extend(sorted(Path(d).glob("*.json")))
    # Deduplicate by stem (same request_id in multiple dirs)
    seen = set()
    unique_files = []
    for f in files:
        if f.stem not in seen:
            seen.add(f.stem)
            unique_files.append(f)
    files = sorted(unique_files, key=lambda f: f.stem)

    summaries = [_summarize_log_file(f) for f in files]

    from collections import Counter
    version_counts = Counter(s["log_version"] for s in summaries)
    task_counts = Counter(s["task"] for s in summaries)

    # Group summaries by task
    by_task: dict[int, list] = {}
    for s in summaries:
        by_task.setdefault(s["task"], []).append(s)

    # Build grouped HTML
    groups_html = []
    for task_num in sorted(by_task.keys()):
        items = by_task[task_num]
        task_label = f"Task {task_num}: {TASK_NAMES.get(task_num, 'Unknown')}" if task_num else "Unclassified"

        rows = []
        for s in items:
            files_str = ", ".join(s["files"]) if s["files"] else ""
            v = s["log_version"]
            rows.append(f"""
            <tr class="log-row v{v} task-{task_num}" onclick="window.open('{s['id']}.html')" style="cursor:pointer;">
              <td><span class="version-badge v{v}">v{v}</span></td>
              <td><code>{s['id'][:12]}</code></td>
              <td class="prompt-col">{html.escape(s['prompt'])}</td>
              <td>{files_str}</td>
              <td>{s['duration']}</td>
            </tr>""")

        groups_html.append(f"""
        <div class="task-group collapsed" data-task="{task_num}">
          <div class="task-header" onclick="toggleGroup(this)">
            <span class="task-toggle">&#9660;</span>
            <span class="task-label">{html.escape(task_label)}</span>
            <span class="task-count">{len(items)}</span>
          </div>
          <table class="task-table">
            <tr><th>Ver</th><th>Request ID</th><th>Prompt</th><th>Files</th><th>Duration</th></tr>
            {''.join(rows)}
          </table>
        </div>""")

    version_buttons = []
    for v in sorted(version_counts.keys()):
        version_buttons.append(
            f'<button class="ver-btn active" data-ver="{v}" onclick="toggleVersion(\'{v}\', this)">v{v} ({version_counts[v]})</button>'
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Log Viewer</title>
<style>
  body {{ font-family: -apple-system, sans-serif; font-size: 13px; background: #0d1117; color: #c9d1d9; padding: 24px 40px; }}
  h1 {{ color: #f0f6fc; margin-bottom: 8px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 16px; }}
  .filters {{ margin-bottom: 20px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  .filters span {{ color: #8b949e; margin-right: 4px; }}
  .ver-btn {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }}
  .ver-btn:hover {{ background: #30363d; }}
  .ver-btn.active {{ border-color: #58a6ff; color: #58a6ff; }}
  .ver-btn.inactive {{ opacity: 0.4; }}

  .task-group {{ margin-bottom: 4px; }}
  .task-header {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 10px 16px; cursor: pointer; display: flex; align-items: center; gap: 10px; }}
  .task-header:hover {{ background: #1c2128; }}
  .task-toggle {{ color: #484f58; font-size: 12px; transition: transform 0.15s; }}
  .task-group.collapsed .task-toggle {{ transform: rotate(-90deg); }}
  .task-group.collapsed .task-table {{ display: none; }}
  .task-label {{ font-weight: 600; color: #58a6ff; }}
  .task-count {{ background: #21262d; color: #8b949e; font-size: 11px; padding: 2px 8px; border-radius: 10px; font-family: 'SF Mono', monospace; }}

  .task-table {{ width: 100%; border-collapse: collapse; margin: 0 0 8px 0; }}
  .task-table th {{ text-align: left; padding: 6px 10px; color: #8b949e; border-bottom: 1px solid #21262d; font-size: 11px; }}
  .task-table td {{ padding: 8px 10px; border-bottom: 1px solid #161b22; }}
  .task-table tr:hover td {{ background: #161b22; }}
  tr.hidden {{ display: none; }}
  code {{ color: #58a6ff; font-family: 'SF Mono', monospace; }}
  .prompt-col {{ color: #f0883e; max-width: 600px; }}
  .version-badge {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; font-family: 'SF Mono', monospace; }}
  .version-badge.v1 {{ background: #1f6feb33; color: #58a6ff; }}
  .version-badge.v2 {{ background: #3fb95033; color: #3fb950; }}
  .version-badge.v3 {{ background: #d2992233; color: #d29922; }}
</style>
</head>
<body>
<h1>Log Viewer</h1>
<div class="subtitle">{len(summaries)} requests across {len(by_task)} task types</div>
<div class="filters">
  <span>Log version:</span>
  {''.join(version_buttons)}
  <span style="margin-left:16px">Groups:</span>
  <button class="ver-btn" onclick="expandAllGroups()">Expand All</button>
  <button class="ver-btn" onclick="collapseAllGroups()">Collapse All</button>
</div>
{''.join(groups_html)}
<script>
var activeVersions = new Set({json.dumps(sorted(version_counts.keys()))});

function toggleVersion(ver, btn) {{
  if (activeVersions.has(ver)) {{
    activeVersions.delete(ver);
    btn.classList.remove('active');
    btn.classList.add('inactive');
  }} else {{
    activeVersions.add(ver);
    btn.classList.add('active');
    btn.classList.remove('inactive');
  }}
  document.querySelectorAll('.log-row').forEach(function(row) {{
    var rowVer = Array.from(row.classList).find(c => c.match(/^v\\d+$/));
    if (rowVer) rowVer = rowVer.substring(1);
    row.classList.toggle('hidden', !activeVersions.has(rowVer));
  }});
  updateCounts();
}}
function updateCounts() {{
  document.querySelectorAll('.task-group').forEach(function(group) {{
    var visible = group.querySelectorAll('.log-row:not(.hidden)').length;
    group.querySelector('.task-count').textContent = visible;
    group.style.display = visible === 0 ? 'none' : '';
  }});
}}
function toggleGroup(header) {{
  header.parentElement.classList.toggle('collapsed');
}}
function expandAllGroups() {{
  document.querySelectorAll('.task-group').forEach(g => g.classList.remove('collapsed'));
}}
function collapseAllGroups() {{
  document.querySelectorAll('.task-group').forEach(g => g.classList.add('collapsed'));
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML log viewer")
    parser.add_argument("--dir", nargs="*", default=["data/parsed_logs_v2", "data/parsed_logs"], help="Log directories")
    parser.add_argument("--file", help="Single log file")
    parser.add_argument("--out", default="/tmp/log_viewer", help="Output directory")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        html_content = render_log(args.file)
        out_path = out_dir / f"{Path(args.file).stem}.html"
        out_path.write_text(html_content)
        print(f"Wrote {out_path}")
        if args.open:
            webbrowser.open(f"file://{out_path}")
    else:
        log_dirs = [d for d in args.dir if Path(d).is_dir()]
        all_files = []
        seen = set()
        for d in log_dirs:
            for f in sorted(Path(d).glob("*.json")):
                if f.stem not in seen:
                    seen.add(f.stem)
                    all_files.append(f)
        all_files.sort(key=lambda f: f.stem)

        print(f"Generating {len(all_files)} log pages + index from {log_dirs}...")

        for i, f in enumerate(all_files):
            html_content = render_log(str(f))
            (out_dir / f"{f.stem}.html").write_text(html_content)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(all_files)}")

        index_html = render_index(log_dirs)
        index_path = out_dir / "index.html"
        index_path.write_text(index_html)
        print(f"Wrote {len(all_files)} pages + index to {out_dir}/")

        if args.open:
            webbrowser.open(f"file://{index_path}")
