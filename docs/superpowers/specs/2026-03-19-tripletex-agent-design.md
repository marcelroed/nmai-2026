# Tripletex AI Accounting Agent — Design Spec

**Date:** 2026-03-19
**Status:** Approved
**Competition:** NM i AI (Norwegian AI Championship) — Tripletex Challenge

---

## 1. Overview

Build an AI agent that exposes an HTTPS `/solve` endpoint. The competition platform sends accounting task prompts (in 7 languages) with Tripletex API credentials. The agent interprets the prompt, calls the Tripletex v2 REST API to complete the task, and gets scored on correctness and efficiency.

### Key constraints

- 30 task types × 56 variants (7 languages × 8 data sets)
- 5-minute timeout per submission
- Fresh Tripletex account per submission (starts empty)
- Efficiency bonus rewards minimal API calls and zero 4xx errors
- Score range: 0.0–6.0 per task (correctness × tier multiplier × efficiency bonus)
- Best score per task retained; bad runs never lower score

## 2. Approach: Plan-then-Execute with Verification

The agent follows a rigorous planning step before making any API calls. This maximizes first-try correctness and protects the efficiency bonus.

**Pipeline:**

```
Request → Intake → Query Rewrite → API Retrieval → Plan → Execute → Verify → Response
```

1. **Intake** — Parse prompt, decode files, extract data from PDFs/images via multimodal LLM
2. **Query Rewrite** — LLM rewrites multilingual prompt into structured English query
3. **API Retrieval** — ZeroEntropy (zembed-1 + zerank-2) retrieves relevant Tripletex API endpoint specs
4. **Plan** — LLM generates a structured JSON execution plan (ordered API calls with payloads and variable references)
5. **Execute** — Run each API call sequentially, resolve variable references, capped retries on failure
6. **Verify** — Optional single GET to confirm the created/modified entity
7. **Return** `{"status": "completed"}`

## 3. Components

### `main.py` — FastAPI app

- `/solve` POST endpoint
- Parses request (prompt, files, tripletex_credentials)
- API key auth: if `SOLVE_API_KEY` env var is set, validates incoming `Authorization: Bearer <key>` header
- Orchestrates the pipeline: intake → planner → retrieval → planner → executor → response

### `intake.py` — File processing

- Decodes base64 file attachments
- Sends PDFs/images to Claude or Gemini vision for structured data extraction
- Returns extracted text + structured data (amounts, names, dates, etc.)
- Passes through when no files are present

### `planner.py` — Query rewrite + execution planning

- **Query rewrite:** Takes prompt + extracted file data, produces a concise English search query for retrieval
- **Execution planning:** Takes prompt + file data + retrieved API specs, produces a JSON execution plan:

**Create flow example:**
```json
[
  {"step": 1, "method": "POST", "path": "/customer", "body": {"name": "Acme AS", "email": "post@acme.no", "isCustomer": true}, "capture": {"customer_id": "value.id"}},
  {"step": 2, "method": "POST", "path": "/order", "body": {"customer": {"id": "$customer_id"}}, "capture": {"order_id": "value.id"}},
  {"step": 3, "method": "POST", "path": "/invoice", "body": {"customer": {"id": "$customer_id"}, "orders": [{"id": "$order_id"}]}}
]
```

**Search-then-modify flow example:**
```json
[
  {"step": 1, "method": "GET", "path": "/customer", "params": {"name": "Acme", "fields": "id,name,email", "count": 1}, "capture": {"customer_id": "values.0.id"}},
  {"step": 2, "method": "PUT", "path": "/customer/$customer_id", "body": {"id": "$customer_id", "name": "Acme AS", "phoneNumber": "+4712345678"}}
]
```

**Delete flow example:**
```json
[
  {"step": 1, "method": "GET", "path": "/travelExpense", "params": {"fields": "id", "count": 100}, "capture": {"expense_id": "values.0.id"}},
  {"step": 2, "method": "DELETE", "path": "/travelExpense/$expense_id"}
]
```

**Module enablement example:**
```json
[
  {"step": 1, "method": "PUT", "path": "/company/modules", "body": {"moduleDepartment": true}},
  {"step": 2, "method": "POST", "path": "/department", "body": {"name": "Sales"}, "capture": {"dept_id": "value.id"}}
]
```

**Plan schema:**
- `method`: GET, POST, PUT, or DELETE
- `path`: endpoint path (may contain `$variable` references)
- `body`: JSON body for POST/PUT (may contain `$variable` references)
- `params`: query parameters for GET (including `fields` for efficiency)
- `capture`: object mapping variable names to JSONPath-like extraction paths:
  - `value.id` — for POST/PUT single-entity responses
  - `values.0.id` — for GET list responses (first result)
  - `values.*.id` — for GET list responses (all results, captured as array)

- Variable references (`$variable_name`) can appear in body values, path segments, and params
- The planner must handle all task patterns: create, search-then-modify, search-then-delete, and multi-step workflows with prerequisites (including module enablement)
- Query rewrite and planning are combined into a single LLM call to save time and tokens

### `retrieval.py` — ZeroEntropy retrieval interface

- `retrieve(query: str) -> list[APIEndpointSpec]`
- Calls zembed-1 (on Modal) for embedding-based search over indexed Tripletex API specs
- Calls zerank-2 (on Modal) for reranking results
- Returns top-5 most relevant endpoint specs (sufficient for multi-step Tier 3 tasks)
- Clean interface — swap provider by reimplementing this module

### `executor.py` — Plan execution

- Takes structured plan + Tripletex credentials
- Executes each step sequentially via `tripletex_client`
- Resolves `$variable` references from prior response payloads using capture paths (e.g., `value.id`, `values.0.id`)
- Supports variable substitution in body values, URL path segments, and query params
- On 4xx error: sends error + plan step + API spec to LLM for one corrected retry
- Global retry cap: 2 retries across entire plan
- No retry on 5xx or 401
- Timeout guard: if approaching 4 minutes, abort remaining steps and return `completed`

### `tripletex_client.py` — Thin HTTP client

- Wraps `httpx` with Basic Auth (username `0`, password = session_token)
- Base URL from credentials (`base_url` field)
- Uses `fields` query parameter where appropriate to minimize response payloads
- Returns parsed JSON response
- Raises on non-2xx with full error body for retry logic

### `llm.py` — LLM provider abstraction

- `complete(messages, model="claude") -> str` — text completion
- `extract_from_file(file_bytes, mime_type, prompt) -> str` — multimodal vision extraction
- Model selection via `LLM_PROVIDER` environment variable (`claude` | `gemini`)
- Supports Anthropic Claude and Google Gemini (model ID configured via env var, e.g., `GEMINI_MODEL=gemini-2.5-pro`)

## 4. Retrieval & Indexing Strategy

### Offline indexing (one-time setup)

1. Fetch the OpenAPI/Swagger spec from `https://kkpqfuj-amager.tripletex.dev/v2-docs/`
2. Parse and chunk by endpoint — each chunk contains:
   - Path + HTTP method (e.g., `POST /employee`)
   - Description
   - Request body schema (field names, types, required/optional, descriptions)
   - Response schema
   - Query parameters (for GET endpoints)
3. Upload chunks to ZeroEntropy as documents via their API
4. Implemented as `scripts/index_api.py`

### Runtime retrieval

1. LLM rewrites prompt into English search query
2. zembed-1 retrieves top-15 relevant endpoint specs
3. zerank-2 reranks to top-5
4. Top-5 specs provided to planner as context (covers multi-step Tier 3 tasks needing 4-5 endpoints)

### Example indexed document

```
Endpoint: POST /employee
Description: Create a new employee
Required fields: firstName (string), lastName (string), email (string)
Optional fields: phoneNumber, department.id, roles[].id, ...
Response: { value: { id: number, firstName: string, ... } }
```

## 5. LLM Usage

### Three calls per task (typical)

| Call | Purpose | Input | Output |
|------|---------|-------|--------|
| 1. File extraction (if files) | Extract data from PDFs/images | File bytes + extraction prompt | Structured JSON data |
| 2. Query rewrite + plan | Interpret prompt, plan API calls | Prompt + file data + retrieved API specs | JSON execution plan |
| 3. Retry (only on 4xx) | Fix a failed call | Failed call + error + API spec | Corrected API call |

### Language handling

Prompts come in 7 languages (nb, en, es, pt, nn, de, fr). Both Claude and Gemini handle multilingual input natively — no explicit translation step needed.

## 6. Error Handling & Efficiency Protection

### Retry policy

- Per-step: 1 retry max (LLM-corrected)
- Global: 2 retries max across entire plan
- No retry on 5xx or 401
- Partial completion preferred over timeout

### Timeout budget

| Phase | Budget | Notes |
|-------|--------|-------|
| File extraction (if files) | 15–30s | Single multimodal LLM call |
| Query rewrite + retrieval + planning | 15–25s | LLM call + 2 ZeroEntropy API calls |
| Plan execution (API calls) | 30–90s | Depends on task complexity (1–6 API calls) |
| Retries (if any) | 15–30s | LLM call + corrected API call, max 2 |
| Safety margin | 60s | Hard cutoff at 240s elapsed |

**Hard timeout at 4 minutes (240s).** If elapsed time exceeds this, abort remaining steps and return `completed`. This leaves 60s of safety margin before the platform's 300s timeout.

### Logging

- Original prompt, retrieved API specs, generated plan
- Each API call: request, response, timing
- Retries with error messages
- All logged locally for debugging between submissions

## 7. Project Structure

```
tripletex/
├── pyproject.toml
├── Dockerfile
├── src/
│   ├── main.py
│   ├── intake.py
│   ├── planner.py
│   ├── retrieval.py
│   ├── executor.py
│   ├── tripletex_client.py
│   └── llm.py
├── scripts/
│   └── index_api.py
└── tests/
    └── ...
```

### Dependencies

- `fastapi`, `uvicorn` — web framework
- `httpx` — HTTP client (async-compatible with FastAPI)
- `anthropic` — Claude API
- `google-genai` — Gemini API
- ZeroEntropy via plain `httpx` to Modal endpoints (no SDK)

## 8. Deployment

- **Platform:** Google Cloud Run
- **Container:** Python 3.13 slim, deps via uv, uvicorn entrypoint
- **Config:** 1 instance min, 300s timeout, 1GB RAM, 1 vCPU
- **Secrets:** `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `ZEROENTROPY_API_KEY`, `SOLVE_API_KEY` (optional), `LLM_PROVIDER`, `GEMINI_MODEL` as env vars
- **Submit:** Endpoint URL at `https://app.ainm.no/submit/tripletex`

### Local development

- `uvicorn src.main:app --reload`
- `npx cloudflared tunnel --url http://localhost:8000` for HTTPS testing

## 9. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Plan-then-Execute over ReAct loop | Efficiency bonus rewards minimal calls; planning upfront reduces trial-and-error |
| ZeroEntropy for retrieval | Competitive edge with zembed-1/zerank-2; clean interface allows swapping providers |
| Full OpenAPI spec indexing | Gives LLM complete field-level context for correct first-try API calls |
| Multimodal LLM for files | Both Claude and Gemini support vision natively; simpler than OCR pipeline |
| Combined query rewrite + planning | One LLM call instead of two; saves time within 5-minute window |
| Capped retries (2 global) | Protects efficiency bonus while allowing recovery from occasional errors |
| No language detection/translation | Claude and Gemini handle multilingual input natively |
| Top-5 retrieval (not top-3) | Multi-step Tier 3 tasks may need 4-5 different endpoints |
| Module enablement in planner | Some tasks require enabling Tripletex modules before API calls work |
| JSONPath-like capture syntax | Handles both single-entity (POST) and list (GET) response shapes |
| httpx over requests | Async-compatible with FastAPI; single HTTP client library |
| `fields` param optimization | Reduces response payload sizes for efficiency |
