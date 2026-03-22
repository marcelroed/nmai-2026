# Tripletex AI Accounting Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AI agent that exposes an HTTPS `/solve` endpoint, interprets multilingual accounting task prompts, and executes them against the Tripletex v2 REST API using a plan-then-execute architecture with ZeroEntropy retrieval.

**Architecture:** Plan-then-Execute pipeline: Intake → Query Rewrite → API Retrieval (ZeroEntropy) → LLM Planning → Sequential Execution → Response. Six modules with clean interfaces: `main.py`, `intake.py`, `planner.py`, `retrieval.py`, `executor.py`, `tripletex_client.py`, `llm.py`.

**Tech Stack:** Python 3.13, FastAPI, httpx, Anthropic SDK, Google GenAI SDK, ZeroEntropy (zembed-1/zerank-2 via Modal endpoints)

**Spec:** `docs/superpowers/specs/2026-03-19-tripletex-agent-design.md`
**Challenge docs:** `tripletex/docs.md`

---

## File Structure

```
tripletex/
├── pyproject.toml                    # Project config + dependencies
├── Dockerfile                        # Cloud Run container
├── src/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, /solve endpoint, orchestration
│   ├── config.py                     # Environment variable loading, constants
│   ├── tripletex_client.py           # Thin httpx wrapper for Tripletex API
│   ├── llm.py                        # LLM provider abstraction (Claude/Gemini)
│   ├── intake.py                     # File decoding + multimodal extraction
│   ├── retrieval.py                  # ZeroEntropy retrieval interface
│   ├── planner.py                    # Query rewrite + execution plan generation
│   └── executor.py                   # Plan execution + variable resolution + retries
├── scripts/
│   └── index_api.py                  # One-time: fetch OpenAPI spec, chunk, upload to ZeroEntropy
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_tripletex_client.py
    ├── test_llm.py
    ├── test_intake.py
    ├── test_retrieval.py
    ├── test_planner.py
    ├── test_executor.py
    └── test_main.py
```

---

### Task 1: Project Setup & Dependencies

**Files:**
- Modify: `tripletex/pyproject.toml`
- Create: `tripletex/src/__init__.py`
- Create: `tripletex/src/config.py`
- Create: `tripletex/tests/__init__.py`
- Create: `tripletex/tests/test_config.py`

- [ ] **Step 1: Update pyproject.toml with all dependencies**

```toml
[project]
name = "tripletex"
version = "0.1.0"
description = "AI Accounting Agent for NM i AI Tripletex Challenge"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "httpx>=0.28.0",
    "anthropic>=0.52.0",
    "google-genai>=1.14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25.0",
    "respx>=0.22.0",
]
```

- [ ] **Step 2: Create package init files**

`tripletex/src/__init__.py`:
```python
```

`tripletex/tests/__init__.py`:
```python
```

- [ ] **Step 3: Write failing test for config**

`tripletex/tests/test_config.py`:
```python
import os

def test_config_loads_defaults():
    from src.config import Config
    config = Config()
    assert config.llm_provider in ("claude", "gemini")
    assert config.timeout_budget == 240

def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from src.config import Config
    config = Config()
    assert config.llm_provider == "gemini"
    assert config.gemini_model == "gemini-2.5-pro"
    assert config.anthropic_api_key == "test-key"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd tripletex && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError` for `src.config`

- [ ] **Step 5: Implement config module**

`tripletex/src/config.py`:
```python
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "claude"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
    zeroentropy_api_key: str = field(default_factory=lambda: os.getenv("ZEROENTROPY_API_KEY", ""))
    zeroentropy_base_url: str = field(default_factory=lambda: os.getenv("ZEROENTROPY_BASE_URL", ""))
    solve_api_key: str = field(default_factory=lambda: os.getenv("SOLVE_API_KEY", ""))
    timeout_budget: int = 240
    max_retries: int = 2
    retrieval_top_k: int = 15
    retrieval_rerank_k: int = 5
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd tripletex && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd tripletex && git add pyproject.toml src/ tests/ && git commit -m "feat: project setup with config module and dependencies"
```

---

### Task 2: Tripletex HTTP Client

**Files:**
- Create: `tripletex/src/tripletex_client.py`
- Create: `tripletex/tests/test_tripletex_client.py`

- [ ] **Step 1: Write failing tests for tripletex client**

`tripletex/tests/test_tripletex_client.py`:
```python
import httpx
import pytest
import respx

from src.tripletex_client import TripletexClient, TripletexAPIError


@pytest.fixture
def client():
    return TripletexClient(
        base_url="https://tx-proxy.ainm.no/v2",
        session_token="test-token-123",
    )


@respx.mock
def test_get_request(client):
    respx.get("https://tx-proxy.ainm.no/v2/employee").mock(
        return_value=httpx.Response(200, json={"fullResultSize": 1, "values": [{"id": 1, "firstName": "Ola"}]})
    )
    result = client.get("/employee", params={"fields": "id,firstName"})
    assert result["values"][0]["firstName"] == "Ola"


@respx.mock
def test_post_request(client):
    respx.post("https://tx-proxy.ainm.no/v2/customer").mock(
        return_value=httpx.Response(201, json={"value": {"id": 42, "name": "Acme AS"}})
    )
    result = client.post("/customer", json={"name": "Acme AS", "isCustomer": True})
    assert result["value"]["id"] == 42


@respx.mock
def test_put_request(client):
    respx.put("https://tx-proxy.ainm.no/v2/customer/42").mock(
        return_value=httpx.Response(200, json={"value": {"id": 42, "name": "Updated"}})
    )
    result = client.put("/customer/42", json={"id": 42, "name": "Updated"})
    assert result["value"]["name"] == "Updated"


@respx.mock
def test_delete_request(client):
    respx.delete("https://tx-proxy.ainm.no/v2/travelExpense/10").mock(
        return_value=httpx.Response(204)
    )
    result = client.delete("/travelExpense/10")
    assert result is None


@respx.mock
def test_4xx_raises_api_error(client):
    respx.post("https://tx-proxy.ainm.no/v2/employee").mock(
        return_value=httpx.Response(422, json={"status": 422, "message": "firstName is required"})
    )
    with pytest.raises(TripletexAPIError) as exc_info:
        client.post("/employee", json={"lastName": "Nordmann"})
    assert exc_info.value.status_code == 422
    assert "firstName is required" in str(exc_info.value)


@respx.mock
def test_basic_auth_is_sent(client):
    route = respx.get("https://tx-proxy.ainm.no/v2/employee").mock(
        return_value=httpx.Response(200, json={"values": []})
    )
    client.get("/employee")
    request = route.calls[0].request
    assert request.headers["authorization"].startswith("Basic ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && uv run pytest tests/test_tripletex_client.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tripletex client**

`tripletex/src/tripletex_client.py`:
```python
import httpx


class TripletexAPIError(Exception):
    def __init__(self, status_code: int, message: str, body: dict | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Tripletex API error {status_code}: {message}")


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            auth=("0", session_token),
            timeout=30.0,
        )

    def _request(self, method: str, path: str, **kwargs) -> dict | None:
        url = f"{self.base_url}{path}"
        response = self._client.request(method, url, **kwargs)
        if response.status_code == 204:
            return None
        if response.status_code >= 400:
            try:
                body = response.json()
                message = body.get("message", response.text)
            except Exception:
                body = None
                message = response.text
            raise TripletexAPIError(response.status_code, message, body)
        return response.json()

    def get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    def post(self, path: str, json: dict | None = None) -> dict:
        return self._request("POST", path, json=json)

    def put(self, path: str, json: dict | None = None) -> dict:
        return self._request("PUT", path, json=json)

    def delete(self, path: str) -> None:
        return self._request("DELETE", path)

    def close(self):
        self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && uv run pytest tests/test_tripletex_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd tripletex && git add src/tripletex_client.py tests/test_tripletex_client.py && git commit -m "feat: tripletex HTTP client with basic auth and error handling"
```

---

### Task 3: LLM Provider Abstraction

**Files:**
- Create: `tripletex/src/llm.py`
- Create: `tripletex/tests/test_llm.py`

- [ ] **Step 1: Write failing tests for LLM abstraction**

`tripletex/tests/test_llm.py`:
```python
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from src.llm import LLMClient
from src.config import Config


@pytest.fixture
def claude_config():
    return Config(llm_provider="claude", anthropic_api_key="test-key")


@pytest.fixture
def gemini_config():
    return Config(llm_provider="gemini", gemini_api_key="test-key", gemini_model="gemini-2.5-pro")


def test_llm_client_init_claude(claude_config):
    client = LLMClient(claude_config)
    assert client.provider == "claude"


def test_llm_client_init_gemini(gemini_config):
    client = LLMClient(gemini_config)
    assert client.provider == "gemini"


@patch("src.llm.anthropic")
def test_complete_claude(mock_anthropic, claude_config):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="test response")]
    )
    client = LLMClient(claude_config)
    result = client.complete([{"role": "user", "content": "hello"}])
    assert result == "test response"
    mock_client.messages.create.assert_called_once()


@patch("src.llm.anthropic")
def test_complete_with_system_prompt(mock_anthropic, claude_config):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="response")]
    )
    client = LLMClient(claude_config)
    result = client.complete(
        [{"role": "user", "content": "hello"}],
        system="You are a helpful assistant",
    )
    assert result == "response"
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["system"] == "You are a helpful assistant"


@patch("src.llm.anthropic")
def test_extract_from_file_claude(mock_anthropic, claude_config):
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"amount": 1500}')]
    )
    client = LLMClient(claude_config)
    result = client.extract_from_file(
        file_bytes=b"fake-pdf-content",
        mime_type="application/pdf",
        prompt="Extract invoice data",
    )
    assert result == '{"amount": 1500}'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && uv run pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement LLM client**

`tripletex/src/llm.py`:
```python
import base64
import logging

import anthropic
import google.genai as genai
from google.genai import types as genai_types

from src.config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Config):
        self.provider = config.llm_provider
        self._config = config
        if self.provider == "claude":
            self._claude = anthropic.Anthropic(api_key=config.anthropic_api_key)
        elif self.provider == "gemini":
            self._gemini = genai.Client(api_key=config.gemini_api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def complete(self, messages: list[dict], system: str | None = None, max_tokens: int = 4096) -> str:
        if self.provider == "claude":
            return self._complete_claude(messages, system, max_tokens)
        return self._complete_gemini(messages, system, max_tokens)

    def extract_from_file(self, file_bytes: bytes, mime_type: str, prompt: str) -> str:
        if self.provider == "claude":
            return self._extract_claude(file_bytes, mime_type, prompt)
        return self._extract_gemini(file_bytes, mime_type, prompt)

    def _complete_claude(self, messages: list[dict], system: str | None, max_tokens: int) -> str:
        kwargs = {
            "model": self._config.claude_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._claude.messages.create(**kwargs)
        return response.content[0].text

    def _complete_gemini(self, messages: list[dict], system: str | None, max_tokens: int) -> str:
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system,
        )
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=msg["content"])]))
        response = self._gemini.models.generate_content(
            model=self._config.gemini_model,
            contents=contents,
            config=config,
        )
        return response.text

    def _extract_claude(self, file_bytes: bytes, mime_type: str, prompt: str) -> str:
        b64 = base64.standard_b64encode(file_bytes).decode()
        content = [
            {"type": "document" if mime_type == "application/pdf" else "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
            {"type": "text", "text": prompt},
        ]
        response = self._claude.messages.create(
            model=self._config.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def _extract_gemini(self, file_bytes: bytes, mime_type: str, prompt: str) -> str:
        part_data = genai_types.Part(inline_data=genai_types.Blob(mime_type=mime_type, data=file_bytes))
        part_text = genai_types.Part(text=prompt)
        response = self._gemini.models.generate_content(
            model=self._config.gemini_model,
            contents=[genai_types.Content(role="user", parts=[part_data, part_text])],
        )
        return response.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && uv run pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd tripletex && git add src/llm.py tests/test_llm.py && git commit -m "feat: LLM provider abstraction for Claude and Gemini"
```

---

### Task 4: Intake — File Processing

**Files:**
- Create: `tripletex/src/intake.py`
- Create: `tripletex/tests/test_intake.py`

- [ ] **Step 1: Write failing tests for intake**

`tripletex/tests/test_intake.py`:
```python
import base64
from unittest.mock import MagicMock

import pytest

from src.intake import process_files


def test_no_files_returns_none():
    mock_llm = MagicMock()
    result = process_files([], mock_llm)
    assert result is None
    mock_llm.extract_from_file.assert_not_called()


def test_single_pdf_file():
    mock_llm = MagicMock()
    mock_llm.extract_from_file.return_value = '{"invoice_number": "INV-001", "amount": 1500}'
    pdf_content = b"fake-pdf-bytes"
    files = [
        {
            "filename": "faktura.pdf",
            "content_base64": base64.b64encode(pdf_content).decode(),
            "mime_type": "application/pdf",
        }
    ]
    result = process_files(files, mock_llm)
    assert result is not None
    assert len(result) == 1
    assert result[0]["filename"] == "faktura.pdf"
    assert result[0]["extracted_data"] == '{"invoice_number": "INV-001", "amount": 1500}'
    call_args = mock_llm.extract_from_file.call_args
    assert call_args[1]["file_bytes"] == pdf_content
    assert call_args[1]["mime_type"] == "application/pdf"
    assert isinstance(call_args[1]["prompt"], str)


def test_multiple_files():
    mock_llm = MagicMock()
    mock_llm.extract_from_file.side_effect = ['{"data": "from_pdf"}', '{"data": "from_image"}']
    files = [
        {"filename": "doc.pdf", "content_base64": base64.b64encode(b"pdf").decode(), "mime_type": "application/pdf"},
        {"filename": "receipt.png", "content_base64": base64.b64encode(b"png").decode(), "mime_type": "image/png"},
    ]
    result = process_files(files, mock_llm)
    assert len(result) == 2
    assert mock_llm.extract_from_file.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && uv run pytest tests/test_intake.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement intake module**

`tripletex/src/intake.py`:
```python
import base64
import logging

from src.llm import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract all structured data from this document. Return a JSON object with all relevant fields you can find, such as:
- Names (people, companies)
- Amounts, prices, totals
- Dates
- Invoice numbers, reference numbers
- Addresses, emails, phone numbers
- Line items (products, descriptions, quantities, unit prices)
- Any other structured data

Return ONLY valid JSON, no explanation."""


def process_files(files: list[dict], llm: LLMClient) -> list[dict] | None:
    if not files:
        return None

    results = []
    for f in files:
        file_bytes = base64.b64decode(f["content_base64"])
        logger.info("Extracting data from %s (%s, %d bytes)", f["filename"], f["mime_type"], len(file_bytes))
        extracted = llm.extract_from_file(
            file_bytes=file_bytes,
            mime_type=f["mime_type"],
            prompt=EXTRACTION_PROMPT,
        )
        results.append({
            "filename": f["filename"],
            "mime_type": f["mime_type"],
            "extracted_data": extracted,
        })

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && uv run pytest tests/test_intake.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd tripletex && git add src/intake.py tests/test_intake.py && git commit -m "feat: intake module for file extraction via multimodal LLM"
```

---

### Task 5: ZeroEntropy Retrieval Interface

**Files:**
- Create: `tripletex/src/retrieval.py`
- Create: `tripletex/tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests for retrieval**

`tripletex/tests/test_retrieval.py`:
```python
import httpx
import pytest
import respx

from src.config import Config
from src.retrieval import RetrievalClient


@pytest.fixture
def config():
    return Config(
        zeroentropy_api_key="test-ze-key",
        zeroentropy_base_url="https://ze.modal.run",
        retrieval_top_k=15,
        retrieval_rerank_k=5,
    )


@pytest.fixture
def client(config):
    return RetrievalClient(config)


@respx.mock
def test_retrieve_returns_ranked_results(client):
    # Mock embedding search
    respx.post("https://ze.modal.run/search").mock(
        return_value=httpx.Response(200, json={
            "results": [
                {"content": "POST /employee - Create employee", "score": 0.95},
                {"content": "GET /employee - List employees", "score": 0.90},
                {"content": "PUT /employee/{id} - Update employee", "score": 0.85},
                {"content": "POST /customer - Create customer", "score": 0.60},
                {"content": "GET /department - List departments", "score": 0.55},
            ]
        })
    )
    # Mock reranking
    respx.post("https://ze.modal.run/rerank").mock(
        return_value=httpx.Response(200, json={
            "results": [
                {"content": "POST /employee - Create employee", "score": 0.98},
                {"content": "PUT /employee/{id} - Update employee", "score": 0.92},
                {"content": "GET /employee - List employees", "score": 0.88},
                {"content": "POST /customer - Create customer", "score": 0.40},
                {"content": "GET /department - List departments", "score": 0.35},
            ]
        })
    )
    results = client.retrieve("create employee with name and email")
    assert len(results) == 5
    assert "POST /employee" in results[0]


@respx.mock
def test_retrieve_empty_query(client):
    respx.post("https://ze.modal.run/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    results = client.retrieve("")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && uv run pytest tests/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement retrieval client**

`tripletex/src/retrieval.py`:
```python
import logging

import httpx

from src.config import Config

logger = logging.getLogger(__name__)


class RetrievalClient:
    def __init__(self, config: Config):
        self._config = config
        self._base_url = config.zeroentropy_base_url.rstrip("/")
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {config.zeroentropy_api_key}"},
            timeout=15.0,
        )

    def retrieve(self, query: str) -> list[str]:
        if not query:
            return []

        # Step 1: Embedding search with zembed-1
        search_response = self._client.post(
            f"{self._base_url}/search",
            json={
                "query": query,
                "top_k": self._config.retrieval_top_k,
            },
        )
        search_response.raise_for_status()
        search_results = search_response.json().get("results", [])

        if not search_results:
            return []

        # Step 2: Rerank with zerank-2
        documents = [r["content"] for r in search_results]
        rerank_response = self._client.post(
            f"{self._base_url}/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_k": self._config.retrieval_rerank_k,
            },
        )
        rerank_response.raise_for_status()
        rerank_results = rerank_response.json().get("results", [])

        ranked_docs = [r["content"] for r in rerank_results]
        logger.info("Retrieved %d API specs for query: %s", len(ranked_docs), query[:80])
        return ranked_docs

    def close(self):
        self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && uv run pytest tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd tripletex && git add src/retrieval.py tests/test_retrieval.py && git commit -m "feat: ZeroEntropy retrieval client with search + rerank"
```

---

### Task 6: Planner — Query Rewrite + Execution Plan

**Files:**
- Create: `tripletex/src/planner.py`
- Create: `tripletex/tests/test_planner.py`

- [ ] **Step 1: Write failing tests for planner**

`tripletex/tests/test_planner.py`:
```python
import json
from unittest.mock import MagicMock

import pytest

from src.planner import Planner, ExecutionPlan, PlanStep, PlanResult


def test_plan_step_model():
    step = PlanStep(step=1, method="POST", path="/employee", body={"firstName": "Ola"}, capture={"emp_id": "value.id"})
    assert step.method == "POST"
    assert step.capture["emp_id"] == "value.id"


def test_plan_step_with_params():
    step = PlanStep(step=1, method="GET", path="/customer", params={"name": "Acme", "fields": "id,name"})
    assert step.params["name"] == "Acme"
    assert step.body is None


def test_execution_plan_from_json():
    raw = [
        {"step": 1, "method": "POST", "path": "/customer", "body": {"name": "Acme"}, "capture": {"cid": "value.id"}},
        {"step": 2, "method": "POST", "path": "/invoice", "body": {"customer": {"id": "$cid"}}},
    ]
    plan = ExecutionPlan(steps=[PlanStep(**s) for s in raw])
    assert len(plan.steps) == 2
    assert plan.steps[1].body["customer"]["id"] == "$cid"


def test_plan():
    mock_llm = MagicMock()
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = ["POST /employee - Create employee. Required: firstName, lastName, email"]
    plan_json = json.dumps({
        "search_query": "create employee with first name Ola, last name Nordmann, email, administrator role",
        "steps": [
            {"step": 1, "method": "POST", "path": "/employee", "body": {"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org"}, "capture": {"emp_id": "value.id"}}
        ]
    })
    mock_llm.complete.return_value = plan_json
    planner = Planner(mock_llm, mock_retrieval)
    result = planner.plan(
        prompt="Opprett en ansatt med navn Ola Nordmann",
        file_data=None,
    )
    assert len(result.execution_plan.steps) == 1
    assert result.execution_plan.steps[0].method == "POST"
    assert result.execution_plan.steps[0].path == "/employee"
    assert "employee" in result.search_query.lower()
    # Two LLM calls: initial plan (to get search_query), then refined plan with retrieved API specs
    assert mock_llm.complete.call_count == 2


def test_plan_with_markdown_fences():
    mock_llm = MagicMock()
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = ["POST /employee"]
    plan_json = '```json\n{"search_query": "create employee", "steps": [{"step": 1, "method": "POST", "path": "/employee", "body": {"firstName": "Ola"}}]}\n```'
    mock_llm.complete.return_value = plan_json
    planner = Planner(mock_llm, mock_retrieval)
    result = planner.plan(prompt="Create employee", file_data=None)
    assert len(result.execution_plan.steps) == 1


def test_create_retry_step():
    mock_llm = MagicMock()
    corrected = json.dumps({"step": 1, "method": "POST", "path": "/employee", "body": {"firstName": "Ola", "lastName": "Nordmann"}})
    mock_llm.complete.return_value = corrected
    planner = Planner(mock_llm)
    step = planner.create_retry_step(
        failed_step=PlanStep(step=1, method="POST", path="/employee", body={"firstName": "Ola"}),
        error_message="lastName is required",
        api_spec="POST /employee - Required: firstName, lastName",
    )
    assert step.body["lastName"] == "Nordmann"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && uv run pytest tests/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement planner**

`tripletex/src/planner.py`:
```python
import json
import logging
import re
from dataclasses import dataclass, field

from src.llm import LLMClient

logger = logging.getLogger(__name__)

PLAN_SYSTEM = """You are an accounting automation agent. Given a task prompt, produce a JSON object with two fields:

1. "search_query": A concise English search query to find relevant Tripletex API endpoints (e.g., "create employee with name email administrator role")
2. "steps": A JSON array of execution steps

Step rules:
- Each step has: step (number), method (GET/POST/PUT/DELETE), path, and optionally body, params, capture
- Use $variable_name to reference values captured from previous steps
- The capture field maps variable names to extraction paths:
  - "value.id" for POST/PUT single-entity responses
  - "values.0.id" for GET list responses (first result)
  - "values.*.id" for GET list responses (all results)
- Use the "fields" query parameter in GET requests to minimize response size
- Include ONLY necessary API calls — efficiency is scored
- All field names must match the API spec exactly
- Dates in YYYY-MM-DD format
- Some tasks require enabling modules first (e.g., PUT /company/modules with moduleDepartment: true)
- For modify/delete tasks, first GET to find the entity ID, then PUT/DELETE

Return ONLY valid JSON, no explanation."""

PLAN_WITH_SPECS_SYSTEM = """You are an accounting automation agent. Given a task prompt and Tripletex API documentation, produce a JSON object with two fields:

1. "search_query": The search query you used (echo it back)
2. "steps": A JSON array of execution steps

Step rules:
- Each step has: step (number), method (GET/POST/PUT/DELETE), path, and optionally body, params, capture
- Use $variable_name to reference values captured from previous steps
- The capture field maps variable names to extraction paths:
  - "value.id" for POST/PUT single-entity responses
  - "values.0.id" for GET list responses (first result)
  - "values.*.id" for GET list responses (all results)
- Use the "fields" query parameter in GET requests to minimize response size
- Include ONLY necessary API calls — efficiency is scored
- All field names must match the API spec exactly
- Dates in YYYY-MM-DD format
- Some tasks require enabling modules first (e.g., PUT /company/modules with moduleDepartment: true)
- For modify/delete tasks, first GET to find the entity ID, then PUT/DELETE

Return ONLY valid JSON, no explanation."""

RETRY_SYSTEM = """You are fixing a failed Tripletex API call. Given the original step, the error message, and the API spec, produce a corrected step as a JSON object.

Fix the issue described in the error message. Return ONLY the corrected JSON step object, no explanation."""


@dataclass
class PlanStep:
    step: int
    method: str
    path: str
    body: dict | None = None
    params: dict | None = None
    capture: dict | None = None


@dataclass
class ExecutionPlan:
    steps: list[PlanStep]


@dataclass
class PlanResult:
    search_query: str
    execution_plan: ExecutionPlan
    api_specs: list[str]


class Planner:
    def __init__(self, llm: LLMClient, retrieval=None):
        self._llm = llm
        self._retrieval = retrieval

    def plan(self, prompt: str, file_data: list[dict] | None = None) -> PlanResult:
        """Combined query rewrite + retrieval + planning in minimal LLM calls.

        Step 1: Single LLM call to produce search_query + initial plan
        Step 2: Retrieve API specs using the search_query
        Step 3: Second LLM call with retrieved specs to produce final plan
        """
        user_content = f"Task prompt:\n{prompt}"
        if file_data:
            file_summary = "\n".join(f"- {f['filename']}: {f['extracted_data']}" for f in file_data)
            user_content += f"\n\nExtracted file data:\n{file_summary}"

        # Single LLM call: get search query + initial plan
        raw = self._llm.complete(
            messages=[{"role": "user", "content": user_content}],
            system=PLAN_SYSTEM,
            max_tokens=4096,
        )
        parsed = _parse_json_object(raw)
        search_query = parsed.get("search_query", "")
        steps_data = parsed.get("steps", [])

        # Retrieve API specs
        api_specs = []
        if self._retrieval and search_query:
            api_specs = self._retrieval.retrieve(search_query)

        # If we got API specs, refine the plan with a second LLM call
        if api_specs:
            specs_text = "\n\n---\n\n".join(api_specs)
            refined_content = user_content + f"\n\nRelevant Tripletex API documentation:\n{specs_text}"
            raw = self._llm.complete(
                messages=[{"role": "user", "content": refined_content}],
                system=PLAN_WITH_SPECS_SYSTEM,
                max_tokens=4096,
            )
            parsed = _parse_json_object(raw)
            steps_data = parsed.get("steps", steps_data)

        steps = [PlanStep(**s) for s in steps_data]
        logger.info("Created execution plan with %d steps", len(steps))
        return PlanResult(
            search_query=search_query,
            execution_plan=ExecutionPlan(steps=steps),
            api_specs=api_specs,
        )

    def create_retry_step(self, failed_step: PlanStep, error_message: str, api_spec: str) -> PlanStep:
        step_json = json.dumps({
            "step": failed_step.step,
            "method": failed_step.method,
            "path": failed_step.path,
            "body": failed_step.body,
            "params": failed_step.params,
            "capture": failed_step.capture,
        })
        user_content = f"Failed step:\n{step_json}\n\nError message:\n{error_message}\n\nAPI spec:\n{api_spec}"
        raw = self._llm.complete(
            messages=[{"role": "user", "content": user_content}],
            system=RETRY_SYSTEM,
            max_tokens=2048,
        )
        step_data = _parse_json_object(raw)
        return PlanStep(**step_data)


def _parse_json_object(text: str) -> dict:
    cleaned = _strip_markdown_fences(text)
    return json.loads(cleaned)


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && uv run pytest tests/test_planner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd tripletex && git add src/planner.py tests/test_planner.py && git commit -m "feat: planner with query rewrite, execution planning, and retry"
```

---

### Task 7: Executor — Plan Execution + Variable Resolution

**Files:**
- Create: `tripletex/src/executor.py`
- Create: `tripletex/tests/test_executor.py`

- [ ] **Step 1: Write failing tests for variable resolution**

`tripletex/tests/test_executor.py`:
```python
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.executor import Executor, resolve_capture, substitute_variables
from src.planner import PlanStep, ExecutionPlan
from src.tripletex_client import TripletexAPIError


# --- Variable resolution tests ---

def test_resolve_capture_value_id():
    response = {"value": {"id": 42, "name": "Test"}}
    result = resolve_capture(response, "value.id")
    assert result == 42


def test_resolve_capture_values_first():
    response = {"values": [{"id": 10}, {"id": 20}]}
    result = resolve_capture(response, "values.0.id")
    assert result == 10


def test_resolve_capture_values_all():
    response = {"values": [{"id": 10}, {"id": 20}, {"id": 30}]}
    result = resolve_capture(response, "values.*.id")
    assert result == [10, 20, 30]


def test_resolve_capture_nested():
    response = {"value": {"customer": {"id": 5}}}
    result = resolve_capture(response, "value.customer.id")
    assert result == 5


def test_substitute_variables_in_body():
    body = {"customer": {"id": "$cid"}, "name": "Test"}
    variables = {"cid": 42}
    result = substitute_variables(body, variables)
    assert result == {"customer": {"id": 42}, "name": "Test"}


def test_substitute_variables_in_path():
    path = "/customer/$cid"
    variables = {"cid": 42}
    result = substitute_variables(path, variables)
    assert result == "/customer/42"


def test_substitute_variables_in_list():
    body = {"orders": [{"id": "$oid"}]}
    variables = {"oid": 99}
    result = substitute_variables(body, variables)
    assert result == {"orders": [{"id": 99}]}


def test_substitute_variables_no_match():
    body = {"name": "Test"}
    variables = {"cid": 42}
    result = substitute_variables(body, variables)
    assert result == {"name": "Test"}


# --- Executor tests ---

@pytest.fixture
def mock_deps():
    client = MagicMock()
    planner = MagicMock()
    return client, planner


def test_executor_simple_post(mock_deps):
    client, planner = mock_deps
    client.post.return_value = {"value": {"id": 1, "firstName": "Ola"}}
    plan = ExecutionPlan(steps=[
        PlanStep(step=1, method="POST", path="/employee", body={"firstName": "Ola"}, capture={"emp_id": "value.id"}),
    ])
    executor = Executor(client, planner, timeout_budget=240, max_retries=2)
    result = executor.execute(plan)
    assert result.variables["emp_id"] == 1
    assert len(result.completed_steps) == 1
    client.post.assert_called_once_with("/employee", json={"firstName": "Ola"})


def test_executor_variable_threading(mock_deps):
    client, planner = mock_deps
    client.post.side_effect = [
        {"value": {"id": 10, "name": "Acme"}},
        {"value": {"id": 20}},
    ]
    plan = ExecutionPlan(steps=[
        PlanStep(step=1, method="POST", path="/customer", body={"name": "Acme"}, capture={"cid": "value.id"}),
        PlanStep(step=2, method="POST", path="/invoice", body={"customer": {"id": "$cid"}}),
    ])
    executor = Executor(client, planner, timeout_budget=240, max_retries=2)
    result = executor.execute(plan)
    assert result.variables["cid"] == 10
    second_call_body = client.post.call_args_list[1][1]["json"]
    assert second_call_body["customer"]["id"] == 10


def test_executor_get_with_params(mock_deps):
    client, planner = mock_deps
    client.get.return_value = {"values": [{"id": 5, "name": "Acme"}]}
    client.put.return_value = {"value": {"id": 5, "name": "Updated"}}
    plan = ExecutionPlan(steps=[
        PlanStep(step=1, method="GET", path="/customer", params={"name": "Acme", "fields": "id,name"}, capture={"cid": "values.0.id"}),
        PlanStep(step=2, method="PUT", path="/customer/$cid", body={"id": "$cid", "name": "Updated"}),
    ])
    executor = Executor(client, planner, timeout_budget=240, max_retries=2)
    result = executor.execute(plan)
    assert result.variables["cid"] == 5
    client.put.assert_called_once_with("/customer/5", json={"id": 5, "name": "Updated"})


def test_executor_delete(mock_deps):
    client, planner = mock_deps
    client.get.return_value = {"values": [{"id": 77}]}
    client.delete.return_value = None
    plan = ExecutionPlan(steps=[
        PlanStep(step=1, method="GET", path="/travelExpense", params={"fields": "id"}, capture={"eid": "values.0.id"}),
        PlanStep(step=2, method="DELETE", path="/travelExpense/$eid"),
    ])
    executor = Executor(client, planner, timeout_budget=240, max_retries=2)
    result = executor.execute(plan)
    client.delete.assert_called_once_with("/travelExpense/77")


def test_executor_retry_on_4xx(mock_deps):
    client, planner = mock_deps
    client.post.side_effect = [
        TripletexAPIError(422, "lastName is required"),
        {"value": {"id": 1}},
    ]
    planner.create_retry_step.return_value = PlanStep(
        step=1, method="POST", path="/employee", body={"firstName": "Ola", "lastName": "Nordmann"}
    )
    plan = ExecutionPlan(steps=[
        PlanStep(step=1, method="POST", path="/employee", body={"firstName": "Ola"}),
    ])
    executor = Executor(client, planner, timeout_budget=240, max_retries=2)
    result = executor.execute(plan)
    assert len(result.completed_steps) == 1
    assert result.retries_used == 1


def test_executor_respects_global_retry_cap(mock_deps):
    client, planner = mock_deps
    client.post.side_effect = [
        TripletexAPIError(422, "error 1"),
        {"value": {"id": 1}},
        TripletexAPIError(422, "error 2"),
        {"value": {"id": 2}},
        TripletexAPIError(422, "error 3"),  # should not retry this
    ]
    planner.create_retry_step.side_effect = [
        PlanStep(step=1, method="POST", path="/a", body={"fixed": True}),
        PlanStep(step=2, method="POST", path="/b", body={"fixed": True}),
    ]
    plan = ExecutionPlan(steps=[
        PlanStep(step=1, method="POST", path="/a", body={}),
        PlanStep(step=2, method="POST", path="/b", body={}),
        PlanStep(step=3, method="POST", path="/c", body={}),
    ])
    executor = Executor(client, planner, timeout_budget=240, max_retries=2)
    result = executor.execute(plan)
    assert result.retries_used == 2
    assert len(result.failed_steps) == 1  # step 3 failed without retry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && uv run pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement executor**

`tripletex/src/executor.py`:
```python
import logging
import time
from dataclasses import dataclass, field

from src.planner import Planner, PlanStep, ExecutionPlan
from src.tripletex_client import TripletexClient, TripletexAPIError

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    variables: dict = field(default_factory=dict)
    completed_steps: list[int] = field(default_factory=list)
    failed_steps: list[int] = field(default_factory=list)
    retries_used: int = 0


class Executor:
    def __init__(self, client: TripletexClient, planner: Planner, timeout_budget: int = 240, max_retries: int = 2):
        self._client = client
        self._planner = planner
        self._timeout_budget = timeout_budget
        self._max_retries = max_retries

    def execute(self, plan: ExecutionPlan, start_time: float | None = None, api_specs: list[str] | None = None) -> ExecutionResult:
        self._api_specs = api_specs or []
        if start_time is None:
            start_time = time.monotonic()

        result = ExecutionResult()

        for step in plan.steps:
            elapsed = time.monotonic() - start_time
            if elapsed >= self._timeout_budget:
                logger.warning("Timeout budget exceeded at step %d (%.1fs elapsed)", step.step, elapsed)
                break

            resolved_step = self._resolve_step(step, result.variables)
            try:
                response = self._execute_step(resolved_step)
                self._capture_variables(response, resolved_step, result.variables)
                result.completed_steps.append(step.step)
                logger.info("Step %d completed: %s %s", step.step, step.method, step.path)
            except TripletexAPIError as e:
                if e.status_code == 401 or e.status_code >= 500:
                    logger.error("Step %d non-retryable error: %s", step.step, e)
                    result.failed_steps.append(step.step)
                    continue

                if result.retries_used >= self._max_retries:
                    logger.warning("Step %d failed, no retries left: %s", step.step, e)
                    result.failed_steps.append(step.step)
                    continue

                logger.info("Step %d failed with %d, attempting retry", step.step, e.status_code)
                result.retries_used += 1
                try:
                    corrected = self._planner.create_retry_step(
                        failed_step=step,
                        error_message=str(e),
                        api_spec="\n\n---\n\n".join(self._api_specs) if self._api_specs else "",
                    )
                    resolved_corrected = self._resolve_step(corrected, result.variables)
                    response = self._execute_step(resolved_corrected)
                    self._capture_variables(response, resolved_corrected, result.variables)
                    result.completed_steps.append(step.step)
                    logger.info("Step %d retry succeeded", step.step)
                except Exception as retry_err:
                    logger.error("Step %d retry failed: %s", step.step, retry_err)
                    result.failed_steps.append(step.step)

        # NOTE: The spec mentions an optional Verify step (single GET to confirm).
        # Deferred to a follow-up iteration — the planner can include a verification
        # GET as the last step of the execution plan if needed. Adding it here as a
        # hardcoded step risks wasting API calls on tasks where it's unnecessary.

        return result

    def _resolve_step(self, step: PlanStep, variables: dict) -> PlanStep:
        path = substitute_variables(step.path, variables)
        body = substitute_variables(step.body, variables) if step.body else None
        params = substitute_variables(step.params, variables) if step.params else None
        return PlanStep(
            step=step.step,
            method=step.method,
            path=path,
            body=body,
            params=params,
            capture=step.capture,
        )

    def _execute_step(self, step: PlanStep) -> dict | None:
        method = step.method.upper()
        if method == "GET":
            return self._client.get(step.path, params=step.params)
        elif method == "POST":
            return self._client.post(step.path, json=step.body)
        elif method == "PUT":
            return self._client.put(step.path, json=step.body)
        elif method == "DELETE":
            return self._client.delete(step.path)
        else:
            raise ValueError(f"Unknown HTTP method: {method}")

    def _capture_variables(self, response: dict | None, step: PlanStep, variables: dict):
        if not step.capture or response is None:
            return
        for var_name, capture_path in step.capture.items():
            variables[var_name] = resolve_capture(response, capture_path)
            logger.info("Captured %s = %s", var_name, variables[var_name])


def resolve_capture(response: dict, path: str):
    parts = path.split(".")
    current = response
    for i, part in enumerate(parts):
        if part == "*":
            # Collect from all items in the list
            remaining = ".".join(parts[i + 1:])
            return [_navigate(item, remaining) for item in current]
        elif part.isdigit():
            current = current[int(part)]
        else:
            current = current[part]
    return current


def _navigate(obj, path: str):
    if not path:
        return obj
    parts = path.split(".")
    current = obj
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = current[part]
    return current


def substitute_variables(value, variables: dict):
    if isinstance(value, str):
        # Check for exact match first (preserves type)
        if value.startswith("$") and value[1:] in variables:
            return variables[value[1:]]
        # Replace embedded references in paths, longest variable names first
        # to avoid $id matching inside $id2
        for var_name in sorted(variables.keys(), key=len, reverse=True):
            value = value.replace(f"${var_name}", str(variables[var_name]))
        return value
    elif isinstance(value, dict):
        return {k: substitute_variables(v, variables) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_variables(item, variables) for item in value]
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && uv run pytest tests/test_executor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd tripletex && git add src/executor.py tests/test_executor.py && git commit -m "feat: executor with variable resolution, capture, and capped retries"
```

---

### Task 8: FastAPI Main App — `/solve` Endpoint

**Files:**
- Create: `tripletex/src/main.py`
- Create: `tripletex/tests/test_main.py`

- [ ] **Step 1: Write failing tests for the /solve endpoint**

`tripletex/tests/test_main.py`:
```python
import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.planner import ExecutionPlan, PlanStep, PlanResult


@pytest.fixture
def client():
    return TestClient(app)


def test_solve_returns_completed(client):
    mock_llm = MagicMock()
    mock_retrieval = MagicMock()
    mock_tx = MagicMock()
    mock_tx.post.return_value = {"value": {"id": 1}}

    plan_result = PlanResult(
        search_query="create employee",
        execution_plan=ExecutionPlan(steps=[
            PlanStep(step=1, method="POST", path="/employee", body={"firstName": "Ola"})
        ]),
        api_specs=["POST /employee"],
    )

    with patch("src.main._create_llm", return_value=mock_llm), \
         patch("src.main._create_retrieval", return_value=mock_retrieval), \
         patch("src.main._create_tx_client", return_value=mock_tx), \
         patch("src.main.Planner") as MockPlanner:
        MockPlanner.return_value.plan.return_value = plan_result
        MockPlanner.return_value.create_retry_step = MagicMock()

        response = client.post("/solve", json={
            "prompt": "Opprett en ansatt med navn Ola Nordmann",
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://tx-proxy.ainm.no/v2",
                "session_token": "test-token",
            },
        })
        assert response.status_code == 200
        assert response.json() == {"status": "completed"}


def test_solve_rejects_bad_api_key(client):
    with patch("src.main.config") as mock_config:
        mock_config.solve_api_key = "my-secret-key"
        response = client.post(
            "/solve",
            json={
                "prompt": "test",
                "files": [],
                "tripletex_credentials": {"base_url": "https://example.com", "session_token": "t"},
            },
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && uv run pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement main app**

`tripletex/src/main.py`:
```python
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import Config
from src.executor import Executor
from src.intake import process_files
from src.llm import LLMClient
from src.planner import Planner
from src.retrieval import RetrievalClient
from src.tripletex_client import TripletexClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config = Config()
app = FastAPI(title="Tripletex AI Agent")


def _create_llm(cfg: Config) -> LLMClient:
    return LLMClient(cfg)


def _create_retrieval(cfg: Config) -> RetrievalClient:
    return RetrievalClient(cfg)


def _create_tx_client(base_url: str, session_token: str) -> TripletexClient:
    return TripletexClient(base_url=base_url, session_token=session_token)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    start_time = time.monotonic()

    # Auth check
    if config.solve_api_key:
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {config.solve_api_key}"
        if auth_header != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    logger.info("Received task: %s", prompt[:100])

    # Initialize clients via factory functions (mockable in tests)
    llm = _create_llm(config)
    retrieval = _create_retrieval(config)
    tx_client = _create_tx_client(base_url=creds["base_url"], session_token=creds["session_token"])
    planner = Planner(llm, retrieval)

    try:
        # 1. Intake — process files
        file_data = process_files(files, llm)

        # 2+3+4. Combined: query rewrite + retrieval + planning
        plan_result = planner.plan(prompt=prompt, file_data=file_data)
        logger.info("Search query: %s", plan_result.search_query)
        logger.info("Retrieved %d API specs", len(plan_result.api_specs))
        logger.info("Execution plan: %d steps", len(plan_result.execution_plan.steps))

        # 5. Execute
        executor = Executor(tx_client, planner, timeout_budget=config.timeout_budget, max_retries=config.max_retries)
        result = executor.execute(plan_result.execution_plan, start_time=start_time, api_specs=plan_result.api_specs)
        logger.info("Execution complete: %d succeeded, %d failed, %d retries",
                     len(result.completed_steps), len(result.failed_steps), result.retries_used)

    except Exception as e:
        logger.exception("Agent error: %s", e)
    finally:
        tx_client.close()
        retrieval.close()

    return JSONResponse({"status": "completed"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && uv run pytest tests/test_main.py -v`
Expected: PASS (may need adjustments to mock patching — fix as needed)

- [ ] **Step 5: Commit**

```bash
cd tripletex && git add src/main.py tests/test_main.py && git commit -m "feat: FastAPI /solve endpoint with full pipeline orchestration"
```

---

### Task 9: OpenAPI Indexing Script

**Files:**
- Create: `tripletex/scripts/__init__.py`
- Create: `tripletex/scripts/index_api.py`

This is a one-time setup script, not unit-tested in CI. It fetches the Tripletex OpenAPI spec, chunks it by endpoint, and uploads to ZeroEntropy.

- [ ] **Step 0: Create scripts package init and add chunks.json to .gitignore**

`tripletex/scripts/__init__.py`:
```python
```

Add to `.gitignore`:
```
scripts/chunks.json
```

- [ ] **Step 1: Implement the indexing script**

`tripletex/scripts/index_api.py`:
```python
"""One-time script: Fetch the Tripletex OpenAPI spec, chunk by endpoint, upload to ZeroEntropy.

Usage:
    ZEROENTROPY_API_KEY=... ZEROENTROPY_BASE_URL=... python -m scripts.index_api

Requires the Tripletex API docs to be accessible.
"""

import json
import os
import sys

import httpx

TRIPLETEX_DOCS_URL = "https://kkpqfuj-amager.tripletex.dev/v2-docs/"
ZE_BASE_URL = os.environ.get("ZEROENTROPY_BASE_URL", "")
ZE_API_KEY = os.environ.get("ZEROENTROPY_API_KEY", "")


def fetch_openapi_spec(url: str) -> dict:
    """Fetch the OpenAPI/Swagger spec. Try common paths."""
    client = httpx.Client(timeout=30.0)
    # Try common Swagger/OpenAPI JSON paths
    for path in ["", "swagger.json", "openapi.json", "v2/api-docs", "api-docs"]:
        try_url = f"{url.rstrip('/')}/{path}".rstrip("/")
        print(f"Trying: {try_url}")
        try:
            resp = client.get(try_url)
            if resp.status_code == 200:
                data = resp.json()
                if "paths" in data or "openapi" in data or "swagger" in data:
                    print(f"Found OpenAPI spec at {try_url}")
                    return data
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError(f"Could not find OpenAPI spec at {url}")


def chunk_by_endpoint(spec: dict) -> list[dict]:
    """Parse the OpenAPI spec and create one document per endpoint+method."""
    paths = spec.get("paths", {})
    definitions = spec.get("definitions", spec.get("components", {}).get("schemas", {}))
    chunks = []

    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ("parameters", "servers", "summary", "description"):
                continue
            method_upper = method.upper()
            summary = details.get("summary", "")
            description = details.get("description", "")
            tags = details.get("tags", [])
            parameters = details.get("parameters", [])

            # Build parameter descriptions
            param_lines = []
            for p in parameters:
                p_name = p.get("name", "?")
                p_in = p.get("in", "?")
                p_required = p.get("required", False)
                p_type = p.get("type", p.get("schema", {}).get("type", "?"))
                p_desc = p.get("description", "")
                req_str = "required" if p_required else "optional"
                param_lines.append(f"  - {p_name} ({p_in}, {p_type}, {req_str}): {p_desc}")

            # Build request body schema if present
            body_lines = []
            request_body = details.get("requestBody", {})
            if not request_body:
                # Swagger 2.0 style
                for p in parameters:
                    if p.get("in") == "body":
                        schema = p.get("schema", {})
                        body_lines = _format_schema(schema, definitions)
            else:
                content = request_body.get("content", {})
                json_content = content.get("application/json", {})
                schema = json_content.get("schema", {})
                body_lines = _format_schema(schema, definitions)

            # Build response schema
            responses = details.get("responses", {})
            resp_lines = []
            for status, resp_detail in responses.items():
                if status.startswith("2"):
                    schema = resp_detail.get("schema", {})
                    if not schema:
                        content = resp_detail.get("content", {})
                        schema = content.get("application/json", {}).get("schema", {})
                    resp_lines = _format_schema(schema, definitions)
                    break

            # Compose document
            doc = f"Endpoint: {method_upper} {path}\n"
            if tags:
                doc += f"Tags: {', '.join(tags)}\n"
            if summary:
                doc += f"Summary: {summary}\n"
            if description:
                doc += f"Description: {description}\n"
            if param_lines:
                doc += f"Parameters:\n" + "\n".join(param_lines) + "\n"
            if body_lines:
                doc += f"Request body:\n" + "\n".join(f"  {l}" for l in body_lines) + "\n"
            if resp_lines:
                doc += f"Response (2xx):\n" + "\n".join(f"  {l}" for l in resp_lines) + "\n"

            chunks.append({
                "content": doc.strip(),
                "metadata": {"method": method_upper, "path": path, "tags": tags},
            })

    return chunks


def _format_schema(schema: dict, definitions: dict, depth: int = 0) -> list[str]:
    """Recursively format a JSON schema into readable lines."""
    if not schema:
        return []

    # Resolve $ref
    ref = schema.get("$ref", "")
    if ref:
        ref_name = ref.split("/")[-1]
        resolved = definitions.get(ref_name, {})
        return _format_schema(resolved, definitions, depth)

    lines = []
    schema_type = schema.get("type", "object")

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "object")
            prop_desc = prop_schema.get("description", "")
            req_str = "required" if prop_name in required else "optional"
            indent = "  " * depth
            lines.append(f"{indent}{prop_name} ({prop_type}, {req_str}): {prop_desc}")
            if prop_type == "object" and depth < 2:
                lines.extend(_format_schema(prop_schema, definitions, depth + 1))
    elif schema_type == "array":
        items = schema.get("items", {})
        lines.append("  " * depth + "[array of:]")
        lines.extend(_format_schema(items, definitions, depth + 1))

    return lines


def upload_to_zeroentropy(chunks: list[dict]):
    """Upload chunked documents to ZeroEntropy."""
    client = httpx.Client(
        headers={"Authorization": f"Bearer {ZE_API_KEY}"},
        timeout=30.0,
    )
    for i, chunk in enumerate(chunks):
        resp = client.post(
            f"{ZE_BASE_URL}/documents",
            json={
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            },
        )
        resp.raise_for_status()
        if (i + 1) % 50 == 0:
            print(f"Uploaded {i + 1}/{len(chunks)} documents")

    print(f"Done. Uploaded {len(chunks)} documents total.")


def main():
    if not ZE_BASE_URL or not ZE_API_KEY:
        print("Error: Set ZEROENTROPY_BASE_URL and ZEROENTROPY_API_KEY env vars")
        sys.exit(1)

    spec = fetch_openapi_spec(TRIPLETEX_DOCS_URL)
    print(f"Spec has {len(spec.get('paths', {}))} paths")

    chunks = chunk_by_endpoint(spec)
    print(f"Created {len(chunks)} endpoint chunks")

    # Save locally for inspection
    with open("scripts/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("Saved chunks to scripts/chunks.json")

    upload_to_zeroentropy(chunks)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the script locally (manual)**

Run: `cd tripletex && python -c "from scripts.index_api import fetch_openapi_spec, chunk_by_endpoint; spec = fetch_openapi_spec('https://kkpqfuj-amager.tripletex.dev/v2-docs/'); print(f'Paths: {len(spec.get(\"paths\", {}))}')"

This verifies we can fetch and parse the spec. The ZeroEntropy upload part requires real credentials and should be tested manually.

- [ ] **Step 3: Commit**

```bash
cd tripletex && git add scripts/__init__.py scripts/index_api.py .gitignore && git commit -m "feat: OpenAPI spec indexing script for ZeroEntropy"
```

---

### Task 10: Dockerfile & Deployment Config

**Files:**
- Create: `tripletex/Dockerfile`

- [ ] **Step 1: Create Dockerfile**

`tripletex/Dockerfile`:
```dockerfile
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY scripts/ scripts/

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Verify Docker build**

Run: `cd tripletex && docker build -t tripletex-agent .`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
cd tripletex && git add Dockerfile && git commit -m "feat: Dockerfile for Cloud Run deployment"
```

---

### Task 11: Integration Smoke Test

**Files:**
- Create: `tripletex/tests/test_integration.py`

A test that exercises the full pipeline with mocked external services (LLM, ZeroEntropy, Tripletex API).

- [ ] **Step 1: Write integration test**

`tripletex/tests/test_integration.py`:
```python
"""Integration test: full pipeline with mocked external services."""

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.planner import ExecutionPlan, PlanStep, PlanResult


@pytest.fixture
def client():
    return TestClient(app)


def test_create_employee_full_pipeline(client):
    """Test the full pipeline for a 'create employee' task."""
    mock_llm = MagicMock()
    mock_retrieval = MagicMock()
    mock_tx = MagicMock()

    plan_result = PlanResult(
        search_query="create employee Ola Nordmann with email and administrator role",
        execution_plan=ExecutionPlan(steps=[
            PlanStep(
                step=1, method="POST", path="/employee",
                body={"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org"},
                capture={"emp_id": "value.id"},
            ),
        ]),
        api_specs=["Endpoint: POST /employee\nRequired: firstName, lastName, email\nOptional: roles"],
    )

    mock_tx.post.return_value = {"value": {"id": 1, "firstName": "Ola", "lastName": "Nordmann"}}

    with patch("src.main._create_llm", return_value=mock_llm), \
         patch("src.main._create_retrieval", return_value=mock_retrieval), \
         patch("src.main._create_tx_client", return_value=mock_tx), \
         patch("src.main.Planner") as MockPlanner:
        MockPlanner.return_value.plan.return_value = plan_result
        MockPlanner.return_value.create_retry_step = MagicMock()

        response = client.post("/solve", json={
            "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://tx-proxy.ainm.no/v2",
                "session_token": "test-token-abc",
            },
        })

        assert response.status_code == 200
        assert response.json() == {"status": "completed"}

        # Verify the Tripletex API was called correctly
        mock_tx.post.assert_called_once()
        call_args = mock_tx.post.call_args
        assert call_args[0][0] == "/employee"
        assert call_args[1]["json"]["firstName"] == "Ola"


def test_search_then_delete_pipeline(client):
    """Test a delete task: search first, then delete."""
    mock_llm = MagicMock()
    mock_retrieval = MagicMock()
    mock_tx = MagicMock()

    plan_result = PlanResult(
        search_query="delete travel expense report",
        execution_plan=ExecutionPlan(steps=[
            PlanStep(step=1, method="GET", path="/travelExpense", params={"fields": "id", "count": 100}, capture={"eid": "values.0.id"}),
            PlanStep(step=2, method="DELETE", path="/travelExpense/$eid"),
        ]),
        api_specs=["Endpoint: GET /travelExpense\nEndpoint: DELETE /travelExpense/{id}"],
    )

    mock_tx.get.return_value = {"values": [{"id": 55}]}
    mock_tx.delete.return_value = None

    with patch("src.main._create_llm", return_value=mock_llm), \
         patch("src.main._create_retrieval", return_value=mock_retrieval), \
         patch("src.main._create_tx_client", return_value=mock_tx), \
         patch("src.main.Planner") as MockPlanner:
        MockPlanner.return_value.plan.return_value = plan_result
        MockPlanner.return_value.create_retry_step = MagicMock()

        response = client.post("/solve", json={
            "prompt": "Slett reiseregningen",
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://tx-proxy.ainm.no/v2",
                "session_token": "test-token",
            },
        })

        assert response.status_code == 200
        assert response.json() == {"status": "completed"}
        mock_tx.delete.assert_called_once_with("/travelExpense/55")
```

- [ ] **Step 2: Run integration tests**

Run: `cd tripletex && uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run all tests**

Run: `cd tripletex && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd tripletex && git add tests/test_integration.py && git commit -m "feat: integration smoke tests for full pipeline"
```

---

### Task 12: Final Wiring & Local Test

- [ ] **Step 1: Sync dependencies**

Run: `cd tripletex && uv sync`
Expected: All dependencies installed, `uv.lock` updated

- [ ] **Step 2: Run full test suite**

Run: `cd tripletex && uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Start local server**

Run: `cd tripletex && uv run uvicorn src.main:app --reload --port 8000`
Expected: Server starts on http://localhost:8000

- [ ] **Step 4: Test with curl (manual)**

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Opprett en ansatt med navn Test Person, test@example.com",
    "files": [],
    "tripletex_credentials": {
      "base_url": "https://tx-proxy.ainm.no/v2",
      "session_token": "placeholder"
    }
  }'
```

Expected: Returns `{"status": "completed"}` (will fail on actual API calls without real credentials, but endpoint should respond)

- [ ] **Step 5: Commit all remaining changes**

```bash
cd tripletex && git add -A && git commit -m "chore: sync dependencies and finalize project setup"
```
