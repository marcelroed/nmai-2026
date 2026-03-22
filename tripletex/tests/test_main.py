from __future__ import annotations

from base64 import b64encode
import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tripletex.client import LoggedHTTPClient
from tripletex.api_log_viewer import LogCollector
from tripletex.herman_tasks.utils import TripletexCredentials
from tripletex.main import (
    InputFile,
    SolveRequest,
    attachment_to_text,
    get_task_attachment_text,
    prefetch_tripletex_data,
    serve,
)
from tripletex.my_log import unique_request_id_ctx
from tripletex.parsers.task1_create_employee import CreateEmployee
from tripletex.parsers.task21_onboarding_from_offer_pdf import OnboardingFromOfferPDF


def _dummy_creds() -> TripletexCredentials:
    return TripletexCredentials(base_url="https://example.invalid", auth=(0, "token"))


def test_attachment_to_text_extracts_pdf_text():
    pdf_bytes = Path("data/files/raw/tilbudsbrev_nb_01.pdf").read_bytes()
    input_file = InputFile(
        filename="tilbudsbrev_nb_01.pdf",
        content_base64=b64encode(pdf_bytes).decode("ascii"),
        mime_type="application/pdf",
    )

    attachment_text = attachment_to_text(input_file)

    assert "TILBUD OM STILLING" in attachment_text
    assert "Kristian Ødegård" in attachment_text


def test_get_task_attachment_text_requires_exactly_one_attachment():
    with pytest.raises(ValueError, match="requires exactly one attachment"):
        get_task_attachment_text(task="Task 21", files=[])


def test_prefetch_tripletex_data_runs_with_configured_workers_and_propagates_context(
    monkeypatch: pytest.MonkeyPatch,
):
    from tripletex import main

    active = 0
    max_active = 0
    started = 0
    total_calls = 0
    seen_request_ids: list[str] = []
    lock = threading.Lock()
    release = threading.Event()
    worker_target = min(
        main.TRIPLETEX_PREFETCH_WORKERS,
        len(main.TRIPLETEX_PREFETCH_ENDPOINTS)
        + len(main.TRIPLETEX_PREFETCH_ENDPOINTS_WITH_PARAMS),
    )
    expected_request_id = "req-prefetch"

    def record_call() -> None:
        nonlocal active, max_active, started, total_calls
        with lock:
            active += 1
            started += 1
            total_calls += 1
            max_active = max(max_active, active)
            seen_request_ids.append(unique_request_id_ctx.get("-"))
            if started >= worker_target:
                release.set()

        assert release.wait(timeout=1), "prefetch did not fill the worker pool"

        with lock:
            active -= 1

    monkeypatch.setattr(main, "get_tripletex_data", lambda **_kwargs: record_call())

    token = unique_request_id_ctx.set(expected_request_id)
    try:
        prefetch_tripletex_data(client_factory=_dummy_creds().to_client)
    finally:
        unique_request_id_ctx.reset(token)

    expected_calls = len(main.TRIPLETEX_PREFETCH_ENDPOINTS) + len(
        main.TRIPLETEX_PREFETCH_ENDPOINTS_WITH_PARAMS
    )
    assert total_calls == expected_calls
    assert max_active == worker_target
    assert seen_request_ids == [expected_request_id] * expected_calls


def test_log_collector_only_keeps_calls_for_its_request_id():
    collector = LogCollector(request_id="req-1")

    collector.emit(
        logging.makeLogRecord(
            {
                "name": "logged_client",
                "levelno": logging.INFO,
                "levelname": "INFO",
                "msg": "request",
                "args": (),
                "api_event": "request",
                "api_call_id": "call-1",
                "api_method": "GET",
                "api_url": "https://example.invalid/employee",
                "api_endpoint": "/employee",
                "api_request_id": "outer-1",
                "api_unique_request_id": "req-1",
                "api_request_params": None,
                "api_request_body": None,
                "api_request_headers": {},
            }
        )
    )
    collector.emit(
        logging.makeLogRecord(
            {
                "name": "logged_client",
                "levelno": logging.INFO,
                "levelname": "INFO",
                "msg": "response",
                "args": (),
                "api_event": "response",
                "api_call_id": "call-1",
                "api_request_id": "outer-1",
                "api_unique_request_id": "req-1",
                "api_status_code": 200,
                "api_status_reason": "OK",
                "api_response_headers": {},
                "api_response_body": {"ok": True},
                "api_duration_ms": 12.3,
            }
        )
    )
    collector.emit(
        logging.makeLogRecord(
            {
                "name": "logged_client",
                "levelno": logging.INFO,
                "levelname": "INFO",
                "msg": "request",
                "args": (),
                "api_event": "request",
                "api_call_id": "call-2",
                "api_method": "GET",
                "api_url": "https://example.invalid/customer",
                "api_endpoint": "/customer",
                "api_request_id": "outer-2",
                "api_unique_request_id": "req-2",
                "api_request_params": None,
                "api_request_body": None,
                "api_request_headers": {},
            }
        )
    )
    collector.emit(
        logging.makeLogRecord(
            {
                "name": "logged_client",
                "levelno": logging.INFO,
                "levelname": "INFO",
                "msg": "response",
                "args": (),
                "api_event": "response",
                "api_call_id": "call-2",
                "api_request_id": "outer-2",
                "api_unique_request_id": "req-2",
                "api_status_code": 200,
                "api_status_reason": "OK",
                "api_response_headers": {},
                "api_response_body": {"ignored": True},
                "api_duration_ms": 45.6,
            }
        )
    )

    assert len(collector.calls) == 1
    assert collector.calls[0].call_id == "call-1"
    assert collector.calls[0].unique_request_id == "req-1"


def test_serve_calls_parse_and_solve_for_non_attachment_task(monkeypatch: pytest.MonkeyPatch):
    from tripletex import main

    calls: dict[str, object] = {}

    monkeypatch.setattr(main, "get_submission_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "prefetch_tripletex_data", lambda **_kwargs: None)
    monkeypatch.setattr(
        main.Classification,
        "classify",
        staticmethod(lambda _prompt: "Task 1"),
    )

    class FakeCollector:
        def install(self) -> FakeCollector:
            return self

        def uninstall(self) -> None:
            pass

        def save_to_disk(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(main, "LogCollector", lambda **_kwargs: FakeCollector())

    class Parsed:
        def solve(self, tripletex_client: LoggedHTTPClient) -> None:
            calls["tripletex_client"] = tripletex_client

        def __repr__(self) -> str:
            return "ParsedTask1()"

    def fake_parse(cls, prompt: str) -> Parsed:
        calls["prompt"] = prompt
        return Parsed()

    monkeypatch.setattr(CreateEmployee, "parse", classmethod(fake_parse))

    payload = SolveRequest(
        prompt="Create an employee",
        files=[],
        tripletex_credentials=_dummy_creds(),
    )
    request = SimpleNamespace(headers={"authorization": "Bearer DEBUG-test"})

    response = serve(payload, request)

    assert response == {"status": "completed"}
    assert calls["prompt"] == "Create an employee"
    assert calls["tripletex_client"].base_url == payload.tripletex_credentials.base_url
    assert calls["tripletex_client"].session.auth == (
        "0",
        payload.tripletex_credentials.auth[1],
    )


def test_serve_calls_parse_and_solve_for_pdf_attachment_task(
    monkeypatch: pytest.MonkeyPatch,
):
    from tripletex import main

    calls: dict[str, object] = {}

    monkeypatch.setattr(main, "get_submission_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "prefetch_tripletex_data", lambda **_kwargs: None)
    monkeypatch.setattr(
        main.Classification,
        "classify",
        staticmethod(lambda _prompt: "Task 21"),
    )

    class FakeCollector:
        def install(self) -> FakeCollector:
            return self

        def uninstall(self) -> None:
            pass

        def save_to_disk(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(main, "LogCollector", lambda **_kwargs: FakeCollector())

    class Parsed:
        def solve(self, tripletex_client: LoggedHTTPClient) -> None:
            calls["tripletex_client"] = tripletex_client

        def __repr__(self) -> str:
            return "ParsedTask21()"

    def fake_parse(cls, prompt: str, attachment: str) -> Parsed:
        calls["prompt"] = prompt
        calls["attachment"] = attachment
        return Parsed()

    monkeypatch.setattr(OnboardingFromOfferPDF, "parse", classmethod(fake_parse))

    pdf_bytes = Path("data/files/raw/tilbudsbrev_nb_01.pdf").read_bytes()
    payload = SolveRequest(
        prompt="Handle attached offer letter",
        files=[
            InputFile(
                filename="tilbudsbrev_nb_01.pdf",
                content_base64=b64encode(pdf_bytes).decode("ascii"),
                mime_type="application/pdf",
            )
        ],
        tripletex_credentials=_dummy_creds(),
    )
    request = SimpleNamespace(headers={"authorization": "Bearer DEBUG-test"})

    response = serve(payload, request)

    assert response == {"status": "completed"}
    assert calls["prompt"] == "Handle attached offer letter"
    assert "TILBUD OM STILLING" in str(calls["attachment"])
    assert "Kristian Ødegård" in str(calls["attachment"])
    assert calls["tripletex_client"].base_url == payload.tripletex_credentials.base_url
    assert calls["tripletex_client"].session.auth == (
        "0",
        payload.tripletex_credentials.auth[1],
    )
