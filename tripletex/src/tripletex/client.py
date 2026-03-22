"""
HTTP client wrapper that logs every API call in full detail.

Works as a drop-in for ``requests`` — same .get()/.post()/.put() etc. interface,
but every call is automatically logged with method, URL, params, headers (auth
redacted), request body, response body, status code, and duration.

See README.md for usage examples.
"""

from __future__ import annotations

import json as json_module
import logging
import time
import uuid
from typing import Any

import requests

from tripletex.my_log import request_id_ctx, unique_request_id_ctx

logger = logging.getLogger("logged_client")

# Headers whose values should never appear in logs.
_REDACTED_HEADERS = frozenset({"authorization", "cookie", "x-api-key", "x-session-token"})


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with sensitive values replaced."""
    return {
        k: "[REDACTED]" if k.lower() in _REDACTED_HEADERS else v
        for k, v in headers.items()
    }


def _try_parse_json(text: str) -> Any:
    """Attempt to parse *text* as JSON; return the raw string on failure."""
    try:
        return json_module.loads(text)
    except (json_module.JSONDecodeError, ValueError):
        return text


def _format_body(resp: requests.Response) -> Any:
    """Extract the response body in the most useful form for logging."""
    if not resp.content:
        return None
    return _try_parse_json(resp.text)


class LoggedHTTPClient:
    """A ``requests.Session`` wrapper that logs every API call in full detail.

    Every call produces two log entries sharing the same ``api_call_id``:
      1. **REQUEST**  – method, url, params, headers, body
      2. **RESPONSE** – status, headers, body, duration

    Log levels: INFO for 2xx, WARNING for 4xx, ERROR for 5xx / exceptions.
    """

    def __init__(
        self,
        base_url: str = "",
        auth: tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if auth:
            self.session.auth = auth
        if headers:
            self.session.headers.update(headers)

    # -- context manager --------------------------------------------------

    def __enter__(self) -> LoggedHTTPClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.session.close()

    # -- public HTTP verbs ------------------------------------------------

    def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._request("PUT", endpoint, **kwargs)

    def patch(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._request("PATCH", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs: Any) -> requests.Response:
        return self._request("DELETE", endpoint, **kwargs)

    # -- internals --------------------------------------------------------

    def _build_url(self, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        url = self._build_url(endpoint)
        call_id = str(uuid.uuid4())[:8]  # short id, enough for correlation
        kwargs.setdefault("timeout", self.timeout)
        request_id = request_id_ctx.get("-")
        unique_request_id = unique_request_id_ctx.get("-")

        # ---- extract what we're about to send ----------------------------
        request_params = kwargs.get("params")
        request_body: Any = kwargs.get("json") or kwargs.get("data")
        # If data= is bytes/str keep it as-is for logging; json= is already a dict.

        merged_headers = {**self.session.headers, **kwargs.get("headers", {})}

        # ---- log the REQUEST ---------------------------------------------
        logger.info(
            "[%s] --> %s %s",
            call_id,
            method,
            url,
            extra={
                "api_call_id": call_id,
                "api_event": "request",
                "api_method": method,
                "api_url": url,
                "api_endpoint": endpoint,
                "api_request_id": request_id,
                "api_unique_request_id": unique_request_id,
                "api_request_params": request_params,
                "api_request_body": request_body,
                "api_request_headers": _redact_headers(merged_headers),
            },
        )

        # ---- execute -----------------------------------------------------
        start = time.perf_counter()
        try:
            resp = self.session.request(method, url, **kwargs)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "[%s] !!! %s %s raised %s after %.1fms",
                call_id,
                method,
                url,
                type(exc).__name__,
                duration_ms,
                extra={
                    "api_call_id": call_id,
                    "api_event": "error",
                    "api_method": method,
                    "api_url": url,
                    "api_endpoint": endpoint,
                    "api_request_id": request_id,
                    "api_unique_request_id": unique_request_id,
                    "api_error": str(exc),
                    "api_error_type": type(exc).__name__,
                    "api_duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        # ---- pick log level by status code -------------------------------
        if resp.status_code < 400:
            log_fn = logger.info
        elif resp.status_code < 500:
            log_fn = logger.warning
        else:
            log_fn = logger.error

        # ---- log the RESPONSE --------------------------------------------
        log_fn(
            "[%s] <-- %s %s  %d (%s) %.1fms",
            call_id,
            method,
            url,
            resp.status_code,
            resp.reason,
            duration_ms,
            extra={
                "api_call_id": call_id,
                "api_event": "response",
                "api_method": method,
                "api_url": url,
                "api_endpoint": endpoint,
                "api_request_id": request_id,
                "api_unique_request_id": unique_request_id,
                "api_status_code": resp.status_code,
                "api_status_reason": resp.reason,
                "api_response_headers": dict(resp.headers),
                "api_response_body": _format_body(resp),
                "api_duration_ms": duration_ms,
            },
        )

        return resp
