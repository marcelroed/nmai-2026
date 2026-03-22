from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

from starlette.datastructures import MutableHeaders
from starlette.requests import Request as StarletteRequest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
unique_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "unique_request_id", default="-"
)
request_method_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_method", default="-"
)
request_path_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_path", default="-"
)
client_ip_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "client_ip", default="-"
)

_RESERVED_LOG_RECORD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {
    "message",
    "asctime",
}
_CONTEXT_ATTRS = {
    "request_id",
    "unique_request_id",
    "request_method",
    "request_path",
    "client_ip",
}
_CONFIGURE_LOGGING_LOCK = threading.Lock()
_LOGGING_CONFIGURED = False
_RUNTIME_ENV_VAR = "TRIPLETEX_RUNTIME"
_LOG_LEVEL_ENV_VAR = "LOG_LEVEL"
_NO_COLOR_ENV_VAR = "NO_COLOR"
_FORCE_COLOR_ENV_VAR = "FORCE_COLOR"
_CONTAINER_RUNTIME = "container"
_LOCAL_LOG_PATH = Path("data") / "app.log"
_LOG_VERSION = "5"
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_ANSI_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m",
}


class RequestContextFilter(logging.Filter):
    """Inject request-scoped fields into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")
        record.unique_request_id = unique_request_id_ctx.get("-")
        record.request_method = request_method_ctx.get("-")
        record.request_path = request_path_ctx.get("-")
        record.client_ip = client_ip_ctx.get("-")
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", "-")
        short_request_id = request_id[:5] if request_id else "-"
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "log_version": _LOG_VERSION,
            "level": record.levelname,
            "logger": record.name,
            "message": f"[{short_request_id}] {record.getMessage()}",
            "request_id": request_id,
            "unique_request_id": getattr(record, "unique_request_id", "-"),
            "request_method": getattr(record, "request_method", "-"),
            "request_path": getattr(record, "request_path", "-"),
            "client_ip": getattr(record, "client_ip", "-"),
            "module": record.module,
            "filename": record.filename,
            "pathname": record.pathname,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "process_name": record.processName,
            "thread": record.thread,
            "thread_name": record.threadName,
        }

        task_name = getattr(record, "taskName", None)
        if task_name:
            payload["task_name"] = task_name

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_ATTRS and key not in _CONTEXT_ATTRS
        }
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class LocalConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()
        self._use_color = _supports_ansi_color()

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        request_id = getattr(record, "request_id", "-")
        short_request_id = request_id[:5] if request_id and request_id != "-" else "-"
        level = f"{record.levelname:<5}"
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno)
            if color:
                level = f"{color}{level}{_ANSI_RESET}"
        return f"{timestamp} {level} [{short_request_id}] {record.getMessage()}"


_logger = logging.getLogger("app")


def _is_local_runtime() -> bool:
    return os.environ.get(_RUNTIME_ENV_VAR, "").lower() != _CONTAINER_RUNTIME


def _supports_ansi_color() -> bool:
    if os.environ.get(_NO_COLOR_ENV_VAR) is not None:
        return False

    force_color = os.environ.get(_FORCE_COLOR_ENV_VAR)
    if force_color == "0":
        return False
    if force_color:
        return True

    return (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and os.environ.get("TERM", "").lower() != "dumb"
    )


def _get_console_log_level(is_local_runtime: bool) -> str:
    configured_level = os.environ.get(_LOG_LEVEL_ENV_VAR, "").upper()
    if configured_level in _VALID_LOG_LEVELS:
        return configured_level
    return "INFO" if is_local_runtime else "DEBUG"


def _build_logging_config() -> dict[str, Any]:
    is_local_runtime = _is_local_runtime()
    console_level = _get_console_log_level(is_local_runtime)
    console_formatter = "local_console" if is_local_runtime else "json"
    shared_handlers = ["console"]
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": console_level,
            "formatter": console_formatter,
            "filters": ["request_context"],
            "stream": "ext://sys.stdout",
        }
    }

    if is_local_runtime:
        _LOCAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filters": ["request_context"],
            "filename": str(_LOCAL_LOG_PATH),
            "encoding": "utf-8",
        }
        shared_handlers.append("file")

    return cast(
        dict[str, Any],
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_context": {
                    "()": RequestContextFilter,
                }
            },
            "formatters": {
                "json": {
                    "()": JsonFormatter,
                },
                "local_console": {
                    "()": LocalConsoleFormatter,
                },
            },
            "handlers": handlers,
            "root": {
                "level": "INFO",
                "handlers": shared_handlers,
            },
            "loggers": {
                "app": {
                    "level": "DEBUG",
                    "handlers": shared_handlers,
                    "propagate": False,
                },
                "logged_client": {
                    "level": "DEBUG",
                    "handlers": shared_handlers,
                    "propagate": False,
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": shared_handlers,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": "INFO",
                    "handlers": shared_handlers,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": [],
                    "propagate": False,
                },
            },
        },
    )


def configure_logging() -> logging.Logger:
    global _LOGGING_CONFIGURED

    with _CONFIGURE_LOGGING_LOCK:
        if _LOGGING_CONFIGURED:
            return _logger

        config = _build_logging_config()
        logging.config.dictConfig(config)
        _LOGGING_CONFIGURED = True
        return _logger


def _sanitize_headers(headers: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return headers


def _body_preview(body: bytes) -> dict[str, Any]:
    try:
        preview = body.decode("utf-8")
    except UnicodeDecodeError:
        preview = body.hex()

    return {
        "size_bytes": len(body),
        "preview": preview,
    }


def _request_line(scope: Scope) -> str:
    method = scope.get("method", "GET")
    raw_path = scope.get("raw_path", scope.get("path", "").encode("utf-8"))
    target = raw_path.decode("latin-1")
    query_string = scope.get("query_string", b"")
    if query_string:
        target = f"{target}?{query_string.decode('latin-1')}"
    http_version = scope.get("http_version", "1.1")
    return f"{method} {target} HTTP/{http_version}"


class RequestContextLogMiddleware:
    """
    Pure ASGI middleware so request-scoped ContextVars behave cleanly.

    It:
      - picks or generates a request id
      - captures request metadata/body for logging
      - injects the request id into all logs in this request
      - adds X-Request-ID to the response
      - logs the response status and duration
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._get_or_create_request_id(scope)
        unique_request_id = str(uuid.uuid4())
        scope["request_id"] = request_id
        scope["unique_request_id"] = unique_request_id

        request_messages: list[Message] = []
        request_body = b""
        more_body = True
        while more_body:
            message = await receive()
            request_messages.append(message)
            if message["type"] == "http.request":
                request_body += message.get("body", b"")
                more_body = message.get("more_body", False)
            else:
                more_body = False

        replay_index = 0

        async def replay_receive() -> Message:
            nonlocal replay_index
            if replay_index < len(request_messages):
                message = request_messages[replay_index]
                replay_index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        request = StarletteRequest(scope)
        client_host = request.client.host if request.client else "-"
        client_port = request.client.port if request.client else None

        tokens = [
            (request_id_ctx, request_id_ctx.set(request_id)),
            (unique_request_id_ctx, unique_request_id_ctx.set(unique_request_id)),
            (request_method_ctx, request_method_ctx.set(request.method)),
            (request_path_ctx, request_path_ctx.set(request.url.path)),
            (client_ip_ctx, client_ip_ctx.set(client_host)),
        ]

        status_code: int | None = None
        response_headers_for_log: list[tuple[str, str]] = []
        start = time.perf_counter()

        _logger.info(
            "request.start",
            extra={
                "event": "request.start",
                "request_line": _request_line(scope),
                "url": str(request.url),
                "http_version": scope.get("http_version"),
                "scheme": scope.get("scheme"),
                "server": scope.get("server"),
                "client": {"host": client_host, "port": client_port},
                "headers": _sanitize_headers(list(request.headers.items())),
                "query_string": scope.get("query_string", b"").decode("latin-1"),
                "body": _body_preview(request_body),
            },
        )

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_headers_for_log

            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                response_headers_for_log = _sanitize_headers(list(headers.items()))

            await send(message)

        try:
            await self.app(scope, replay_receive, send_wrapper)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            _logger.exception(
                "request.error",
                extra={
                    "event": "request.error",
                    "duration_ms": duration_ms,
                    "request_line": _request_line(scope),
                },
            )
            raise
        else:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            _logger.info(
                "request.end",
                extra={
                    "event": "request.end",
                    "duration_ms": duration_ms,
                    "status_code": status_code,
                    "response_headers": response_headers_for_log,
                },
            )
        finally:
            for ctx_var, token in reversed(tokens):
                ctx_var.reset(token)

    @staticmethod
    def _get_or_create_request_id(scope: Scope) -> str:
        for raw_key, raw_value in scope.get("headers", []):
            if raw_key.decode("latin-1").lower() == "x-request-id":
                return raw_value.decode("latin-1")
        return str(uuid.uuid4())
