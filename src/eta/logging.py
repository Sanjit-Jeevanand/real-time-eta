from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import structlog
from structlog.types import EventDict, Processor

if TYPE_CHECKING:
    from typing import TextIO

__all__ = [
    "bind_request_id",
    "configure_logging",
    "current_request_id",
    "get_logger",
    "request_context",
]

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

_configured = False


def _add_request_id(_: Any, __: str, event_dict: EventDict) -> EventDict:
    rid = _request_id.get()
    if rid is not None:
        event_dict["request_id"] = rid
    return event_dict


def _add_service_context(_: Any, __: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("service", "eta")
    return event_dict


def configure_logging(
    level: str | int = "INFO",
    *,
    json_output: bool | None = None,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    global _configured
    if _configured and not force:
        return

    out = stream if stream is not None else sys.stderr
    if json_output is None:
        json_output = not out.isatty()

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        _add_service_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(out)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_request_id(request_id: str | None = None) -> str:
    rid = request_id or uuid.uuid4().hex[:16]
    _request_id.set(rid)
    return rid


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    token = _request_id.set(request_id or uuid.uuid4().hex[:16])
    try:
        rid = _request_id.get()
        assert rid is not None
        yield rid
    finally:
        _request_id.reset(token)
