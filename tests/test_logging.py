from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import Iterator

import pytest

from eta.logging import (
    bind_request_id,
    configure_logging,
    current_request_id,
    get_logger,
    request_context,
)


@pytest.fixture
def sink() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    configure_logging(level="DEBUG", json_output=True, stream=buf, force=True)
    yield buf
    logging.getLogger().handlers.clear()


def _lines(sink: io.StringIO) -> list[dict[str, object]]:
    text = sink.getvalue().strip()
    lines = [json.loads(line) for line in text.split("\n") if line.strip()]
    sink.truncate(0)
    sink.seek(0)
    return lines


def test_emits_json_with_standard_fields(sink: io.StringIO) -> None:
    get_logger("eta.test").info("data_filtered", filter_name="negative_duration", dropped=1423)
    (rec,) = _lines(sink)
    assert rec["event"] == "data_filtered"
    assert rec["filter_name"] == "negative_duration"
    assert rec["dropped"] == 1423
    assert rec["level"] == "info"
    assert rec["service"] == "eta"
    assert "timestamp" in rec


def test_request_id_absent_when_unset(sink: io.StringIO) -> None:
    get_logger("eta.test").info("no_context")
    (rec,) = _lines(sink)
    assert "request_id" not in rec


def test_request_context_attaches_and_restores(sink: io.StringIO) -> None:
    log = get_logger("eta.test")
    with request_context("abc123") as rid:
        assert rid == "abc123"
        log.info("inside")
    log.info("outside")

    inside, outside = _lines(sink)
    assert inside["request_id"] == "abc123"
    assert "request_id" not in outside
    assert current_request_id() is None


def test_nested_contexts_restore_the_outer_id() -> None:
    with request_context("run") as outer:
        with request_context("event") as inner:
            assert inner == "event"
            assert current_request_id() == "event"
        assert current_request_id() == outer == "run"


def test_stdlib_library_logs_carry_the_request_id(sink: io.StringIO) -> None:
    with request_context("req-42"):
        logging.getLogger("uvicorn.error").warning("third party line")
    (rec,) = _lines(sink)
    assert rec["request_id"] == "req-42"
    assert rec["event"] == "third party line"
    assert rec["level"] == "warning"


def test_ids_do_not_leak_between_concurrent_tasks() -> None:
    seen: dict[str, str | None] = {}

    async def handler(name: str, delay: float) -> None:
        with request_context(name):
            await asyncio.sleep(delay)
            seen[name] = current_request_id()

    async def main() -> None:
        await asyncio.gather(handler("a", 0.02), handler("b", 0.01))

    asyncio.run(main())
    assert seen == {"a": "a", "b": "b"}


def test_bind_generates_an_id_when_none_supplied() -> None:
    rid = bind_request_id()
    assert rid and len(rid) == 16
    assert current_request_id() == rid
    bind_request_id(None)


def test_configure_is_idempotent(sink: io.StringIO) -> None:
    before = len(logging.getLogger().handlers)
    configure_logging(level="DEBUG", json_output=True)
    configure_logging(level="DEBUG", json_output=True)
    assert len(logging.getLogger().handlers) == before

    get_logger("eta.test").info("once")
    assert len(_lines(sink)) == 1


def test_exception_info_is_rendered(sink: io.StringIO) -> None:
    log = get_logger("eta.test")
    try:
        raise ValueError("zone matrix missing")
    except ValueError:
        log.exception("lookup_failed")
    (rec,) = _lines(sink)
    assert rec["event"] == "lookup_failed"
    assert "ValueError: zone matrix missing" in str(rec.get("exception", ""))
