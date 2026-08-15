"""Redis-backed feature store, and the fallback that keeps the service answering.

The degradation contract, stated plainly: **a Redis failure must never fail a
request.** Congestion state is an enhancement to the prediction, not a prerequisite
for one -- the model was trained with that state legitimately missing on cold-start
rows (Phase 4), so it already knows how to handle its absence.

What degradation does *not* do is substitute zeros. "No observation" is not
"observed zero trips", and a zero in a busy zone is a confident lie about traffic.
Missing stays missing, `degraded` is set on the response, and the caller can see
that the answer came from a thinner feature set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["FeatureStore", "RedisStore", "StoreResult"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StoreResult:
    values: dict[str, float | None]
    degraded: bool
    error: str | None = None


@dataclass(slots=True)
class RedisStore:
    """Thin wrapper doing exactly one pipelined round trip per request."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    socket_timeout_s: float = 0.05
    _client: Any = None

    def connect(self) -> RedisStore:
        import redis

        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            socket_timeout=self.socket_timeout_s,
            socket_connect_timeout=self.socket_timeout_s,
            decode_responses=True,
        )
        return self

    def mget(self, keys: Sequence[str]) -> list[float | str | None]:
        if self._client is None:
            msg = "RedisStore.connect() was not called"
            raise RuntimeError(msg)
        # One round trip for every key the registry declared -- not one per feature.
        raw: list[str | None] = self._client.mget(list(keys))
        return [None if v is None else float(v) for v in raw]

    def mset(self, mapping: Mapping[str, float | str]) -> None:
        if self._client is None:
            msg = "RedisStore.connect() was not called"
            raise RuntimeError(msg)
        self._client.mset(dict(mapping))


@dataclass(slots=True)
class FeatureStore:
    """Wraps any Store and turns its failures into degradation instead of errors."""

    backend: Any
    degrade_on_failure: bool = True
    _failures: int = 0
    _requests: int = 0

    def fetch(self, keys: Sequence[str]) -> StoreResult:
        self._requests += 1
        if not keys:
            return StoreResult(values={}, degraded=False)
        try:
            raw = self.backend.mget(list(keys))
        except Exception as exc:  # any backend failure degrades identically
            if not self.degrade_on_failure:
                raise
            self._failures += 1
            log.warning(
                "store_unavailable_degrading",
                error=type(exc).__name__,
                keys=len(keys),
                failures=self._failures,
            )
            # Missing, not zero. The model has seen missing congestion state before.
            return StoreResult(values=dict.fromkeys(keys), degraded=True, error=type(exc).__name__)

        values: dict[str, float | None] = {}
        for key, value in zip(keys, raw, strict=True):
            values[key] = None if value is None else float(value)
        # A cold zone with no state yet is a legitimate miss, not a failure.
        return StoreResult(values=values, degraded=False)

    @property
    def failure_rate(self) -> float:
        return self._failures / self._requests if self._requests else 0.0
