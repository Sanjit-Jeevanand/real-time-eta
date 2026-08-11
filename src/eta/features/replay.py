from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

from eta.features.congestion import BUCKET_MINUTES, STATE_COLUMNS
from eta.features.context import CONGESTION_WINDOWS, congestion_key
from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

__all__ = [
    "CLOCK_KEY",
    "DictStore",
    "ReplayStream",
    "Store",
    "replay_to_store",
]

log = get_logger(__name__)

CLOCK_KEY = "cong:clock"


@runtime_checkable
class Store(Protocol):
    def mset(self, mapping: Mapping[str, float | str]) -> None: ...
    def mget(self, keys: Sequence[str]) -> list[float | str | None]: ...


@dataclass(slots=True)
class DictStore:
    data: dict[str, float | str] = field(default_factory=dict)
    writes: int = 0

    def mset(self, mapping: Mapping[str, float | str]) -> None:
        self.data.update(mapping)
        self.writes += len(mapping)

    def mget(self, keys: Sequence[str]) -> list[float | str | None]:
        return [self.data.get(k) for k in keys]


@dataclass(slots=True)
class _Window:
    span: dt.timedelta
    buckets: deque[tuple[dt.datetime, tuple[float, float, float]]] = field(default_factory=deque)
    totals: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def add(self, bucket: dt.datetime, values: tuple[float, float, float]) -> None:
        self.buckets.append((bucket, values))
        for i in range(3):
            self.totals[i] += values[i]

    def evict(self, now: dt.datetime) -> None:
        cutoff = now - self.span
        while self.buckets and self.buckets[0][0] <= cutoff:
            _, values = self.buckets.popleft()
            for i in range(3):
                self.totals[i] -= values[i]


@dataclass(slots=True)
class ReplayStream:
    store: Store
    windows: Sequence[int] = CONGESTION_WINDOWS
    _state: dict[tuple[int, int], _Window] = field(default_factory=dict)
    _clock: dt.datetime | None = None
    buckets_seen: int = 0

    def _window(self, zone: int, minutes: int) -> _Window:
        key = (zone, minutes)
        if key not in self._state:
            self._state[key] = _Window(span=dt.timedelta(minutes=minutes))
        return self._state[key]

    def advance(
        self, bucket: dt.datetime, completions: Mapping[int, tuple[float, float, float]]
    ) -> None:
        if self._clock is not None and bucket < self._clock:
            msg = f"replay went backwards: {bucket} after {self._clock}"
            raise ValueError(msg)
        self._clock = bucket
        self.buckets_seen += 1

        touched: set[int] = set(completions)
        for zone, values in completions.items():
            for w in self.windows:
                self._window(zone, w).add(bucket, values)

        payload: dict[str, float | str] = {CLOCK_KEY: bucket.isoformat()}
        for (zone, w), window in self._state.items():
            window.evict(bucket)
            if zone not in touched and not window.buckets:
                continue
            for i, name in enumerate(STATE_COLUMNS):
                kind = {
                    "completed": "completed",
                    "distance_m": "distance",
                    "duration_s": "duration",
                }[name]
                payload[congestion_key(kind, zone, w)] = window.totals[i]
        self.store.mset(payload)

    def snapshot(self, zone: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for w in self.windows:
            window = self._window(zone, w)
            for i, name in enumerate(STATE_COLUMNS):
                kind = {
                    "completed": "completed",
                    "distance_m": "distance",
                    "duration_s": "duration",
                }[name]
                out[congestion_key(kind, zone, w)] = window.totals[i]
        return out


def _event_buckets(
    enriched_glob: Path,
) -> Iterator[tuple[dt.datetime, dict[int, tuple[float, float, float]]]]:
    from eta.features.congestion import METRES_PER_MILE

    frame = (
        pl.scan_parquet(enriched_glob)
        .select(
            pl.col("dropoff_datetime")
            .dt.truncate(f"{BUCKET_MINUTES}m")
            .dt.cast_time_unit("us")
            .alias("bucket"),
            pl.col("pu_zone").alias("zone"),
            (pl.col("trip_miles") * METRES_PER_MILE).alias("distance_m"),
            pl.col("trip_duration_s").alias("duration_s"),
        )
        .filter(pl.col("duration_s") > 0)
        .group_by("zone", "bucket")
        .agg(
            pl.len().alias("completed"),
            pl.col("distance_m").sum().alias("distance_m"),
            pl.col("duration_s").sum().alias("duration_s"),
        )
        .sort("bucket", "zone")
        .collect(engine="streaming")
    )

    current: dt.datetime | None = None
    batch: dict[int, tuple[float, float, float]] = {}
    for row in frame.iter_rows(named=True):
        bucket = row["bucket"]
        if current is not None and bucket != current:
            yield current, batch
            batch = {}
        current = bucket
        batch[int(row["zone"])] = (
            float(row["completed"]),
            float(row["distance_m"]),
            float(row["duration_s"]),
        )
    if current is not None:
        yield current, batch


def replay_to_store(
    enriched_glob: Path, store: Store, windows: Sequence[int] = CONGESTION_WINDOWS
) -> ReplayStream:
    stream = ReplayStream(store=store, windows=windows)
    for bucket, completions in _event_buckets(enriched_glob):
        stream.advance(bucket, completions)
    log.info(
        "replay_complete",
        buckets=stream.buckets_seen,
        zones=len({z for z, _ in stream._state}),
        windows=list(windows),
    )
    return stream
