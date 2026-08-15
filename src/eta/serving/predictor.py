"""Request-time prediction: lookup, fetch, assemble, infer, conformalise.

Each stage is timed separately and the breakdown ships on the response as
`X-Stage-Timings`. A single end-to-end number tells you that you missed the budget;
the breakdown tells you which stage to go and fix.

The online feature path is the *same* `OnlineContext` the parity gate checks against
batch (Phase 4.2), so what is served here is what was trained on -- that equivalence
is enforced by a blocking test, not by care.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.features.context import OnlineContext
from eta.features.registry import REGISTRY
from eta.logging import get_logger

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from eta.serving.store import FeatureStore

__all__ = ["Prediction", "Predictor", "StageTimer"]

log = get_logger(__name__)


@dataclass(slots=True)
class StageTimer:
    """Wall-clock per stage, in milliseconds."""

    stages: dict[str, float] = field(default_factory=dict)
    _start: float = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages[stage] = round((now - self._start) * 1000.0, 3)
        self._start = now

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 3)

    def header(self) -> str:
        return ", ".join(f"{k}={v:.2f}ms" for k, v in self.stages.items())


@dataclass(frozen=True, slots=True)
class Prediction:
    quantiles: dict[str, float]
    interval: tuple[float, float]
    degraded: bool
    timings: dict[str, float]
    total_ms: float


@dataclass(slots=True)
class Predictor:
    """Holds every artifact in memory; the request path does no file or disk I/O."""

    alphas: tuple[float, ...]
    feature_names: Sequence[str]
    route_lookup: dict[tuple[int, int], dict[str, float]]
    zone_lookup: dict[int, dict[str, float]]
    store: FeatureStore
    booster_predict: Any
    conformal_q: float = 0.0
    lo_idx: int = 0
    hi_idx: int = 4
    weather_defaults: dict[str, float] = field(default_factory=dict)

    def _redis_keys(self, pu_zone: int) -> list[str]:
        """Derived from the registry, so a new live feature cannot be forgotten.

        The registry declares two key families: per-zone congestion
        (`cong:<kind>:<zone>:<window>m`) and global weather (`wx:<name>`). Both are
        fetched in the same round trip.
        """
        return [t.format(zone=pu_zone) for t in REGISTRY.required_redis_keys()]

    @staticmethod
    def _split(values: dict[str, float | None]) -> tuple[dict[str, float], dict[str, float]]:
        """Route fetched keys to the two namespaces OnlineContext reads.

        Congestion stays keyed by the full Redis key, because `ctx.cong` rebuilds
        that exact string. Weather is keyed by the bare name after the `wx:` prefix,
        because `ctx.wx` looks up the column name. Getting this wrong is silent --
        every weather feature would read NaN and the model would still answer.
        """
        congestion: dict[str, float] = {}
        weather: dict[str, float] = {}
        for key, value in values.items():
            if value is None:
                continue
            if key.startswith("wx:"):
                weather[key[3:]] = value
            else:
                congestion[key] = value
        return congestion, weather

    def predict(self, pu_zone: int, do_zone: int, request_time: dt.datetime) -> Prediction:
        timer = StageTimer()
        with timer:
            route = self.route_lookup.get((pu_zone, do_zone), {})
            pu = self.zone_lookup.get(pu_zone, {})
            do = self.zone_lookup.get(do_zone, {})
            timer.mark("route_lookup")

            keys = self._redis_keys(pu_zone)
            fetched = self.store.fetch(keys)
            timer.mark("store_fetch")

            congestion, weather = self._split(fetched.values)
            ctx = OnlineContext(
                pu_zone,
                do_zone,
                request_time,
                route=route,
                pu=pu,
                do=do,
                congestion=congestion,
                weather=self.weather_defaults | weather,
            )
            row = REGISTRY.online_row(ctx, list(self.feature_names))
            x = np.asarray([[row[name] for name in self.feature_names]], dtype=np.float64)
            timer.mark("feature_assembly")

            raw: NDArray[np.float64] = np.asarray(self.booster_predict(x), dtype=np.float64)
            matrix = raw.reshape(1, -1)
            # Ordering is enforced at serve time too: a crossed row would produce a
            # promise below the median, which is worse than a slightly wide one.
            matrix = np.maximum.accumulate(matrix, axis=1)
            timer.mark("inference")

            lo = float(matrix[0, self.lo_idx] - self.conformal_q)
            hi = float(matrix[0, self.hi_idx] + self.conformal_q)
            quantiles = {
                f"p{round(a * 100)}": float(matrix[0, i]) for i, a in enumerate(self.alphas)
            }
            timer.mark("conformal")

        return Prediction(
            quantiles=quantiles,
            interval=(lo, hi),
            degraded=fetched.degraded,
            timings=timer.stages,
            total_ms=timer.total_ms,
        )
