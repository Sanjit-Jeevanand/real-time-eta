from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from eta.features.context import BatchContext, congestion_key
from eta.features.pipeline import assemble, online_context
from eta.features.registry import REGISTRY
from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from eta.features.context import StaticTables

__all__ = [
    "EXACT_DTYPES",
    "FLOAT_TOLERANCE",
    "ParityReport",
    "check_parity",
]

log = get_logger(__name__)

FLOAT_TOLERANCE: Final = 1e-5


EXACT_DTYPES: Final = (
    pl.Boolean,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)


def _is_exact(dtype: object) -> bool:
    return dtype in EXACT_DTYPES


@dataclass(frozen=True, slots=True)
class ParityReport:
    rows: int
    features: int
    mismatches: dict[str, int]
    worst: dict[str, float]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        if self.ok:
            return f"parity ok: {self.rows} rows x {self.features} features"
        worst = ", ".join(f"{k}={v}" for k, v in sorted(self.mismatches.items()))
        return f"parity FAILED on {len(self.mismatches)} features: {worst}"


def _equal(a: object, b: object, exact: bool) -> tuple[bool, float]:
    if a is None and b is None:
        return True, 0.0
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b), 0.0
    if a is None or b is None:
        av = float("nan") if a is None else float(a)  # type: ignore[arg-type]
        bv = float("nan") if b is None else float(b)  # type: ignore[arg-type]
        return math.isnan(av) and math.isnan(bv), float("inf")
    av, bv = float(a), float(b)  # type: ignore[arg-type]
    if math.isnan(av) and math.isnan(bv):
        return True, 0.0
    if math.isnan(av) or math.isnan(bv):
        return False, float("inf")
    if exact:
        return av == bv, abs(av - bv)
    scale = max(1.0, abs(av), abs(bv))
    delta = abs(av - bv) / scale
    return delta <= FLOAT_TOLERANCE, delta


def check_parity(
    requests: pl.LazyFrame,
    static: StaticTables,
    zone_state: pl.LazyFrame | None,
    congestion_by_row: Sequence[Mapping[str, float]] | None = None,
    names: Sequence[str] | None = None,
) -> ParityReport:
    features = REGISTRY.select(names)
    assembled = assemble(requests, static, zone_state).collect()
    batch_df = REGISTRY.batch_frame(BatchContext(frame=assembled.lazy()), names).collect()

    route_lookup = static.route_lookup()
    zone_lookup = static.zone_lookup()
    exact_names = {f.name for f in features if _is_exact(f.dtype)}

    mismatches: dict[str, int] = {}
    worst: dict[str, float] = {}

    for i, row in enumerate(assembled.iter_rows(named=True)):
        congestion = (
            dict(congestion_by_row[i])
            if congestion_by_row is not None
            else _congestion_from_row(row, int(row["pu_zone"]))
        )
        weather = {
            k: row[k]
            for k in ("temp_c", "wind_ms", "visibility_m", "precip_mm_h", "snow_depth_cm")
            if k in row and row[k] is not None
        }
        ctx = online_context(row, static, route_lookup, zone_lookup, congestion, weather)
        online = REGISTRY.online_row(ctx, names)
        batch_row = batch_df.row(i, named=True)

        for f in features:
            same, delta = _equal(batch_row[f.name], online[f.name], f.name in exact_names)
            if not same:
                mismatches[f.name] = mismatches.get(f.name, 0) + 1
            worst[f.name] = max(worst.get(f.name, 0.0), delta if math.isfinite(delta) else 1e9)

    report = ParityReport(
        rows=assembled.height, features=len(features), mismatches=mismatches, worst=worst
    )
    log.info(
        "parity_checked",
        rows=report.rows,
        features=report.features,
        mismatched=len(report.mismatches),
        ok=report.ok,
    )
    return report


def _congestion_from_row(row: Mapping[str, object], zone: int) -> dict[str, float]:
    out: dict[str, float] = {}
    mapping = {"completed": "completed", "distance": "distance_m", "duration": "duration_s"}
    for kind, col in mapping.items():
        for w in (15, 30, 60):
            value = row.get(f"{col}_{w}m")
            out[congestion_key(kind, zone, w)] = (
                float(value) if value is not None else float("nan")  # type: ignore[arg-type]
            )
    return out
