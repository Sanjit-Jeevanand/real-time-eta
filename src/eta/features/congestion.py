from __future__ import annotations

from typing import TYPE_CHECKING, Final

import polars as pl

from eta.data.splits import SPLIT_COLUMN
from eta.features.context import CONGESTION_WINDOWS
from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "BUCKET_MINUTES",
    "MIN_NIGHT_TRIPS",
    "NIGHT_HOURS",
    "STATE_COLUMNS",
    "attach_congestion",
    "build_zone_history",
    "build_zone_state",
    "observed_night_speed",
    "zone_routed_speed",
]

log = get_logger(__name__)

BUCKET_MINUTES: Final = 5
METRES_PER_MILE: Final = 1609.344

STATE_COLUMNS: Final = ("completed", "distance_m", "duration_s")


def _bucketed_completions(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.select(
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
    )


def build_zone_state(
    enriched_glob: Path, windows: Sequence[int] = CONGESTION_WINDOWS
) -> pl.LazyFrame:
    completions = _bucketed_completions(pl.scan_parquet(enriched_glob))

    grid = completions.select("zone", "bucket").unique()
    state = grid.join(completions, on=["zone", "bucket"], how="left").sort("zone", "bucket")

    exprs: list[pl.Expr] = []
    for w in windows:
        span = f"{w}m"
        for col in STATE_COLUMNS:
            exprs.append(
                pl.col(col)
                .fill_null(0)
                .rolling_sum_by("bucket", window_size=span, closed="right")
                .over("zone")
                .alias(f"{col}_{w}m")
            )
    return state.with_columns(exprs)


def attach_congestion(
    requests: pl.LazyFrame,
    zone_state: pl.LazyFrame,
    windows: Sequence[int] = CONGESTION_WINDOWS,
) -> pl.LazyFrame:
    keep = ["zone", "bucket"] + [f"{c}_{w}m" for w in windows for c in STATE_COLUMNS]
    state = zone_state.select(keep).sort("bucket")

    prepared = requests.with_columns(
        (
            pl.col("request_datetime").dt.truncate(f"{BUCKET_MINUTES}m").dt.cast_time_unit("us")
            - pl.duration(microseconds=1)
        ).alias("bucket")
    ).sort("bucket")

    joined = prepared.join_asof(
        state,
        on="bucket",
        by_left=["pu_zone"],
        by_right=["zone"],
        strategy="backward",
    )
    return joined.drop("zone", strict=False)


def zone_routed_speed(matrix: pl.DataFrame) -> pl.DataFrame:
    """Median OSRM routed speed out of each zone.

    This is a *routing* speed: distance over OSRM's profile duration, which uses
    static design speeds and no traffic. Measured against real 3-5am trips it runs
    systematically fast for local-road zones (ratio 0.61 overall, but 0.87 for
    airport zones, whose trips are highway-dominated like the routes this is built
    from). Kept only as the fallback reference for zones with too few night trips.
    """
    return (
        matrix.filter(~pl.col("is_intra_zone") & (pl.col("free_flow_duration_s") > 0))
        .with_columns((pl.col("route_distance_m") / pl.col("free_flow_duration_s")).alias("mps"))
        .group_by("pu_zone")
        .agg(pl.col("mps").median().alias("routed_speed_ms"))
        .rename({"pu_zone": "zone_id"})
    )


NIGHT_HOURS: Final = (3, 4)
MIN_NIGHT_TRIPS: Final = 200


def observed_night_speed(enriched_glob: Path, train: str) -> pl.DataFrame:
    """Median observed trip speed per zone during the 3-5am window, TRAIN SPLIT ONLY.

    The split filter is load-bearing, not hygiene: this becomes a frozen artifact
    consumed by cal/val/test and by serving, so any non-train row here would leak
    held-out traffic into every downstream feature value.
    """
    return (
        pl.scan_parquet(enriched_glob)
        .filter(
            (pl.col(SPLIT_COLUMN) == train)
            & pl.col("request_datetime").dt.hour().is_between(*NIGHT_HOURS)
            & (pl.col("trip_duration_s") > 60)
            & (pl.col("trip_miles") > 0.3)
        )
        .group_by("pu_zone")
        .agg(
            (pl.col("trip_miles") * METRES_PER_MILE / pl.col("trip_duration_s"))
            .median()
            .alias("night_speed_ms"),
            pl.len().alias("night_trips"),
        )
        .rename({"pu_zone": "zone_id"})
        .collect(engine="streaming")
    )


def build_zone_history(enriched_glob: Path, matrix: pl.DataFrame, train: str) -> pl.DataFrame:
    """Per-zone static reference table, built from the training split only.

    `reference_speed_ms` is an *empirical low-congestion reference*: the median
    observed 3-5am speed in that zone. It is deliberately not called a free-flow
    speed -- 3-5am is the quietest window available in the data, but nothing here
    demonstrates it is uncongested in the traffic-engineering sense. Its job is to
    normalise current speed into a relative degradation ratio, so what matters is
    that it is a stable per-zone reference, not that it is theoretically maximal.
    """
    hist = (
        pl.scan_parquet(enriched_glob)
        .filter(pl.col(SPLIT_COLUMN) == train)
        .group_by("pu_zone")
        .agg(
            pl.col("total_time_s").mean().alias("hist_mean_duration_s"),
            pl.len().alias("hist_trips"),
        )
        .rename({"pu_zone": "zone_id"})
        .collect(engine="streaming")
    )
    routed = zone_routed_speed(matrix)
    night = observed_night_speed(enriched_glob, train)

    out = (
        hist.join(routed, on="zone_id", how="full", coalesce=True)
        .join(night, on="zone_id", how="full", coalesce=True)
        .with_columns(
            pl.when(pl.col("night_trips").fill_null(0) >= MIN_NIGHT_TRIPS)
            .then(pl.col("night_speed_ms"))
            .otherwise(pl.col("routed_speed_ms"))
            .alias("reference_speed_ms")
        )
        .with_columns(
            pl.col("zone_id").cast(pl.UInt16),
            pl.col("hist_mean_duration_s").cast(pl.Float32),
            pl.col("reference_speed_ms").cast(pl.Float32),
            pl.col("routed_speed_ms").cast(pl.Float32),
            pl.col("hist_trips").fill_null(0).cast(pl.UInt32),
            pl.col("night_trips").fill_null(0).cast(pl.UInt32),
            (pl.col("night_trips").fill_null(0) >= MIN_NIGHT_TRIPS).alias("reference_observed"),
        )
        .drop("night_speed_ms")
    )
    log.info(
        "zone_history_built",
        zones=out.height,
        reference_observed=int(out["reference_observed"].sum()),
        reference_routed=int((~out["reference_observed"]).sum()),
        min_night_trips=MIN_NIGHT_TRIPS,
    )
    return out
