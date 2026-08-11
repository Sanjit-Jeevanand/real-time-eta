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
    "STATE_COLUMNS",
    "attach_congestion",
    "build_zone_state",
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


def zone_free_flow_speed(matrix: pl.DataFrame) -> pl.DataFrame:
    return (
        matrix.filter(~pl.col("is_intra_zone") & (pl.col("free_flow_duration_s") > 0))
        .with_columns((pl.col("route_distance_m") / pl.col("free_flow_duration_s")).alias("mps"))
        .group_by("pu_zone")
        .agg(pl.col("mps").median().alias("free_flow_speed_ms"))
        .rename({"pu_zone": "zone_id"})
    )


def build_zone_history(enriched_glob: Path, matrix: pl.DataFrame, train: str) -> pl.DataFrame:
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
    speeds = zone_free_flow_speed(matrix)
    out = hist.join(speeds, on="zone_id", how="full", coalesce=True).with_columns(
        pl.col("zone_id").cast(pl.UInt16),
        pl.col("hist_mean_duration_s").cast(pl.Float32),
        pl.col("free_flow_speed_ms").cast(pl.Float32),
        pl.col("hist_trips").fill_null(0).cast(pl.UInt32),
    )
    log.info("zone_history_built", zones=out.height)
    return out
