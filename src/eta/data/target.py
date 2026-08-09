from __future__ import annotations

import polars as pl

__all__ = [
    "COMPONENT_COLUMNS",
    "TARGET_COLUMN",
    "with_target",
]

TARGET_COLUMN = "total_time_s"
COMPONENT_COLUMNS = ("dispatch_approach_s", "curb_wait_s", "trip_duration_s")


def with_target(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        (pl.col("dropoff_datetime") - pl.col("request_datetime"))
        .dt.total_seconds()
        .alias(TARGET_COLUMN),
        (pl.col("on_scene_datetime") - pl.col("request_datetime"))
        .dt.total_seconds()
        .alias("dispatch_approach_s"),
        (pl.col("pickup_datetime") - pl.col("on_scene_datetime"))
        .dt.total_seconds()
        .alias("curb_wait_s"),
        (pl.col("dropoff_datetime") - pl.col("pickup_datetime"))
        .dt.total_seconds()
        .alias("trip_duration_s"),
    )


def components_reconcile(lf: pl.LazyFrame, tolerance_s: float = 1.0) -> pl.LazyFrame:
    total = pl.sum_horizontal(
        pl.col("dispatch_approach_s"), pl.col("curb_wait_s"), pl.col("trip_duration_s")
    )
    return lf.filter(
        pl.col("dispatch_approach_s").is_not_null()
        & ((total - pl.col(TARGET_COLUMN)).abs() > tolerance_s)
    )
