from __future__ import annotations

from typing import TYPE_CHECKING, Final

import polars as pl

from eta.data.splits import SPLIT_COLUMN, Split
from eta.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DETOUR_COLUMNS",
    "MIN_TRIPS_FOR_DETOUR",
    "build_detour_ratios",
]

log = get_logger(__name__)

METRES_PER_MILE: Final = 1609.344
MIN_TRIPS_FOR_DETOUR: Final = 30

DETOUR_COLUMNS: Final = ("pu_zone", "do_zone", "structural_detour_ratio", "detour_trips")


def build_detour_ratios(
    enriched_glob: Path, matrix: pl.DataFrame, min_trips: int = MIN_TRIPS_FOR_DETOUR
) -> pl.DataFrame:
    observed = (
        pl.scan_parquet(enriched_glob)
        .filter(
            (pl.col(SPLIT_COLUMN) == Split.TRAIN.value)
            & (pl.col("trip_miles") > 0)
            & pl.col("trip_miles").is_not_null()
        )
        .group_by("pu_zone", "do_zone")
        .agg(
            pl.col("trip_miles").median().alias("actual_miles"),
            pl.len().alias("detour_trips"),
        )
        .collect(engine="streaming")
    )

    joined = matrix.select("pu_zone", "do_zone", "route_distance_m", "is_intra_zone").join(
        observed, on=["pu_zone", "do_zone"], how="left"
    )

    ratio = (pl.col("actual_miles") * METRES_PER_MILE) / pl.col("route_distance_m")

    out = joined.with_columns(
        pl.when(
            pl.col("detour_trips").fill_null(0) >= min_trips,
        )
        .then(ratio)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("structural_detour_ratio"),
        pl.col("detour_trips").fill_null(0).cast(pl.UInt32),
    ).select(DETOUR_COLUMNS)

    covered = int(out["structural_detour_ratio"].is_not_null().sum())
    log.info(
        "detour_ratios_built",
        pairs=out.height,
        covered=covered,
        coverage_pct=round(100.0 * covered / out.height, 2),
        min_trips=min_trips,
    )
    return out
