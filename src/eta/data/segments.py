from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from eta.types import (
    TimeBucket,
    TripLengthBucket,
    WeatherBucket,
    ZoneDensityBucket,
    polars_enum,
)

if TYPE_CHECKING:
    from eta.config import SegmentConfig

__all__ = [
    "SEGMENT_COLUMNS",
    "assign_segments",
    "load_zone_lookup",
    "zone_density_map",
]

SEGMENT_COLUMNS = ("seg_time", "seg_trip_length", "seg_zone_density", "seg_weather")


def load_zone_lookup(path: Path) -> pl.DataFrame:
    return pl.read_csv(path).rename(
        {"LocationID": "zone_id", "Borough": "borough", "Zone": "zone_name"}
    )


def zone_density_map(lookup: pl.DataFrame, cfg: SegmentConfig) -> pl.DataFrame:
    density = (
        pl.when(pl.col("zone_id").is_in(cfg.airport_zone_ids))
        .then(pl.lit(ZoneDensityBucket.AIRPORT.value))
        .when(pl.col("borough") == cfg.manhattan_core_borough)
        .then(pl.lit(ZoneDensityBucket.MANHATTAN_CORE.value))
        .otherwise(pl.lit(ZoneDensityBucket.OUTER_BOROUGH.value))
    )
    return lookup.select(
        pl.col("zone_id").cast(pl.UInt16).alias("pu_zone"),
        density.cast(polars_enum(ZoneDensityBucket)).alias("seg_zone_density"),
    )


def _time_bucket(cfg: SegmentConfig) -> pl.Expr:
    hour = pl.col("request_datetime").dt.hour()
    ln_start, ln_end = cfg.late_night_hours
    late_night = (hour >= ln_start) | (hour < ln_end)
    return (
        pl.when(late_night)
        .then(pl.lit(TimeBucket.LATE_NIGHT.value))
        .when(hour.is_between(cfg.peak_am_hours[0], cfg.peak_am_hours[1], closed="left"))
        .then(pl.lit(TimeBucket.PEAK_AM.value))
        .when(hour.is_between(cfg.peak_pm_hours[0], cfg.peak_pm_hours[1], closed="left"))
        .then(pl.lit(TimeBucket.PEAK_PM.value))
        .otherwise(pl.lit(TimeBucket.OFF_PEAK.value))
        .cast(polars_enum(TimeBucket))
        .alias("seg_time")
    )


def _trip_length_bucket(cfg: SegmentConfig) -> pl.Expr:
    miles = pl.col("trip_miles")
    return (
        pl.when(miles.is_null())
        .then(None)
        .when(miles < cfg.short_trip_max_miles)
        .then(pl.lit(TripLengthBucket.SHORT.value))
        .when(miles > cfg.long_trip_min_miles)
        .then(pl.lit(TripLengthBucket.LONG.value))
        .otherwise(pl.lit(TripLengthBucket.MEDIUM.value))
        .cast(polars_enum(TripLengthBucket))
        .alias("seg_trip_length")
    )


def _weather_bucket(cfg: SegmentConfig) -> pl.Expr:
    precip = pl.col("precip_mm_h").fill_null(0.0)
    temp = pl.col("temp_c")
    return (
        pl.when((temp <= 0.0) & (precip >= cfg.snow_mm_threshold))
        .then(pl.lit(WeatherBucket.SNOW.value))
        .when(precip >= cfg.rain_mm_threshold)
        .then(pl.lit(WeatherBucket.RAIN.value))
        .otherwise(pl.lit(WeatherBucket.CLEAR.value))
        .cast(polars_enum(WeatherBucket))
        .alias("seg_weather")
    )


def assign_segments(lf: pl.LazyFrame, cfg: SegmentConfig, density: pl.DataFrame) -> pl.LazyFrame:
    return lf.with_columns(_time_bucket(cfg), _trip_length_bucket(cfg), _weather_bucket(cfg)).join(
        density.lazy(), on="pu_zone", how="left"
    )
