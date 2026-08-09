from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Final, cast

import polars as pl

from eta.data.download import download
from eta.data.schema import NYC_TZ
from eta.logging import get_logger

__all__ = [
    "STATIONS",
    "build_hourly_weather",
    "download_station_year",
    "nearest_station",
    "parse_isd",
]

log = get_logger(__name__)

ISD_BASE_URL: Final = "https://www.ncei.noaa.gov/data/global-hourly/access"

STATIONS: Final[dict[str, str]] = {
    "central_park": "72505394728",
    "lga": "72503014732",
    "jfk": "74486094789",
}

_MISSING_TMP: Final = 9999
_MISSING_WND_SPEED: Final = 9999
_MISSING_VIS: Final = 999999
_MISSING_PRECIP: Final = 9999


def download_station_year(station: str, year: int, dest: Path) -> Path:
    return download(f"{ISD_BASE_URL}/{year}/{station}.csv", dest / f"{station}_{year}.csv")


def _split_field(col: str, index: int) -> pl.Expr:
    return pl.col(col).str.split(",").list.get(index, null_on_oob=True)


def parse_isd(path: Path, name: str) -> pl.LazyFrame:
    lf = pl.scan_csv(path, infer_schema_length=0).select("DATE", "TMP", "WND", "VIS", "AA1", "AJ1")

    temp = _split_field("TMP", 0).cast(pl.Int32, strict=False)
    wind = _split_field("WND", 3).cast(pl.Int32, strict=False)
    vis = _split_field("VIS", 0).cast(pl.Int32, strict=False)
    precip_hours = _split_field("AA1", 0).cast(pl.Int32, strict=False)
    precip_depth = _split_field("AA1", 1).cast(pl.Int32, strict=False)
    snow_depth = _split_field("AJ1", 0).cast(pl.Int32, strict=False)

    return (
        lf.with_columns(
            pl.col("DATE")
            .str.to_datetime("%Y-%m-%dT%H:%M:%S", time_zone="UTC")
            .dt.convert_time_zone(NYC_TZ)
            .dt.truncate("1h")
            .alias("hour"),
            pl.when(temp.abs() != _MISSING_TMP).then(temp / 10.0).otherwise(None).alias("temp_c"),
            pl.when(wind != _MISSING_WND_SPEED).then(wind / 10.0).otherwise(None).alias("wind_ms"),
            pl.when(vis != _MISSING_VIS).then(vis).otherwise(None).alias("visibility_m"),
            pl.when((precip_depth != _MISSING_PRECIP) & (precip_hours > 0) & (precip_hours < 24))
            .then(precip_depth / 10.0 / precip_hours)
            .otherwise(None)
            .alias("precip_mm_h"),
            pl.when(snow_depth.is_between(0, 500))
            .then(snow_depth)
            .otherwise(None)
            .alias("snow_depth_cm"),
        )
        .group_by("hour")
        .agg(
            pl.col("temp_c").mean(),
            pl.col("wind_ms").mean(),
            pl.col("visibility_m").min(),
            pl.col("precip_mm_h").max(),
            pl.col("snow_depth_cm").max(),
        )
        .with_columns(pl.lit(name).alias("station"))
    )


def build_hourly_weather(
    raw_dir: Path,
    years: list[int],
    *,
    max_gap_hours: int = 2,
) -> pl.DataFrame:
    frames = [
        parse_isd(download_station_year(sid, year, raw_dir), name)
        for name, sid in STATIONS.items()
        for year in years
    ]
    obs = pl.concat(frames).collect(engine="streaming")

    lo = cast("dt.datetime", obs["hour"].min())
    hi = cast("dt.datetime", obs["hour"].max())
    grid_s = pl.datetime_range(lo, hi, interval="1h", time_zone=NYC_TZ, eager=True).alias("hour")
    grid = pl.Series(grid_s)

    out: list[pl.DataFrame] = []
    value_cols = ["temp_c", "wind_ms", "visibility_m", "precip_mm_h", "snow_depth_cm"]
    for name in STATIONS:
        station_obs = obs.filter(pl.col("station") == name).sort("hour")
        joined = (
            pl.DataFrame({"hour": grid})
            .join(station_obs.drop("station"), on="hour", how="left")
            .with_columns(pl.lit(name).alias("station"))
        )
        joined = joined.with_columns(
            pl.col("precip_mm_h").fill_null(0.0),
            pl.col("snow_depth_cm").fill_null(strategy="forward", limit=24),
        ).with_columns(
            [
                pl.col(c).fill_null(strategy="forward", limit=max_gap_hours)
                for c in ("temp_c", "wind_ms", "visibility_m")
            ]
        )
        out.append(joined)

    weather = pl.concat(out).sort("station", "hour")
    missing = {c: int(weather[c].null_count()) for c in value_cols}
    log.info(
        "weather_built",
        hours=grid.len(),
        stations=len(STATIONS),
        rows=weather.height,
        nulls_after_fill=missing,
    )
    return weather


def nearest_station(zone_col: str = "pu_zone") -> pl.Expr:
    return (
        pl.when(pl.col(zone_col) == 132)
        .then(pl.lit("jfk"))
        .when(pl.col(zone_col) == 138)
        .then(pl.lit("lga"))
        .when(pl.col(zone_col) == 1)
        .then(pl.lit("lga"))
        .otherwise(pl.lit("central_park"))
        .alias("station")
    )
