from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from eta.data.segments import assign_segments, load_zone_lookup, zone_density_map
from eta.data.splits import assign_split, holdout_week_index
from eta.data.weather import join_weather
from eta.logging import get_logger

if TYPE_CHECKING:
    from eta.config import Settings

__all__ = ["enrich_all", "enrich_month"]

log = get_logger(__name__)


def enrich_month(
    src: Path,
    dest_dir: Path,
    settings: Settings,
    weather: pl.DataFrame,
    density: pl.DataFrame,
) -> tuple[str, int]:
    month = src.stem.removeprefix("trips_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"enriched_{month}.parquet"

    lf = pl.scan_parquet(src)
    lf = join_weather(lf, weather)
    lf = assign_segments(lf, settings.segments, density)
    lf = assign_split(lf, settings.splits)
    lf = holdout_week_index(lf, settings.splits)
    lf.sink_parquet(out, compression="zstd")

    rows = int(pl.scan_parquet(out).select(pl.len()).collect().item())
    log.info("month_enriched", month=month, rows=rows)
    return month, rows


def enrich_all(settings: Settings) -> int:
    paths = settings.paths.resolve()
    weather = pl.read_parquet(paths["processed_dir"] / "weather_hourly.parquet")
    density = zone_density_map(
        load_zone_lookup(paths["raw_dir"] / "taxi_zone_lookup.csv"), settings.segments
    )

    sources = sorted((paths["processed_dir"] / "trips").glob("trips_*.parquet"))
    if not sources:
        msg = "no ingested trips found; run `make data-trips` first"
        raise RuntimeError(msg)

    total = 0
    for src in sources:
        _, rows = enrich_month(src, paths["processed_dir"] / "enriched", settings, weather, density)
        total += rows
    log.info("enrich_complete", months=len(sources), rows=total)
    return total
