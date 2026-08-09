from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl
import pyproj
import shapefile
from shapely.geometry import shape
from shapely.ops import transform

from eta.data.download import download
from eta.data.schema import AIRPORT_ZONE_IDS
from eta.logging import get_logger
from eta.types import stat_float

if TYPE_CHECKING:
    from eta.config import SegmentConfig

__all__ = [
    "TAXI_ZONES_URL",
    "ZONE_COLUMNS",
    "build_zone_table",
    "load_zone_geometry",
]

log = get_logger(__name__)

TAXI_ZONES_URL: Final = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

SOURCE_CRS: Final = "EPSG:2263"
TARGET_CRS: Final = "EPSG:4326"
SQ_FEET_PER_SQ_KM: Final = 10_763_910.4

ZONE_COLUMNS: Final = (
    "zone_id",
    "centroid_lat",
    "centroid_lon",
    "area_km2",
    "borough",
    "zone_name",
    "is_airport",
)


def _fetch(dest: Path) -> Path:
    archive = download(TAXI_ZONES_URL, dest / "taxi_zones.zip")
    target = dest / "taxi_zones"
    if not (target / "taxi_zones.shp").exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target.parent)
    return target / "taxi_zones.shp"


def load_zone_geometry(dest: Path) -> pl.DataFrame:
    shp_path = _fetch(dest)

    to_wgs84 = pyproj.Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True).transform

    rows: list[dict[str, object]] = []
    with shapefile.Reader(str(shp_path)) as reader:
        fields = [f[0] for f in reader.fields[1:]]
        for record, shape_rec in zip(reader.records(), reader.shapes(), strict=True):
            attrs = dict(zip(fields, record, strict=True))
            geom = shape(shape_rec.__geo_interface__)
            centroid = transform(to_wgs84, geom.centroid)
            zone_id = int(attrs["LocationID"])
            rows.append(
                {
                    "zone_id": zone_id,
                    "centroid_lat": float(centroid.y),
                    "centroid_lon": float(centroid.x),
                    "area_km2": float(geom.area) / SQ_FEET_PER_SQ_KM,
                    "borough": str(attrs["borough"]),
                    "zone_name": str(attrs["zone"]),
                }
            )

    df = pl.DataFrame(rows).sort("zone_id")
    log.info("zone_geometry_loaded", zones=df.height)
    return df


def build_zone_table(dest: Path, cfg: SegmentConfig) -> pl.DataFrame:
    geom = load_zone_geometry(dest)
    return geom.with_columns(
        pl.col("zone_id").cast(pl.UInt16),
        pl.col("centroid_lat").cast(pl.Float64),
        pl.col("centroid_lon").cast(pl.Float64),
        pl.col("area_km2").cast(pl.Float32),
        pl.col("zone_id").is_in(cfg.airport_zone_ids).alias("is_airport"),
    ).select(ZONE_COLUMNS)


def validate_zone_table(df: pl.DataFrame) -> None:
    if df.height == 0:
        msg = "zone table is empty"
        raise ValueError(msg)

    lat_lo, lat_hi = stat_float(df["centroid_lat"].min()), stat_float(df["centroid_lat"].max())
    lon_lo, lon_hi = stat_float(df["centroid_lon"].min()), stat_float(df["centroid_lon"].max())
    if not (lat_lo > 40.0 and lat_hi < 41.2):
        msg = f"centroid latitudes outside the NYC region: {lat_lo}..{lat_hi}"
        raise ValueError(msg)
    if not (lon_lo > -74.5 and lon_hi < -73.5):
        msg = f"centroid longitudes outside the NYC region: {lon_lo}..{lon_hi}"
        raise ValueError(msg)
    if df["zone_id"].n_unique() != df.height:
        msg = "duplicate zone ids in the zone table"
        raise ValueError(msg)
    missing = set(AIRPORT_ZONE_IDS) - set(df["zone_id"].to_list())
    if missing:
        msg = f"airport zones missing from the zone table: {sorted(missing)}"
        raise ValueError(msg)
