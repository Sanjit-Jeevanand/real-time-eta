from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Final

import numpy as np
import polars as pl

from eta.data.splits import SPLIT_COLUMN, Split
from eta.logging import get_logger
from eta.routing.osrm import OsrmClient, RouteError
from eta.types import stat_float

if TYPE_CHECKING:
    from pathlib import Path

    from eta.config import Settings

__all__ = [
    "MATRIX_COLUMNS",
    "build_matrix",
    "intra_zone_estimate",
]

log = get_logger(__name__)

MATRIX_COLUMNS: Final = (
    "pu_zone",
    "do_zone",
    "route_distance_m",
    "free_flow_duration_s",
    "turn_count",
    "highway_fraction",
    "is_intra_zone",
    "is_estimated",
)

MEAN_DISTANCE_IN_SQUARE: Final = 0.5214
MIN_INTRA_ZONE_TRIPS: Final = 200
METRES_PER_MILE: Final = 1609.344
CHUNK: Final = 4000
MAX_PASSES: Final = 6


def intra_zone_estimate(area_km2: float, speed_ms: float) -> tuple[float, float]:
    distance_m = MEAN_DISTANCE_IN_SQUARE * math.sqrt(max(area_km2, 1e-6)) * 1000.0
    return distance_m, distance_m / max(speed_ms, 1.0)


def intra_zone_distances(
    enriched_glob: Path, zones: pl.DataFrame, min_trips: int = MIN_INTRA_ZONE_TRIPS
) -> pl.DataFrame:
    observed = (
        pl.scan_parquet(enriched_glob)
        .filter(
            (pl.col(SPLIT_COLUMN) == Split.TRAIN.value)
            & (pl.col("pu_zone") == pl.col("do_zone"))
            & (pl.col("trip_miles") > 0)
        )
        .group_by("pu_zone")
        .agg(
            (pl.col("trip_miles").median() * METRES_PER_MILE).alias("observed_m"),
            pl.len().alias("trips"),
        )
        .collect(engine="streaming")
    )

    base = zones.select(
        pl.col("zone_id").cast(pl.UInt16).alias("pu_zone"),
        pl.col("area_km2").sqrt().cast(pl.Float64).alias("root_area"),
    ).join(observed.with_columns(pl.col("pu_zone").cast(pl.UInt16)), on="pu_zone", how="left")

    fit_rows = base.filter(pl.col("trips") >= min_trips)
    x = fit_rows["root_area"].to_numpy()
    y = fit_rows["observed_m"].to_numpy()
    design = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])

    out = base.with_columns(
        pl.when(pl.col("trips") >= min_trips)
        .then(pl.col("observed_m"))
        .otherwise(intercept + slope * pl.col("root_area"))
        .alias("distance_m"),
        (pl.col("trips") >= min_trips).fill_null(value=False).alias("from_observation"),
    )
    log.info(
        "intra_zone_distances_built",
        zones=out.height,
        observed=int(out["from_observation"].sum()),
        fitted=int((~out["from_observation"]).sum()),
        intercept_m=round(intercept, 1),
        slope_m_per_root_km=round(slope, 1),
        min_trips=min_trips,
    )
    return out.select("pu_zone", "distance_m", "from_observation")


async def _route_all(
    zones: pl.DataFrame, base_url: str, timeout_s: float, concurrency: int
) -> pl.DataFrame:
    ids = zones["zone_id"].to_list()
    coords = {
        int(r["zone_id"]): (float(r["centroid_lat"]), float(r["centroid_lon"]))
        for r in zones.iter_rows(named=True)
    }

    pairs = [(o, d) for o in ids for d in ids if o != d]
    log.info("matrix_start", zones=len(ids), pairs=len(pairs))

    rows: list[dict[str, object]] = []
    no_route: list[tuple[int, int]] = []

    async with OsrmClient(base_url, timeout_s=timeout_s, concurrency=concurrency) as client:
        if not await client.health():
            msg = f"OSRM is not answering at {base_url}; run `make osrm-up` first"
            raise RuntimeError(msg)

        pending = list(pairs)
        for attempt in range(MAX_PASSES):
            retry: list[tuple[int, int]] = []
            for start in range(0, len(pending), CHUNK):
                batch = pending[start : start + CHUNK]
                legs = await client.route_many([(coords[o], coords[d]) for o, d in batch])
                for (o, d), leg in zip(batch, legs, strict=True):
                    if isinstance(leg, RouteError):
                        retry.append((o, d))
                    elif leg is None:
                        no_route.append((o, d))
                    else:
                        rows.append(
                            {
                                "pu_zone": o,
                                "do_zone": d,
                                "route_distance_m": leg.distance_m,
                                "free_flow_duration_s": leg.duration_s,
                                "turn_count": leg.turn_count,
                                "highway_fraction": leg.highway_fraction,
                                "is_intra_zone": False,
                            }
                        )
                log.info(
                    "matrix_progress",
                    pass_=attempt,
                    done=min(start + CHUNK, len(pending)),
                    total=len(pending),
                    routed=len(rows),
                )
            if not retry:
                break
            log.warning("matrix_retry_pass", pass_=attempt, pairs=len(retry))
            pending = retry
        else:
            msg = f"{len(pending)} zone pairs still failing after {MAX_PASSES} passes"
            raise RuntimeError(msg)

    if no_route:
        log.warning("matrix_no_route", pairs=len(no_route), sample=no_route[:5])
    return pl.DataFrame(rows)


def _repair_degenerate(inter: pl.DataFrame, intra: pl.DataFrame, speed_ms: float) -> pl.DataFrame:
    intra_m = {int(r["pu_zone"]): float(r["distance_m"]) for r in intra.iter_rows(named=True)}
    degenerate = inter.filter(
        (pl.col("route_distance_m") <= 0) | (pl.col("free_flow_duration_s") <= 0)
    )
    if degenerate.is_empty():
        return inter.with_columns(pl.lit(value=False).alias("is_estimated"))

    log.warning(
        "matrix_degenerate_pairs",
        pairs=degenerate.height,
        sample=[
            (int(a), int(b))
            for a, b in zip(degenerate["pu_zone"], degenerate["do_zone"], strict=True)
        ][:4],
    )
    fixed: list[dict[str, object]] = []
    for r in degenerate.iter_rows(named=True):
        a, b = int(r["pu_zone"]), int(r["do_zone"])
        distance = (intra_m.get(a, 1000.0) + intra_m.get(b, 1000.0)) / 2.0
        duration = distance / max(speed_ms, 1.0)
        fixed.append({"pu_zone": a, "do_zone": b, "d": distance, "t": duration})

    patch = pl.DataFrame(fixed)
    return (
        inter.join(patch, on=["pu_zone", "do_zone"], how="left")
        .with_columns(
            pl.coalesce("d", "route_distance_m").alias("route_distance_m"),
            pl.coalesce("t", "free_flow_duration_s").alias("free_flow_duration_s"),
            pl.col("d").is_not_null().alias("is_estimated"),
        )
        .drop("d", "t")
    )


def _add_intra_zone(inter: pl.DataFrame, intra: pl.DataFrame) -> pl.DataFrame:
    speeds = (
        inter.filter(pl.col("free_flow_duration_s") > 0)
        .with_columns((pl.col("route_distance_m") / pl.col("free_flow_duration_s")).alias("mps"))
        .group_by("pu_zone")
        .agg(pl.col("mps").median().alias("zone_speed_ms"))
    )
    global_speed = stat_float(speeds["zone_speed_ms"].median())
    inter = _repair_degenerate(inter, intra, global_speed)

    joined = intra.with_columns(pl.col("pu_zone").cast(pl.Int64)).join(
        speeds, on="pu_zone", how="left"
    )

    rows: list[dict[str, object]] = []
    for r in joined.iter_rows(named=True):
        speed = r["zone_speed_ms"]
        distance = float(r["distance_m"])
        duration = distance / max(float(speed) if speed is not None else global_speed, 1.0)
        rows.append(
            {
                "pu_zone": int(r["pu_zone"]),
                "do_zone": int(r["pu_zone"]),
                "route_distance_m": distance,
                "free_flow_duration_s": duration,
                "turn_count": 0,
                "highway_fraction": 0.0,
                "is_intra_zone": True,
                "is_estimated": True,
            }
        )
    return pl.concat([inter, pl.DataFrame(rows)], how="vertical")


def build_matrix(settings: Settings, zones: pl.DataFrame, enriched_glob: Path) -> Path:
    paths = settings.paths.resolve()
    out = paths["processed_dir"] / "zone_pair_matrix.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    inter = asyncio.run(
        _route_all(
            zones,
            settings.routing.osrm_url,
            settings.routing.osrm_timeout_s,
            concurrency=12,
        )
    )
    intra = intra_zone_distances(enriched_glob, zones)
    full = _add_intra_zone(inter, intra)

    full = (
        full.with_columns(
            pl.col("pu_zone").cast(pl.UInt16),
            pl.col("do_zone").cast(pl.UInt16),
            pl.col("route_distance_m").cast(pl.Float32),
            pl.col("free_flow_duration_s").cast(pl.Float32),
            pl.col("turn_count").cast(pl.UInt16),
            pl.col("highway_fraction").cast(pl.Float32),
        )
        .select(MATRIX_COLUMNS)
        .sort("pu_zone", "do_zone")
    )

    full.write_parquet(out, compression="zstd")
    log.info(
        "matrix_written",
        path=str(out),
        rows=full.height,
        bytes=out.stat().st_size,
        intra=int(full["is_intra_zone"].sum()),
    )
    return out


def validate_matrix(df: pl.DataFrame, n_zones: int) -> None:
    expected = n_zones * n_zones
    if df.height != expected:
        msg = f"matrix has {df.height} rows, expected {expected} for {n_zones} zones"
        raise ValueError(msg)
    if df.select(pl.struct("pu_zone", "do_zone").n_unique()).item() != expected:
        msg = "duplicate zone pairs in the matrix"
        raise ValueError(msg)
    if stat_float(df["route_distance_m"].min()) <= 0:
        msg = "matrix contains a non-positive route distance"
        raise ValueError(msg)
    if stat_float(df["free_flow_duration_s"].min()) <= 0:
        msg = "matrix contains a non-positive duration"
        raise ValueError(msg)
    estimated = int(df["is_estimated"].sum())
    if estimated > n_zones + max(10, n_zones // 20):
        msg = f"{estimated} estimated pairs is more than the intra-zone diagonal plus a few"
        raise ValueError(msg)
    bad = df.filter((pl.col("highway_fraction") < 0) | (pl.col("highway_fraction") > 1))
    if bad.height:
        msg = f"{bad.height} rows have a highway fraction outside [0, 1]"
        raise ValueError(msg)
