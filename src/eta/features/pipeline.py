from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, cast

import polars as pl

from eta.features.congestion import attach_congestion
from eta.features.context import BOROUGH_IDS, BatchContext, OnlineContext
from eta.features.registry import REGISTRY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from eta.features.context import StaticTables

__all__ = [
    "REQUEST_COLUMNS",
    "assemble",
    "feature_frame",
    "online_context",
]

REQUEST_COLUMNS = ("request_datetime", "pu_zone", "do_zone")


def _embedding_columns(embeddings: pl.DataFrame) -> list[str]:
    return [c for c in embeddings.columns if c.startswith("zone_emb_")]


def assemble(
    requests: pl.LazyFrame, static: StaticTables, zone_state: pl.LazyFrame | None = None
) -> pl.LazyFrame:
    frame = requests
    if zone_state is not None:
        frame = attach_congestion(frame, zone_state)

    frame = frame.join(static.matrix.lazy(), on=["pu_zone", "do_zone"], how="left").join(
        static.detour.lazy(), on=["pu_zone", "do_zone"], how="left"
    )

    for side in ("pu", "do"):
        zones = static.zones.select(
            pl.col("zone_id").cast(pl.UInt16).alias(f"{side}_zone"),
            pl.col("area_km2").alias(f"{side}_area_km2"),
            pl.col("is_airport").alias(f"{side}_is_airport"),
            pl.col("borough").replace_strict(BOROUGH_IDS, default=0).alias(f"{side}_borough_id"),
        )
        history = static.zone_history.select(
            pl.col("zone_id").cast(pl.UInt16).alias(f"{side}_zone"),
            pl.col("hist_mean_duration_s").alias(f"{side}_hist_mean_duration_s"),
            pl.col("free_flow_speed_ms").alias(f"{side}_free_flow_speed_ms"),
        )
        frame = frame.join(zones.lazy(), on=f"{side}_zone", how="left").join(
            history.lazy(), on=f"{side}_zone", how="left"
        )

    cols = _embedding_columns(static.embeddings)
    for side in ("pu", "do"):
        emb = static.embeddings.select(
            [pl.col("zone_id").cast(pl.UInt16).alias(f"{side}_zone")]
            + [pl.col(c).alias(f"{side}_{c}") for c in cols]
        )
        frame = frame.join(emb.lazy(), on=f"{side}_zone", how="left")

    dot = pl.sum_horizontal([pl.col(f"pu_{c}") * pl.col(f"do_{c}") for c in cols])
    norm_pu = pl.sum_horizontal([pl.col(f"pu_{c}") ** 2 for c in cols]).sqrt()
    norm_do = pl.sum_horizontal([pl.col(f"do_{c}") ** 2 for c in cols]).sqrt()
    return frame.with_columns((dot / (norm_pu * norm_do)).alias("zone_embedding_similarity"))


def feature_frame(
    requests: pl.LazyFrame,
    static: StaticTables,
    zone_state: pl.LazyFrame | None = None,
    names: Sequence[str] | None = None,
) -> pl.LazyFrame:
    return REGISTRY.batch_frame(BatchContext(frame=assemble(requests, static, zone_state)), names)


def online_context(
    request: Mapping[str, object],
    static: StaticTables,
    route_lookup: Mapping[tuple[int, int], Mapping[str, float]],
    zone_lookup: Mapping[int, Mapping[str, float]],
    congestion: Mapping[str, float],
    weather: Mapping[str, float],
) -> OnlineContext:
    pu = int(cast("int", request["pu_zone"]))
    do = int(cast("int", request["do_zone"]))
    return OnlineContext(
        pu_zone=pu,
        do_zone=do,
        request_datetime=cast("dt.datetime", request["request_datetime"]),
        static=static,
        route=route_lookup.get((pu, do), {}),
        pu=zone_lookup.get(pu, {}),
        do=zone_lookup.get(do, {}),
        congestion=congestion,
        weather=weather,
    )
