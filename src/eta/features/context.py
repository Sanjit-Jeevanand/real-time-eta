from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import polars as pl

from eta.data.weather import WEATHER_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "CONGESTION_WINDOWS",
    "BatchContext",
    "OnlineContext",
    "StaticTables",
    "congestion_key",
    "load_static_tables",
]

CONGESTION_WINDOWS: Final = (15, 30, 60)

EPOCH: Final = dt.date(2023, 1, 1)


def congestion_key(kind: str, zone: int, window_min: int) -> str:
    return f"cong:{kind}:{zone}:{window_min}m"


@dataclass(frozen=True, slots=True)
class StaticTables:
    zones: pl.DataFrame
    matrix: pl.DataFrame
    detour: pl.DataFrame
    embeddings: pl.DataFrame
    zone_history: pl.DataFrame

    def route_lookup(self) -> dict[tuple[int, int], dict[str, float]]:
        joined = self.matrix.join(self.detour, on=["pu_zone", "do_zone"], how="left")
        out: dict[tuple[int, int], dict[str, float]] = {}
        for r in joined.iter_rows(named=True):
            out[(int(r["pu_zone"]), int(r["do_zone"]))] = {
                "route_distance_m": float(r["route_distance_m"]),
                "free_flow_duration_s": float(r["free_flow_duration_s"]),
                "turn_count": float(r["turn_count"]),
                "highway_fraction": float(r["highway_fraction"]),
                "is_intra_zone": float(r["is_intra_zone"]),
                "structural_detour_ratio": (
                    float(r["structural_detour_ratio"])
                    if r["structural_detour_ratio"] is not None
                    else float("nan")
                ),
                "detour_trips": float(r["detour_trips"] or 0),
            }
        return out

    def zone_lookup(self) -> dict[int, dict[str, float]]:
        hist = {int(r["zone_id"]): r for r in self.zone_history.iter_rows(named=True)}
        emb_cols = [c for c in self.embeddings.columns if c.startswith("zone_emb_")]
        emb = {
            int(r["zone_id"]): [float(r[c]) for c in emb_cols]
            for r in self.embeddings.iter_rows(named=True)
        }
        out: dict[int, dict[str, float]] = {}
        for r in self.zones.iter_rows(named=True):
            zid = int(r["zone_id"])
            h = hist.get(zid, {})
            out[zid] = {
                "area_km2": float(r["area_km2"]),
                "is_airport": float(bool(r["is_airport"])),
                "borough_id": float(BOROUGH_IDS.get(str(r["borough"]), 0)),
                "hist_mean_duration_s": float(h.get("hist_mean_duration_s") or 0.0),
                "reference_speed_ms": float(h.get("reference_speed_ms") or 8.0),
            }
            out[zid]["_embedding"] = emb.get(zid, [])  # type: ignore[assignment]
        return out


BOROUGH_IDS: Final[dict[str, int]] = {
    "Manhattan": 1,
    "Brooklyn": 2,
    "Queens": 3,
    "Bronx": 4,
    "Staten Island": 5,
    "EWR": 6,
}


def load_static_tables(processed: Path) -> StaticTables:
    return StaticTables(
        zones=pl.read_parquet(processed / "zones.parquet"),
        matrix=pl.read_parquet(processed / "zone_pair_matrix.parquet"),
        detour=pl.read_parquet(processed / "detour_ratios.parquet"),
        embeddings=pl.read_parquet(processed / "zone_embeddings.parquet"),
        zone_history=pl.read_parquet(processed / "zone_history.parquet"),
    )


@dataclass(frozen=True, slots=True)
class BatchContext:
    frame: pl.LazyFrame
    epoch: dt.date = EPOCH

    def col(self, name: str) -> pl.Expr:
        return pl.col(name)


@dataclass(slots=True)
class OnlineContext:
    pu_zone: int
    do_zone: int
    request_datetime: dt.datetime
    static: StaticTables | None = None
    route: Mapping[str, float] = field(default_factory=dict)
    pu: Mapping[str, float] = field(default_factory=dict)
    do: Mapping[str, float] = field(default_factory=dict)
    congestion: Mapping[str, float] = field(default_factory=dict)
    weather: Mapping[str, float] = field(default_factory=dict)
    epoch: dt.date = EPOCH

    def cong(self, kind: str, zone: int, window_min: int) -> float:
        return float(self.congestion.get(congestion_key(kind, zone, window_min), float("nan")))

    def wx(self, name: str) -> float:
        return float(self.weather.get(name, float("nan")))

    def required_weather(self) -> tuple[str, ...]:
        return WEATHER_COLUMNS
