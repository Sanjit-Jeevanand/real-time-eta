from __future__ import annotations

import datetime as dt
import math
from typing import TYPE_CHECKING

import polars as pl

from eta.features.context import CONGESTION_WINDOWS
from eta.features.registry import REGISTRY, Family
from eta.features.temporal import US_HOLIDAYS_2023_2024

if TYPE_CHECKING:
    from eta.features.context import OnlineContext

__all__ = ["register_all"]

F32 = pl.Float32
F64 = pl.Float64
U16 = pl.UInt16
BOOL = pl.Boolean

NAN = float("nan")


def _f(value: float | int | None) -> float:
    return float(value) if value is not None else NAN


# ---------------------------------------------------------------- route -----
def _register_route() -> None:
    for name, dtype in (
        ("route_distance_m", F32),
        ("free_flow_duration_s", F32),
        ("turn_count", U16),
        ("highway_fraction", F32),
        ("structural_detour_ratio", F32),
    ):
        REGISTRY.add(
            name,
            Family.ROUTE,
            dtype,
            (lambda n: lambda _ctx: pl.col(n))(name),
            (lambda n: lambda ctx: _f(ctx.route.get(n)))(name),
        )

    REGISTRY.add(
        "is_intra_zone",
        Family.ROUTE,
        BOOL,
        lambda _ctx: pl.col("pu_zone") == pl.col("do_zone"),
        lambda ctx: ctx.pu_zone == ctx.do_zone,
    )
    REGISTRY.add(
        "route_free_flow_speed_ms",
        Family.ROUTE,
        F32,
        lambda _ctx: pl.col("route_distance_m") / pl.col("free_flow_duration_s").clip(1.0, None),
        lambda ctx: _f(ctx.route.get("route_distance_m"))
        / max(_f(ctx.route.get("free_flow_duration_s")), 1.0),
    )
    REGISTRY.add(
        "detour_confidence",
        Family.ROUTE,
        F32,
        lambda _ctx: pl.col("detour_trips").fill_null(0).log1p(),
        lambda ctx: math.log1p(max(_f(ctx.route.get("detour_trips")), 0.0)),
    )


# ------------------------------------------------------------- temporal -----
def _register_temporal() -> None:
    def cyc(name: str, value: pl.Expr, period: int) -> None:
        angle = value * (2.0 * math.pi / period)
        REGISTRY.add(
            f"{name}_sin",
            Family.TEMPORAL,
            F32,
            (lambda a: lambda _ctx: a.sin())(angle),
            (lambda n, p: lambda ctx: math.sin(_clock(ctx, n) * 2 * math.pi / p))(name, period),
        )
        REGISTRY.add(
            f"{name}_cos",
            Family.TEMPORAL,
            F32,
            (lambda a: lambda _ctx: a.cos())(angle),
            (lambda n, p: lambda ctx: math.cos(_clock(ctx, n) * 2 * math.pi / p))(name, period),
        )

    ts = pl.col("request_datetime")
    cyc("hour", ts.dt.hour() + ts.dt.minute() / 60.0, 24)
    cyc("dow", (ts.dt.weekday() - 1).cast(F64), 7)
    cyc("month", (ts.dt.month() - 1).cast(F64), 12)

    REGISTRY.add(
        "minute_of_day",
        Family.TEMPORAL,
        F32,
        lambda _ctx: pl.col("request_datetime").dt.hour().cast(F32) * 60
        + pl.col("request_datetime").dt.minute().cast(F32),
        lambda ctx: ctx.request_datetime.hour * 60 + ctx.request_datetime.minute,
    )
    REGISTRY.add(
        "is_weekend",
        Family.TEMPORAL,
        BOOL,
        lambda _ctx: pl.col("request_datetime").dt.weekday() >= 6,
        lambda ctx: ctx.request_datetime.weekday() >= 5,
    )
    REGISTRY.add(
        "is_holiday",
        Family.TEMPORAL,
        BOOL,
        lambda _ctx: pl.col("request_datetime")
        .dt.date()
        .is_in([dt.date.fromisoformat(d) for d in US_HOLIDAYS_2023_2024]),
        lambda ctx: ctx.request_datetime.date().isoformat() in US_HOLIDAYS_2023_2024,
    )
    REGISTRY.add(
        "days_since_epoch",
        Family.TEMPORAL,
        F32,
        lambda ctx: (pl.col("request_datetime").dt.date() - ctx.epoch).dt.total_days(),
        lambda ctx: (ctx.request_datetime.date() - ctx.epoch).days,
    )


def _clock(ctx: OnlineContext, name: str) -> float:
    t = ctx.request_datetime
    if name == "hour":
        return t.hour + t.minute / 60.0
    if name == "dow":
        return float(t.weekday())
    return float(t.month - 1)


# ----------------------------------------------------------- congestion -----
def _register_congestion() -> None:
    for w in CONGESTION_WINDOWS:
        REGISTRY.add(
            f"zone_speed_ratio_{w}m",
            Family.CONGESTION,
            F32,
            (
                lambda w_: lambda _ctx: (
                    pl.col(f"distance_m_{w_}m") / pl.col(f"duration_s_{w_}m").clip(1.0, None)
                )
                / pl.col("pu_free_flow_speed_ms").clip(0.1, None)
            )(w),
            (
                lambda w_: lambda ctx: (
                    ctx.cong("distance", ctx.pu_zone, w_)
                    / max(ctx.cong("duration", ctx.pu_zone, w_), 1.0)
                )
                / max(_f(ctx.pu.get("free_flow_speed_ms")), 0.1)
            )(w),
            redis_keys=(f"cong:distance:{{zone}}:{w}m", f"cong:duration:{{zone}}:{w}m"),
        )
        REGISTRY.add(
            f"zone_completed_trips_{w}m",
            Family.CONGESTION,
            F32,
            (lambda w_: lambda _ctx: pl.col(f"completed_{w_}m"))(w),
            (lambda w_: lambda ctx: ctx.cong("completed", ctx.pu_zone, w_))(w),
            redis_keys=(f"cong:completed:{{zone}}:{w}m",),
        )
        REGISTRY.add(
            f"zone_trip_density_{w}m",
            Family.CONGESTION,
            F32,
            (
                lambda w_: lambda _ctx: pl.col(f"completed_{w_}m")
                / pl.col("pu_area_km2").clip(0.01, None)
            )(w),
            (
                lambda w_: lambda ctx: ctx.cong("completed", ctx.pu_zone, w_)
                / max(_f(ctx.pu.get("area_km2")), 0.01)
            )(w),
            redis_keys=(f"cong:completed:{{zone}}:{w}m",),
        )
        REGISTRY.add(
            f"zone_mean_trip_duration_{w}m",
            Family.CONGESTION,
            F32,
            (
                lambda w_: lambda _ctx: pl.col(f"duration_s_{w_}m")
                / pl.col(f"completed_{w_}m").clip(1, None)
            )(w),
            (
                lambda w_: lambda ctx: ctx.cong("duration", ctx.pu_zone, w_)
                / max(ctx.cong("completed", ctx.pu_zone, w_), 1.0)
            )(w),
            redis_keys=(
                f"cong:duration:{{zone}}:{w}m",
                f"cong:completed:{{zone}}:{w}m",
            ),
        )


# ----------------------------------------------------------------- zone -----
def _register_zone() -> None:
    for side in ("pu", "do"):
        REGISTRY.add(
            f"{side}_area_km2",
            Family.ZONE,
            F32,
            (lambda s: lambda _ctx: pl.col(f"{s}_area_km2"))(side),
            (lambda s: lambda ctx: _f(getattr(ctx, s).get("area_km2")))(side),
        )
        REGISTRY.add(
            f"{side}_is_airport",
            Family.ZONE,
            BOOL,
            (lambda s: lambda _ctx: pl.col(f"{s}_is_airport"))(side),
            (lambda s: lambda ctx: bool(getattr(ctx, s).get("is_airport", 0.0)))(side),
        )
        REGISTRY.add(
            f"{side}_hist_mean_duration_s",
            Family.ZONE,
            F32,
            (lambda s: lambda _ctx: pl.col(f"{s}_hist_mean_duration_s"))(side),
            (lambda s: lambda ctx: _f(getattr(ctx, s).get("hist_mean_duration_s")))(side),
        )

    REGISTRY.add(
        "same_borough",
        Family.ZONE,
        BOOL,
        lambda _ctx: pl.col("pu_borough_id") == pl.col("do_borough_id"),
        lambda ctx: _f(ctx.pu.get("borough_id")) == _f(ctx.do.get("borough_id")),
    )
    REGISTRY.add(
        "zone_embedding_similarity",
        Family.ZONE,
        F32,
        lambda _ctx: pl.col("zone_embedding_similarity"),
        lambda ctx: _cosine(
            list(ctx.pu.get("_embedding", [])),  # type: ignore[arg-type]
            list(ctx.do.get("_embedding", [])),  # type: ignore[arg-type]
        ),
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return NAN
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return NAN
    return dot / (na * nb)


# -------------------------------------------------------------- weather -----
def _register_weather() -> None:
    for name in ("temp_c", "wind_ms", "visibility_m", "precip_mm_h", "snow_depth_cm"):
        REGISTRY.add(
            name,
            Family.WEATHER,
            F32,
            (lambda n: lambda _ctx: pl.col(n))(name),
            (lambda n: lambda ctx: ctx.wx(n))(name),
            redis_keys=(f"wx:{name}",),
        )
    REGISTRY.add(
        "precip_x_peak",
        Family.WEATHER,
        F32,
        lambda _ctx: pl.col("precip_mm_h").fill_null(0.0)
        * pl.col("request_datetime").dt.hour().is_between(16, 20, closed="left").cast(F32),
        lambda ctx: (0.0 if math.isnan(ctx.wx("precip_mm_h")) else ctx.wx("precip_mm_h"))
        * float(16 <= ctx.request_datetime.hour < 20),
        redis_keys=("wx:precip_mm_h",),
    )


def register_all() -> None:
    if len(REGISTRY):
        return
    _register_route()
    _register_temporal()
    _register_congestion()
    _register_zone()
    _register_weather()


register_all()
