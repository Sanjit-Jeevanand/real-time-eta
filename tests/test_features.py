from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from eta.data.schema import NYC_TZ
from eta.features.congestion import BUCKET_MINUTES, attach_congestion, build_zone_state
from eta.features.context import BatchContext, OnlineContext, congestion_key
from eta.features.families import register_all
from eta.features.registry import REGISTRY, Family

TZ = ZoneInfo(NYC_TZ)
register_all()

EXPECTED_COUNTS = {
    Family.ROUTE: 8,
    Family.TEMPORAL: 10,
    Family.CONGESTION: 12,
    Family.ZONE: 8,
    Family.WEATHER: 6,
}


def _t(day: int, hour: int, minute: int = 0) -> dt.datetime:
    return dt.datetime(2023, 6, day, hour, minute, tzinfo=TZ)


# ------------------------------------------------------------- registry -----
def test_family_counts_match_the_plan() -> None:
    assert REGISTRY.counts() == EXPECTED_COUNTS
    assert len(REGISTRY) == 44


def test_every_feature_declares_both_implementations() -> None:
    for f in REGISTRY:
        assert callable(f.batch), f.name
        assert callable(f.online), f.name
        assert f.family in Family, f.name


def test_feature_names_are_unique_and_registration_is_guarded() -> None:
    assert len(set(REGISTRY.names)) == len(REGISTRY)
    existing = next(iter(REGISTRY))
    with pytest.raises(ValueError, match="duplicate feature name"):
        REGISTRY.register(existing)


def test_registering_twice_is_a_noop() -> None:
    before = len(REGISTRY)
    register_all()
    assert len(REGISTRY) == before


def test_congestion_features_declare_their_redis_keys() -> None:
    for f in REGISTRY.by_family(Family.CONGESTION):
        assert f.redis_keys, f.name
        assert all("{zone}" in k for k in f.redis_keys), f.name
    assert REGISTRY.required_redis_keys()


def test_non_stateful_families_need_no_redis() -> None:
    for family in (Family.ROUTE, Family.TEMPORAL, Family.ZONE):
        for f in REGISTRY.by_family(family):
            assert f.redis_keys == (), f.name


# ---------------------------------------------------------- point in time ---
def _trips(rows: list[tuple[int, dt.datetime, dt.datetime, float, int]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema={
            "pu_zone": pl.UInt16,
            "request_datetime": pl.Datetime("us", NYC_TZ),
            "dropoff_datetime": pl.Datetime("us", NYC_TZ),
            "trip_miles": pl.Float64,
            "trip_duration_s": pl.Int64,
        },
        orient="row",
    )


def test_state_excludes_a_trip_that_completes_after_the_request() -> None:
    straddler = _trips([(7, _t(1, 8, 0), _t(1, 9, 30), 10.0, 5400)])
    state = build_zone_state_frame(straddler)

    request = pl.LazyFrame(
        {
            "pu_zone": pl.Series([7], dtype=pl.UInt16),
            "request_datetime": pl.Series([_t(1, 9, 0)], dtype=pl.Datetime("us", NYC_TZ)),
        }
    )
    out = attach_congestion(request, state).collect()
    assert out["completed_60m"][0] in (None, 0), (
        "a trip that started before but finished after the request must not be visible"
    )


def test_state_includes_a_trip_that_completed_before_the_request() -> None:
    done = _trips([(7, _t(1, 8, 0), _t(1, 8, 30), 10.0, 1800)])
    state = build_zone_state_frame(done)
    request = pl.LazyFrame(
        {
            "pu_zone": pl.Series([7], dtype=pl.UInt16),
            "request_datetime": pl.Series([_t(1, 9, 0)], dtype=pl.Datetime("us", NYC_TZ)),
        }
    )
    out = attach_congestion(request, state).collect()
    assert out["completed_60m"][0] == 1


def test_window_boundary_is_left_closed() -> None:
    same_bucket = _trips([(7, _t(1, 8, 55), _t(1, 9, 0), 1.0, 300)])
    state = build_zone_state_frame(same_bucket)
    request = pl.LazyFrame(
        {
            "pu_zone": pl.Series([7], dtype=pl.UInt16),
            "request_datetime": pl.Series([_t(1, 9, 0)], dtype=pl.Datetime("us", NYC_TZ)),
        }
    )
    out = attach_congestion(request, state).collect()
    assert out["completed_60m"][0] in (None, 0), (
        "a trip completing in the request's own bucket is not strictly before it"
    )


def test_state_does_not_bleed_between_zones() -> None:
    other_zone = _trips([(9, _t(1, 8, 0), _t(1, 8, 30), 10.0, 1800)])
    state = build_zone_state_frame(other_zone)
    request = pl.LazyFrame(
        {
            "pu_zone": pl.Series([7], dtype=pl.UInt16),
            "request_datetime": pl.Series([_t(1, 9, 0)], dtype=pl.Datetime("us", NYC_TZ)),
        }
    )
    out = attach_congestion(request, state).collect()
    assert out["completed_60m"][0] is None


def test_window_is_anchored_at_the_matched_state_bucket() -> None:
    trips = _trips(
        [
            (7, _t(1, 7, 0), _t(1, 8, 0), 1.0, 300),
            (7, _t(1, 8, 30), _t(1, 8, 55), 1.0, 300),
        ]
    )
    state = build_zone_state_frame(trips)
    request = pl.LazyFrame(
        {
            "pu_zone": pl.Series([7], dtype=pl.UInt16),
            "request_datetime": pl.Series([_t(1, 9, 0)], dtype=pl.Datetime("us", NYC_TZ)),
        }
    )
    out = attach_congestion(request, state).collect()
    assert out["completed_60m"][0] == 2, (
        "the window is anchored at the last state bucket strictly before the request "
        "(08:55), covering (07:55, 08:55] -- not at the request time itself"
    )


def test_window_left_edge_is_open() -> None:
    trips = _trips(
        [
            (7, _t(1, 7, 30), _t(1, 7, 55), 1.0, 300),
            (7, _t(1, 8, 30), _t(1, 8, 55), 1.0, 300),
        ]
    )
    state = build_zone_state_frame(trips)
    request = pl.LazyFrame(
        {
            "pu_zone": pl.Series([7], dtype=pl.UInt16),
            "request_datetime": pl.Series([_t(1, 9, 0)], dtype=pl.Datetime("us", NYC_TZ)),
        }
    )
    out = attach_congestion(request, state).collect()
    assert out["completed_60m"][0] == 1, (
        "a trip completing exactly one window before the anchor is outside (B-60m, B] "
        "and must not be double-counted across adjacent windows"
    )


def test_bucket_size_divides_every_window() -> None:
    from eta.features.context import CONGESTION_WINDOWS

    for w in CONGESTION_WINDOWS:
        assert w % BUCKET_MINUTES == 0


def build_zone_state_frame(trips: pl.LazyFrame) -> pl.LazyFrame:
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    trips.collect().write_parquet(tmp / "enriched_x.parquet")
    return build_zone_state(tmp / "enriched_*.parquet")


# ------------------------------------------------------------- temporal -----
def _batch_one(row: Mapping[str, object], names: list[str]) -> dict[str, object]:
    ctx = BatchContext(frame=pl.LazyFrame([row]))
    return REGISTRY.batch_frame(ctx, names).collect().row(0, named=True)


def test_cyclical_batch_and_online_agree() -> None:
    when = _t(1, 23, 45)
    names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "minute_of_day"]
    batch = _batch_one({"request_datetime": when}, names)
    online = REGISTRY.online_row(OnlineContext(1, 2, when), names)
    for n in names:
        assert batch[n] == pytest.approx(online[n], abs=1e-5), n


def test_holiday_and_weekend_agree() -> None:
    for when in (_t(4, 12), dt.datetime(2023, 7, 4, 12, tzinfo=TZ)):
        names = ["is_holiday", "is_weekend"]
        batch = _batch_one({"request_datetime": when}, names)
        online = REGISTRY.online_row(OnlineContext(1, 2, when), names)
        assert batch == online, when


def test_days_since_epoch_agrees() -> None:
    when = _t(15, 9)
    batch = _batch_one({"request_datetime": when}, ["days_since_epoch"])
    online = REGISTRY.online_row(OnlineContext(1, 2, when), ["days_since_epoch"])
    assert batch["days_since_epoch"] == pytest.approx(online["days_since_epoch"])


# --------------------------------------------------------------- weather ----
def test_precip_x_peak_agrees_inside_and_outside_peak() -> None:
    for hour, expected in ((18, 2.5), (11, 0.0)):
        when = _t(1, hour)
        batch = _batch_one({"request_datetime": when, "precip_mm_h": 2.5}, ["precip_x_peak"])
        online = REGISTRY.online_row(
            OnlineContext(1, 2, when, weather={"precip_mm_h": 2.5}), ["precip_x_peak"]
        )
        assert batch["precip_x_peak"] == pytest.approx(expected)
        assert online["precip_x_peak"] == pytest.approx(expected)


# ----------------------------------------------------------------- route ----
def test_route_features_agree() -> None:
    row = {
        "pu_zone": 7,
        "do_zone": 9,
        "route_distance_m": 8000.0,
        "free_flow_duration_s": 800.0,
        "detour_trips": 120,
    }
    names = ["route_free_flow_speed_ms", "detour_confidence", "is_intra_zone"]
    batch = _batch_one(row, names)
    online = REGISTRY.online_row(
        OnlineContext(
            7,
            9,
            _t(1, 9),
            route={
                "route_distance_m": 8000.0,
                "free_flow_duration_s": 800.0,
                "detour_trips": 120.0,
            },
        ),
        names,
    )
    assert batch["route_free_flow_speed_ms"] == pytest.approx(10.0)
    assert online["route_free_flow_speed_ms"] == pytest.approx(10.0)
    assert batch["detour_confidence"] == pytest.approx(math.log1p(120), abs=1e-5)
    assert online["detour_confidence"] == pytest.approx(math.log1p(120), abs=1e-5)
    assert batch["is_intra_zone"] is False
    assert online["is_intra_zone"] is False


# ------------------------------------------------------------ congestion ----
def test_congestion_speed_ratio_agrees() -> None:
    row = {
        "distance_m_15m": 20_000.0,
        "duration_s_15m": 2_000.0,
        "completed_15m": 40,
        "pu_reference_speed_ms": 12.5,
        "pu_area_km2": 2.0,
    }
    names = ["zone_speed_ratio_15m", "zone_trip_density_15m", "zone_mean_trip_duration_15m"]
    batch = _batch_one(row, names)
    online = REGISTRY.online_row(
        OnlineContext(
            7,
            9,
            _t(1, 9),
            pu={"reference_speed_ms": 12.5, "area_km2": 2.0},
            congestion={
                congestion_key("distance", 7, 15): 20_000.0,
                congestion_key("duration", 7, 15): 2_000.0,
                congestion_key("completed", 7, 15): 40.0,
            },
        ),
        names,
    )
    assert batch["zone_speed_ratio_15m"] == pytest.approx(0.8, abs=1e-5)
    assert online["zone_speed_ratio_15m"] == pytest.approx(0.8, abs=1e-5)
    assert batch["zone_trip_density_15m"] == pytest.approx(20.0)
    assert online["zone_trip_density_15m"] == pytest.approx(20.0)
    assert batch["zone_mean_trip_duration_15m"] == pytest.approx(50.0)
    assert online["zone_mean_trip_duration_15m"] == pytest.approx(50.0)


# ------------------------------------------------------------------ zone ----
def test_embedding_similarity_agrees_and_is_bounded() -> None:
    ctx = OnlineContext(
        7,
        9,
        _t(1, 9),
        pu={"_embedding": [1.0, 0.0, 1.0]},  # type: ignore[dict-item]
        do={"_embedding": [1.0, 0.0, 1.0]},  # type: ignore[dict-item]
    )
    same = REGISTRY.online_row(ctx, ["zone_embedding_similarity"])
    assert same["zone_embedding_similarity"] == pytest.approx(1.0)

    ctx.do = {"_embedding": [0.0, 1.0, 0.0]}  # type: ignore[dict-item]
    orth = REGISTRY.online_row(ctx, ["zone_embedding_similarity"])
    assert orth["zone_embedding_similarity"] == pytest.approx(0.0, abs=1e-9)


def test_embedding_similarity_is_nan_when_missing() -> None:
    ctx = OnlineContext(7, 9, _t(1, 9))
    out = REGISTRY.online_row(ctx, ["zone_embedding_similarity"])
    assert math.isnan(float(out["zone_embedding_similarity"]))  # type: ignore[arg-type]


def test_same_borough_agrees() -> None:
    batch = _batch_one({"pu_borough_id": 1, "do_borough_id": 1}, ["same_borough"])
    online = REGISTRY.online_row(
        OnlineContext(7, 9, _t(1, 9), pu={"borough_id": 1.0}, do={"borough_id": 1.0}),
        ["same_borough"],
    )
    assert batch["same_borough"] is True
    assert online["same_borough"] is True


def test_congestion_join_matches_across_time_units() -> None:
    trips = pl.LazyFrame(
        [(7, _t(1, 8, 0), _t(1, 8, 30), 10.0, 1800)],
        schema={
            "pu_zone": pl.UInt16,
            "request_datetime": pl.Datetime("ns", NYC_TZ),
            "dropoff_datetime": pl.Datetime("ns", NYC_TZ),
            "trip_miles": pl.Float64,
            "trip_duration_s": pl.Int64,
        },
        orient="row",
    )
    state = build_zone_state_frame(trips)
    request = pl.LazyFrame(
        {
            "pu_zone": pl.Series([7], dtype=pl.UInt16),
            "request_datetime": pl.Series([_t(1, 9, 0)], dtype=pl.Datetime("us", NYC_TZ)),
        }
    )
    out = attach_congestion(request, state).collect()
    assert out["completed_60m"][0] == 1


def test_reference_speed_prefers_observed_night_speed(tmp_path: object) -> None:
    import pathlib

    from eta.features.congestion import MIN_NIGHT_TRIPS, build_zone_history

    d = pathlib.Path(str(tmp_path))
    zone = 5
    night_rows = [(zone, _t(1, 3), 8.0, 3600, zone) for _ in range(MIN_NIGHT_TRIPS)]
    pl.DataFrame(
        night_rows,
        schema={
            "pu_zone": pl.UInt16,
            "request_datetime": pl.Datetime("us", NYC_TZ),
            "trip_miles": pl.Float64,
            "trip_duration_s": pl.Int64,
            "do_zone": pl.UInt16,
        },
        orient="row",
    ).with_columns(
        pl.lit("train").alias("split"),
        pl.col("request_datetime").alias("dropoff_datetime"),
        pl.lit(3600).alias("total_time_s"),
    ).write_parquet(d / "enriched_synth.parquet")

    matrix = pl.DataFrame(
        {
            "pu_zone": pl.Series([zone], dtype=pl.UInt16),
            "do_zone": pl.Series([zone + 1], dtype=pl.UInt16),
            "route_distance_m": pl.Series([9000.0], dtype=pl.Float32),
            "free_flow_duration_s": pl.Series([300.0], dtype=pl.Float32),
            "is_intra_zone": [False],
        }
    )
    hist = build_zone_history(d / "enriched_*.parquet", matrix, "train")
    row = hist.filter(pl.col("zone_id") == zone).row(0, named=True)

    assert row["reference_observed"] is True
    night_speed_ms = 8.0 * 1609.344 / 3600.0
    assert row["reference_speed_ms"] == pytest.approx(night_speed_ms, rel=1e-3)
    assert row["reference_speed_ms"] != pytest.approx(row["routed_speed_ms"])


def _night_trips(zone: int, mph: float, split: str, n: int) -> pl.DataFrame:
    """n identical 3am trips in `zone` at `mph`, tagged with `split`."""
    rows = [(zone, _t(1, 3), mph, 3600, zone) for _ in range(n)]
    return pl.DataFrame(
        rows,
        schema={
            "pu_zone": pl.UInt16,
            "request_datetime": pl.Datetime("us", NYC_TZ),
            "trip_miles": pl.Float64,
            "trip_duration_s": pl.Int64,
            "do_zone": pl.UInt16,
        },
        orient="row",
    ).with_columns(
        pl.lit(split).alias("split"),
        pl.col("request_datetime").alias("dropoff_datetime"),
        pl.lit(3600).alias("total_time_s"),
    )


def _tiny_matrix(zone: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pu_zone": pl.Series([zone], dtype=pl.UInt16),
            "do_zone": pl.Series([zone + 1], dtype=pl.UInt16),
            "route_distance_m": pl.Series([9000.0], dtype=pl.Float32),
            "free_flow_duration_s": pl.Series([300.0], dtype=pl.Float32),
            "is_intra_zone": [False],
        }
    )


def test_reference_speed_is_built_from_the_train_split_only(tmp_path: object) -> None:
    """Adversarial provenance check.

    The reference is frozen into an artifact that cal/val/test and serving all read,
    so a non-train row leaking in would contaminate every downstream feature value.
    Poison the held-out splits with absurdly fast night trips; the artifact must not
    move by a single bit.
    """
    import pathlib

    from eta.features.congestion import MIN_NIGHT_TRIPS, build_zone_history

    zone = 5
    clean_dir = pathlib.Path(str(tmp_path)) / "clean"
    poisoned_dir = pathlib.Path(str(tmp_path)) / "poisoned"
    clean_dir.mkdir()
    poisoned_dir.mkdir()

    honest = _night_trips(zone, 8.0, "train", MIN_NIGHT_TRIPS)
    honest.write_parquet(clean_dir / "enriched_a.parquet")

    # 200mph night trips in every split we are not allowed to look at.
    poison = pl.concat(
        [_night_trips(zone, 200.0, s, MIN_NIGHT_TRIPS * 5) for s in ("cal", "val", "test")]
    )
    pl.concat([honest, poison]).write_parquet(poisoned_dir / "enriched_a.parquet")

    matrix = _tiny_matrix(zone)
    clean = build_zone_history(clean_dir / "enriched_*.parquet", matrix, "train")
    poisoned = build_zone_history(poisoned_dir / "enriched_*.parquet", matrix, "train")

    assert clean.equals(poisoned), (
        "held-out night trips changed the reference speed artifact -- the split "
        "filter in observed_night_speed is not holding"
    )
    honest_speed = 8.0 * 1609.344 / 3600.0
    assert poisoned.filter(pl.col("zone_id") == zone)["reference_speed_ms"][0] == pytest.approx(
        honest_speed, rel=1e-3
    )


def test_reference_speed_falls_back_to_routed_below_the_night_trip_floor(
    tmp_path: object,
) -> None:
    import pathlib

    from eta.features.congestion import MIN_NIGHT_TRIPS, build_zone_history

    d = pathlib.Path(str(tmp_path))
    zone = 5
    _night_trips(zone, 8.0, "train", MIN_NIGHT_TRIPS - 1).write_parquet(d / "enriched_a.parquet")

    hist = build_zone_history(d / "enriched_*.parquet", _tiny_matrix(zone), "train")
    row = hist.filter(pl.col("zone_id") == zone).row(0, named=True)

    assert row["reference_observed"] is False
    assert row["reference_speed_ms"] == pytest.approx(row["routed_speed_ms"])
    assert row["reference_speed_ms"] == pytest.approx(9000.0 / 300.0, rel=1e-3)
