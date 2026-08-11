from __future__ import annotations

import datetime as dt
import random
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from eta.data.schema import NYC_TZ
from eta.features.congestion import build_zone_state
from eta.features.context import StaticTables
from eta.features.families import register_all
from eta.features.parity import FLOAT_TOLERANCE, check_parity
from eta.features.registry import REGISTRY
from eta.features.replay import CLOCK_KEY, DictStore, ReplayStream, replay_to_store

pytestmark = pytest.mark.parity

TZ = ZoneInfo(NYC_TZ)
register_all()

PARITY_ROWS = 10_000
N_ZONES = 24
EMB_DIMS = 4
START = dt.datetime(2023, 6, 1, tzinfo=TZ)


def _static() -> StaticTables:
    rng = random.Random(0)
    ids = list(range(1, N_ZONES + 1))
    zones = pl.DataFrame(
        {
            "zone_id": pl.Series(ids, dtype=pl.UInt16),
            "centroid_lat": [40.7 + i * 0.001 for i in ids],
            "centroid_lon": [-73.9 - i * 0.001 for i in ids],
            "area_km2": pl.Series([0.5 + i * 0.1 for i in ids], dtype=pl.Float32),
            "borough": [("Manhattan", "Brooklyn", "Queens", "Bronx")[i % 4] for i in ids],
            "zone_name": [f"zone {i}" for i in ids],
            "is_airport": [i in (3, 7) for i in ids],
        }
    )
    pairs = [(a, b) for a in ids for b in ids]
    matrix = pl.DataFrame(
        {
            "pu_zone": pl.Series([a for a, _ in pairs], dtype=pl.UInt16),
            "do_zone": pl.Series([b for _, b in pairs], dtype=pl.UInt16),
            "route_distance_m": pl.Series(
                [500.0 + 900.0 * abs(a - b) for a, b in pairs], dtype=pl.Float32
            ),
            "free_flow_duration_s": pl.Series(
                [60.0 + 70.0 * abs(a - b) for a, b in pairs], dtype=pl.Float32
            ),
            "turn_count": pl.Series([abs(a - b) % 17 for a, b in pairs], dtype=pl.UInt16),
            "highway_fraction": pl.Series(
                [min(1.0, abs(a - b) / 20.0) for a, b in pairs], dtype=pl.Float32
            ),
            "is_intra_zone": [a == b for a, b in pairs],
            "is_estimated": [a == b for a, b in pairs],
        }
    )
    detour = pl.DataFrame(
        {
            "pu_zone": pl.Series([a for a, _ in pairs], dtype=pl.UInt16),
            "do_zone": pl.Series([b for _, b in pairs], dtype=pl.UInt16),
            "structural_detour_ratio": pl.Series(
                [None if (a + b) % 11 == 0 else 0.9 + 0.02 * ((a * b) % 20) for a, b in pairs],
                dtype=pl.Float32,
            ),
            "detour_trips": pl.Series([(a * b) % 400 for a, b in pairs], dtype=pl.UInt32),
        }
    )
    embeddings = pl.DataFrame(
        {"zone_id": pl.Series(ids, dtype=pl.UInt16)}
        | {
            f"zone_emb_{d}": pl.Series([rng.uniform(-1.0, 1.0) for _ in ids], dtype=pl.Float32)
            for d in range(EMB_DIMS)
        }
    )
    history = pl.DataFrame(
        {
            "zone_id": pl.Series(ids, dtype=pl.UInt16),
            "hist_mean_duration_s": pl.Series([900.0 + 20.0 * i for i in ids], dtype=pl.Float32),
            "free_flow_speed_ms": pl.Series([6.0 + 0.2 * i for i in ids], dtype=pl.Float32),
            "hist_trips": pl.Series([1000 + i for i in ids], dtype=pl.UInt32),
        }
    )
    return StaticTables(zones, matrix, detour, embeddings, history)


def _trips(n: int, seed: int = 1) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        pu = rng.randint(1, N_ZONES)
        do = rng.randint(1, N_ZONES)
        req = START + dt.timedelta(minutes=rng.randint(0, 60 * 24 * 3))
        dur = rng.randint(120, 3600)
        rows.append(
            (
                pl.Series([pu], dtype=pl.UInt16)[0],
                req,
                req + dt.timedelta(seconds=dur),
                rng.uniform(0.3, 18.0),
                dur,
                do,
                rng.uniform(-5.0, 32.0),
                rng.uniform(0.0, 12.0),
                rng.uniform(200.0, 16000.0),
                rng.choice([0.0, 0.0, 0.0, 0.4, 2.5]),
                rng.choice([0.0, 0.0, 3.0]),
                i,
            )
        )
    return pl.DataFrame(
        rows,
        schema={
            "pu_zone": pl.UInt16,
            "request_datetime": pl.Datetime("us", NYC_TZ),
            "dropoff_datetime": pl.Datetime("us", NYC_TZ),
            "trip_miles": pl.Float64,
            "trip_duration_s": pl.Int64,
            "do_zone": pl.UInt16,
            "temp_c": pl.Float32,
            "wind_ms": pl.Float32,
            "visibility_m": pl.Float32,
            "precip_mm_h": pl.Float32,
            "snow_depth_cm": pl.Float32,
            "row_id": pl.Int64,
        },
        orient="row",
    )


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> tuple[pl.DataFrame, pl.LazyFrame]:
    trips = _trips(PARITY_ROWS)
    d = tmp_path_factory.mktemp("parity")
    trips.write_parquet(d / "enriched_synth.parquet")
    return trips, build_zone_state(d / "enriched_*.parquet")


def test_parity_gate_on_ten_thousand_rows(
    sample: tuple[pl.DataFrame, pl.LazyFrame],
) -> None:
    trips, state = sample
    report = check_parity(trips.lazy(), _static(), state)
    assert report.rows == PARITY_ROWS
    assert report.features == len(REGISTRY) == 44
    assert report.ok, report.summary()


def test_parity_reports_a_planted_divergence(
    sample: tuple[pl.DataFrame, pl.LazyFrame],
) -> None:
    trips, state = sample
    feature = REGISTRY["route_free_flow_speed_ms"]
    original = feature.online

    def perturbed(ctx: object) -> float:
        return float(original(ctx)) * 1.5  # type: ignore[arg-type]

    object.__setattr__(feature, "online", perturbed)
    try:
        report = check_parity(trips.head(200).lazy(), _static(), state)
        assert not report.ok
        assert "route_free_flow_speed_ms" in report.mismatches
        assert "FAILED" in report.summary()
    finally:
        object.__setattr__(feature, "online", original)


def test_float_tolerance_is_tight() -> None:
    assert FLOAT_TOLERANCE <= 1e-5


# ------------------------------------------------------------- replay -------
def test_replay_reaches_the_same_totals_as_the_batch_state(
    tmp_path: object,
) -> None:
    import pathlib

    d = pathlib.Path(str(tmp_path))
    trips = _trips(2_000, seed=7)
    trips.write_parquet(d / "enriched_synth.parquet")

    store = DictStore()
    stream = replay_to_store(d / "enriched_*.parquet", store)
    assert stream.buckets_seen > 0
    assert CLOCK_KEY in store.data

    batch = build_zone_state(d / "enriched_*.parquet").collect()
    last = batch.sort("bucket").group_by("zone").last()
    checked = 0
    for row in last.iter_rows(named=True):
        zone = int(row["zone"])
        if row["bucket"] != stream._clock:
            continue
        assert stream.snapshot(zone)[f"cong:completed:{zone}:60m"] == pytest.approx(
            float(row["completed_60m"])
        ), zone
        checked += 1
    assert checked >= 1, "no zone shared the final bucket; nothing was compared"


def test_replay_is_a_sliding_window_not_a_cumulative_sum() -> None:
    store = DictStore()
    stream = ReplayStream(store=store, windows=(15,))
    t0 = START
    stream.advance(t0, {5: (10.0, 1000.0, 600.0)})
    assert stream.snapshot(5)["cong:completed:5:15m"] == 10.0

    stream.advance(t0 + dt.timedelta(minutes=20), {})
    assert stream.snapshot(5)["cong:completed:5:15m"] == 0.0, (
        "a completion 20 minutes old must have left the 15-minute window"
    )


def test_replay_refuses_to_go_backwards() -> None:
    stream = ReplayStream(store=DictStore(), windows=(15,))
    stream.advance(START + dt.timedelta(minutes=10), {})
    with pytest.raises(ValueError, match="replay went backwards"):
        stream.advance(START, {})


def test_replay_publishes_the_stream_clock() -> None:
    store = DictStore()
    stream = ReplayStream(store=store, windows=(15,))
    when = START + dt.timedelta(minutes=35)
    stream.advance(when, {1: (1.0, 100.0, 60.0)})
    assert store.data[CLOCK_KEY] == when.isoformat()


def test_replay_never_sees_a_trip_before_it_completes() -> None:
    store = DictStore()
    stream = ReplayStream(store=store, windows=(60,))
    stream.advance(START, {})
    assert stream.snapshot(9)["cong:completed:9:60m"] == 0.0
    stream.advance(START + dt.timedelta(minutes=30), {9: (1.0, 5000.0, 1800.0)})
    assert stream.snapshot(9)["cong:completed:9:60m"] == 1.0
