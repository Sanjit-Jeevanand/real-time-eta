from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from eta.data.schema import NYC_TZ
from eta.features.temporal import CYCLICAL_COLUMNS, cyclical_encodings, is_holiday
from eta.routing.detour import build_detour_ratios
from eta.routing.matrix import intra_zone_estimate, validate_matrix
from eta.routing.osrm import RouteLeg, parse_route

TZ = ZoneInfo(NYC_TZ)


def _step(
    distance: float, maneuver: str, modifier: str = "left", classes: list[str] | None = None
) -> dict[str, object]:
    return {
        "distance": distance,
        "maneuver": {"type": maneuver, "modifier": modifier},
        "intersections": [{"classes": classes}] if classes else [{}],
    }


def _payload(steps: list[dict[str, object]], distance: float, duration: float) -> dict[str, object]:
    return {
        "code": "Ok",
        "routes": [{"distance": distance, "duration": duration, "legs": [{"steps": steps}]}],
    }


def test_parse_route_counts_turns_and_highway_share() -> None:
    steps = [
        _step(100.0, "depart", "straight"),
        _step(800.0, "on ramp", "slight right", ["motorway"]),
        _step(50.0, "turn", "left"),
        _step(50.0, "arrive", "straight"),
    ]
    leg = parse_route(_payload(steps, 1000.0, 120.0))
    assert leg is not None
    assert leg.distance_m == 1000.0
    assert leg.duration_s == 120.0
    assert leg.turn_count == 2
    assert leg.highway_fraction == pytest.approx(0.8)


def test_straight_continue_is_not_a_turn() -> None:
    steps = [_step(500.0, "turn", "straight"), _step(500.0, "continue", "straight")]
    leg = parse_route(_payload(steps, 1000.0, 100.0))
    assert leg is not None
    assert leg.turn_count == 0


def test_highway_fraction_is_capped_at_one() -> None:
    steps = [_step(900.0, "depart", "straight", ["motorway"])]
    leg = parse_route(_payload(steps, 500.0, 60.0))
    assert leg is not None
    assert leg.highway_fraction == 1.0


def test_parse_route_returns_none_when_unroutable() -> None:
    assert parse_route({"code": "NoRoute", "routes": []}) is None


def test_zero_distance_route_does_not_divide_by_zero() -> None:
    leg = parse_route(_payload([_step(0.0, "depart", "straight")], 0.0, 0.0))
    assert leg is not None
    assert leg.highway_fraction == 0.0


def test_intra_zone_estimate_scales_with_area() -> None:
    small, _ = intra_zone_estimate(1.0, 8.0)
    large, _ = intra_zone_estimate(4.0, 8.0)
    assert large == pytest.approx(2 * small)
    assert small == pytest.approx(0.5214 * 1000.0)


def test_intra_zone_duration_follows_speed() -> None:
    distance, duration = intra_zone_estimate(1.0, 10.0)
    assert duration == pytest.approx(distance / 10.0)
    _, slow = intra_zone_estimate(1.0, 5.0)
    assert slow > duration


def test_intra_zone_estimate_survives_zero_area() -> None:
    distance, duration = intra_zone_estimate(0.0, 8.0)
    assert distance > 0
    assert math.isfinite(duration)


def _matrix(rows: list[tuple[int, int, float, float, int, float, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(*r, r[6]) for r in rows],
        schema={
            "pu_zone": pl.UInt16,
            "do_zone": pl.UInt16,
            "route_distance_m": pl.Float32,
            "free_flow_duration_s": pl.Float32,
            "turn_count": pl.UInt16,
            "highway_fraction": pl.Float32,
            "is_intra_zone": pl.Boolean,
            "is_estimated": pl.Boolean,
        },
        orient="row",
    )


def test_validate_matrix_requires_every_pair() -> None:
    df = _matrix([(1, 1, 500.0, 60.0, 0, 0.0, True), (1, 2, 900.0, 90.0, 3, 0.2, False)])
    with pytest.raises(ValueError, match="expected 4"):
        validate_matrix(df, n_zones=2)


def test_validate_matrix_rejects_zero_distance() -> None:
    df = _matrix(
        [
            (1, 1, 0.0, 60.0, 0, 0.0, True),
            (1, 2, 900.0, 90.0, 3, 0.2, False),
            (2, 1, 900.0, 90.0, 3, 0.2, False),
            (2, 2, 500.0, 60.0, 0, 0.0, True),
        ]
    )
    with pytest.raises(ValueError, match="non-positive route distance"):
        validate_matrix(df, n_zones=2)


def test_validate_matrix_accepts_a_complete_grid() -> None:
    df = _matrix(
        [
            (1, 1, 500.0, 60.0, 0, 0.0, True),
            (1, 2, 900.0, 90.0, 3, 0.2, False),
            (2, 1, 900.0, 90.0, 3, 0.2, False),
            (2, 2, 500.0, 60.0, 0, 0.0, True),
        ]
    )
    validate_matrix(df, n_zones=2)


def test_cyclical_encoding_is_continuous_across_midnight() -> None:
    frame = pl.DataFrame(
        {
            "request_datetime": [
                dt.datetime(2023, 6, 1, 23, 59, tzinfo=TZ),
                dt.datetime(2023, 6, 2, 0, 1, tzinfo=TZ),
            ]
        }
    ).with_columns(cyclical_encodings())
    a = (frame["hour_sin"][0], frame["hour_cos"][0])
    b = (frame["hour_sin"][1], frame["hour_cos"][1])
    gap = math.dist(a, b)
    assert gap < 0.02, "23:59 and 00:01 must be adjacent on the circle"


def test_cyclical_encoding_is_continuous_across_the_week() -> None:
    frame = pl.DataFrame(
        {
            "request_datetime": [
                dt.datetime(2023, 6, 4, 12, tzinfo=TZ),
                dt.datetime(2023, 6, 5, 12, tzinfo=TZ),
            ]
        }
    ).with_columns(cyclical_encodings())
    gap = math.dist(
        (frame["dow_sin"][0], frame["dow_cos"][0]),
        (frame["dow_sin"][1], frame["dow_cos"][1]),
    )
    assert gap == pytest.approx(2 * math.sin(math.pi / 7), abs=1e-4)


def test_cyclical_columns_are_unit_circle() -> None:
    frame = pl.DataFrame(
        {"request_datetime": [dt.datetime(2023, 6, 1, h, tzinfo=TZ) for h in range(24)]}
    ).with_columns(cyclical_encodings())
    assert set(CYCLICAL_COLUMNS) <= set(frame.columns)
    radius = (frame["hour_sin"] ** 2 + frame["hour_cos"] ** 2).to_numpy()
    assert all(abs(r - 1.0) < 1e-5 for r in radius)


def test_holiday_flag() -> None:
    frame = pl.DataFrame(
        {
            "request_datetime": [
                dt.datetime(2023, 7, 4, 12, tzinfo=TZ),
                dt.datetime(2023, 7, 5, 12, tzinfo=TZ),
            ]
        }
    ).with_columns(is_holiday())
    assert frame["is_holiday"].to_list() == [True, False]


def test_detour_ratio_needs_enough_trips(tmp_path: object) -> None:
    import pathlib

    d = pathlib.Path(str(tmp_path))
    pl.DataFrame(
        {
            "pu_zone": pl.Series([1] * 40 + [2] * 5, dtype=pl.UInt16),
            "do_zone": pl.Series([2] * 40 + [1] * 5, dtype=pl.UInt16),
            "trip_miles": [2.0] * 40 + [2.0] * 5,
            "split": ["train"] * 45,
        }
    ).write_parquet(d / "enriched_2023-01.parquet")

    matrix = _matrix(
        [
            (1, 2, 1609.344, 200.0, 3, 0.0, False),
            (2, 1, 1609.344, 200.0, 3, 0.0, False),
        ]
    )
    out = build_detour_ratios(d / "enriched_*.parquet", matrix, min_trips=30)
    by_pair = {(r["pu_zone"], r["do_zone"]): r for r in out.iter_rows(named=True)}

    assert by_pair[(1, 2)]["structural_detour_ratio"] == pytest.approx(2.0, abs=1e-3)
    assert by_pair[(1, 2)]["detour_trips"] == 40
    assert by_pair[(2, 1)]["structural_detour_ratio"] is None
    assert by_pair[(2, 1)]["detour_trips"] == 5


def test_detour_ratio_uses_training_split_only(tmp_path: object) -> None:
    import pathlib

    d = pathlib.Path(str(tmp_path))
    pl.DataFrame(
        {
            "pu_zone": pl.Series([1] * 40, dtype=pl.UInt16),
            "do_zone": pl.Series([2] * 40, dtype=pl.UInt16),
            "trip_miles": [2.0] * 40,
            "split": ["test"] * 40,
        }
    ).write_parquet(d / "enriched_2023-01.parquet")

    matrix = _matrix([(1, 2, 1609.344, 200.0, 3, 0.0, False)])
    out = build_detour_ratios(d / "enriched_*.parquet", matrix, min_trips=30)
    assert out["structural_detour_ratio"][0] is None


def test_route_leg_is_immutable() -> None:
    leg = RouteLeg(1000.0, 120.0, 3, 0.5)
    with pytest.raises((AttributeError, TypeError)):
        leg.distance_m = 2000.0  # type: ignore[misc]


def test_validate_matrix_rejects_too_many_estimated_pairs() -> None:
    rows = [(a, b, 500.0, 60.0, 0, 0.0, a == b) for a in range(1, 5) for b in range(1, 5)]
    df = pl.DataFrame(
        [(*r, True) for r in rows],
        schema={
            "pu_zone": pl.UInt16,
            "do_zone": pl.UInt16,
            "route_distance_m": pl.Float32,
            "free_flow_duration_s": pl.Float32,
            "turn_count": pl.UInt16,
            "highway_fraction": pl.Float32,
            "is_intra_zone": pl.Boolean,
            "is_estimated": pl.Boolean,
        },
        orient="row",
    )
    with pytest.raises(ValueError, match="estimated pairs"):
        validate_matrix(df, n_zones=4)


def test_validate_matrix_allows_the_intra_zone_diagonal() -> None:
    rows = [(a, b, 500.0, 60.0, 0, 0.0, a == b) for a in range(1, 5) for b in range(1, 5)]
    df = pl.DataFrame(
        [(*r, r[6]) for r in rows],
        schema={
            "pu_zone": pl.UInt16,
            "do_zone": pl.UInt16,
            "route_distance_m": pl.Float32,
            "free_flow_duration_s": pl.Float32,
            "turn_count": pl.UInt16,
            "highway_fraction": pl.Float32,
            "is_intra_zone": pl.Boolean,
            "is_estimated": pl.Boolean,
        },
        orient="row",
    )
    validate_matrix(df, n_zones=4)


def test_intra_zone_distances_use_the_training_split_only(tmp_path: object) -> None:
    import pathlib

    from eta.routing.matrix import intra_zone_distances

    d = pathlib.Path(str(tmp_path))
    pl.DataFrame(
        {
            "pu_zone": pl.Series([7] * 500 + [7] * 500, dtype=pl.UInt16),
            "do_zone": pl.Series([7] * 500 + [7] * 500, dtype=pl.UInt16),
            "trip_miles": [1.0] * 500 + [99.0] * 500,
            "split": ["train"] * 500 + ["test"] * 500,
        }
    ).write_parquet(d / "enriched_2023-01.parquet")

    zones = pl.DataFrame(
        {"zone_id": pl.Series([7], dtype=pl.UInt16), "area_km2": pl.Series([2.0], dtype=pl.Float32)}
    )
    out = intra_zone_distances(d / "enriched_*.parquet", zones, min_trips=200)
    assert out["from_observation"][0]
    assert out["distance_m"][0] == pytest.approx(1609.344, rel=1e-3)


def test_holdout_rows_cannot_move_the_intra_zone_map(tmp_path: object) -> None:
    import pathlib

    from eta.routing.matrix import intra_zone_distances

    d = pathlib.Path(str(tmp_path))
    zones = pl.DataFrame(
        {"zone_id": pl.Series([7], dtype=pl.UInt16), "area_km2": pl.Series([2.0], dtype=pl.Float32)}
    )
    train_only = pl.DataFrame(
        {
            "pu_zone": pl.Series([7] * 400, dtype=pl.UInt16),
            "do_zone": pl.Series([7] * 400, dtype=pl.UInt16),
            "trip_miles": [1.0] * 400,
            "split": ["train"] * 400,
        }
    )
    train_only.write_parquet(d / "enriched_2023-01.parquet")
    before = intra_zone_distances(d / "enriched_*.parquet", zones, min_trips=200)["distance_m"][0]

    for split in ("cal", "val", "test", "holdout"):
        contaminated = pl.concat(
            [
                train_only,
                pl.DataFrame(
                    {
                        "pu_zone": pl.Series([7] * 5000, dtype=pl.UInt16),
                        "do_zone": pl.Series([7] * 5000, dtype=pl.UInt16),
                        "trip_miles": [50.0] * 5000,
                        "split": [split] * 5000,
                    }
                ),
            ]
        )
        contaminated.write_parquet(d / "enriched_2023-01.parquet")
        after = intra_zone_distances(d / "enriched_*.parquet", zones, min_trips=200)["distance_m"][
            0
        ]
        assert after == pytest.approx(before), f"{split} rows leaked into the intra-zone map"


def test_routing_manifest_pins_the_inputs() -> None:
    from eta.config import get_settings
    from eta.routing.manifest import build_manifest

    settings = get_settings()
    paths = settings.paths.resolve()
    m = build_manifest(
        settings,
        paths["processed_dir"] / "zone_pair_matrix.parquet",
        paths["processed_dir"] / "zones.parquet",
    )
    assert m["router"]["image"].startswith("ghcr.io/project-osrm/osrm-backend:v")
    assert m["router"]["algorithm"] == "mld"
    assert m["zone_geometry"]["crs_source"] == "EPSG:2263"
    assert "TRAIN split only" in m["provenance"]["intra_zone_distance"]
    assert set(m["provenance"]) == {
        "intra_zone_distance",
        "structural_detour_ratio",
        "zone_embeddings",
    }
