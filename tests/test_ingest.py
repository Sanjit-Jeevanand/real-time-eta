from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from eta.data.ingest import (
    FILTER_RULES,
    apply_filters,
    audit_filters,
    month_range,
    scan_month,
)
from eta.data.schema import NYC_TZ
from eta.data.target import TARGET_COLUMN, components_reconcile, with_target

TZ = ZoneInfo(NYC_TZ)


def _t(day: int, hour: int, minute: int = 0) -> dt.datetime:
    return dt.datetime(2023, 6, day, hour, minute, tzinfo=TZ)


@pytest.fixture
def frame() -> pl.LazyFrame:
    rows = [
        ("HV0003", _t(1, 8), _t(1, 8, 8), _t(1, 8, 10), _t(1, 8, 30), 100, 200, 4.0, 1200),
        ("HV0003", _t(2, 8), _t(2, 8, 8), _t(2, 8, 10), None, 100, 200, 4.0, 1200),
        ("HV0003", _t(3, 8), _t(3, 8, 8), _t(3, 8, 30), _t(3, 8, 10), 100, 200, 4.0, 1200),
        ("HV0003", _t(4, 8), _t(4, 8, 8), _t(4, 8, 10), _t(4, 15), 100, 200, 40.0, 24000),
        ("HV0003", _t(5, 8), _t(5, 10, 8), _t(5, 11), _t(5, 11, 20), 100, 200, 4.0, 1200),
        ("HV0003", _t(6, 8), _t(6, 8, 8), _t(6, 8, 10), _t(6, 8, 30), 100, 200, 0.0, 1200),
        ("HV0003", _t(7, 8), _t(7, 8, 8), _t(7, 8, 10), _t(7, 8, 30), 100, 200, 60.0, 600),
        ("HV0003", _t(8, 8), _t(8, 8, 8), _t(8, 8, 10), _t(8, 8, 30), 264, 200, 4.0, 1200),
        ("HV0005", _t(9, 8), None, _t(9, 8, 10), _t(9, 8, 30), 100, 200, 4.0, 1200),
    ]
    return pl.LazyFrame(
        rows,
        schema={
            "hvfhs_license_num": pl.String,
            "request_datetime": pl.Datetime("us", NYC_TZ),
            "on_scene_datetime": pl.Datetime("us", NYC_TZ),
            "pickup_datetime": pl.Datetime("us", NYC_TZ),
            "dropoff_datetime": pl.Datetime("us", NYC_TZ),
            "PULocationID": pl.Int64,
            "DOLocationID": pl.Int64,
            "trip_miles": pl.Float64,
            "trip_time": pl.Int64,
        },
        orient="row",
    )


def test_every_rule_fires_and_overlaps_are_visible(frame: pl.LazyFrame) -> None:
    rows, audits = audit_filters(frame)
    assert rows == 9
    by_name = {a.name: a.rejected_alone for a in audits}

    assert by_name == {
        "timestamps_present": 1,
        "timestamps_ordered": 2,
        "positive_duration": 1,
        "duration_under_6h": 2,
        "dispatch_under_2h": 1,
        "distance_consistent": 1,
        "speed_plausible": 1,
        "zones_routable": 1,
    }
    for rule in FILTER_RULES:
        assert by_name[rule.name] >= 1, f"{rule.name} caught nothing"


def test_only_the_clean_rows_survive(frame: pl.LazyFrame) -> None:
    out = apply_filters(frame).collect()
    assert out.height == 2
    assert set(out["hvfhs_license_num"]) == {"HV0003", "HV0005"}


def test_lyft_rows_survive_without_on_scene(frame: pl.LazyFrame) -> None:
    out = apply_filters(frame).collect()
    lyft = out.filter(pl.col("hvfhs_license_num") == "HV0005")
    assert lyft.height == 1
    assert lyft["on_scene_datetime"].null_count() == 1


def test_nulls_do_not_survive_by_three_valued_logic() -> None:
    lf = pl.LazyFrame(
        [("HV0003", None, None, None, None, 100, 200, 4.0, 1200)],
        schema={
            "hvfhs_license_num": pl.String,
            "request_datetime": pl.Datetime("us", NYC_TZ),
            "on_scene_datetime": pl.Datetime("us", NYC_TZ),
            "pickup_datetime": pl.Datetime("us", NYC_TZ),
            "dropoff_datetime": pl.Datetime("us", NYC_TZ),
            "PULocationID": pl.Int64,
            "DOLocationID": pl.Int64,
            "trip_miles": pl.Float64,
            "trip_time": pl.Int64,
        },
        orient="row",
    )
    assert apply_filters(lf).collect().height == 0


def test_audit_percentages_are_bounded(frame: pl.LazyFrame) -> None:
    _, audits = audit_filters(frame)
    assert all(0.0 <= a.pct_alone <= 100.0 for a in audits)
    assert all(a.reason for a in audits), "every rule must carry a stated reason"


def test_target_is_request_to_dropoff(frame: pl.LazyFrame) -> None:
    out = with_target(apply_filters(frame)).collect()
    assert out[TARGET_COLUMN].to_list() == [1800, 1800]


def test_components_sum_to_the_target(frame: pl.LazyFrame) -> None:
    out = with_target(apply_filters(frame)).collect()
    uber = out.filter(pl.col("hvfhs_license_num") == "HV0003")
    total = uber["dispatch_approach_s"][0] + uber["curb_wait_s"][0] + uber["trip_duration_s"][0]
    assert total == uber[TARGET_COLUMN][0]
    assert components_reconcile(with_target(apply_filters(frame))).collect().height == 0


def test_components_are_null_not_zero_without_on_scene(frame: pl.LazyFrame) -> None:
    out = with_target(apply_filters(frame)).collect()
    lyft = out.filter(pl.col("hvfhs_license_num") == "HV0005")
    assert lyft["dispatch_approach_s"][0] is None
    assert lyft["curb_wait_s"][0] is None
    assert lyft[TARGET_COLUMN][0] == 1800


def test_month_range_spans_the_configured_window() -> None:
    months = month_range(dt.date(2023, 1, 1), dt.date(2024, 1, 15))
    assert months[0] == "2023-01"
    assert months[-1] == "2024-01"
    assert len(months) == 13


def test_month_range_single_month() -> None:
    assert month_range(dt.date(2023, 5, 3), dt.date(2023, 5, 28)) == ["2023-05"]


def test_scan_localises_to_new_york(tmp_path: object) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "fhvhv_tripdata_2023-11.parquet"

    def naive_ts(hour: int, minute: int) -> dt.datetime:
        return dt.datetime(2023, 11, 5, hour, minute)

    naive = pl.DataFrame(
        {
            "hvfhs_license_num": ["HV0003"],
            "request_datetime": [naive_ts(1, 30)],
            "on_scene_datetime": [naive_ts(1, 35)],
            "pickup_datetime": [naive_ts(1, 40)],
            "dropoff_datetime": [naive_ts(2, 5)],
            "PULocationID": [100],
            "DOLocationID": [200],
            "trip_miles": [4.0],
            "trip_time": [1200],
        }
    )
    naive.write_parquet(p)

    out = scan_month(p).collect()
    dtype = out["request_datetime"].dtype
    assert isinstance(dtype, pl.Datetime)
    assert dtype.time_zone == NYC_TZ
    assert out["request_datetime"][0].utcoffset() == dt.timedelta(hours=-4)


def test_corrupt_parquet_fails_with_a_named_error(tmp_path: object) -> None:
    import pathlib

    from eta.data.ingest import verify_readable

    p = pathlib.Path(str(tmp_path)) / "fhvhv_tripdata_2023-09.parquet"
    pl.DataFrame(
        {
            "request_datetime": [dt.datetime(2023, 9, 1, 8, tzinfo=TZ)],
            "trip_miles": [1.0],
        }
    ).write_parquet(p)

    good = p.read_bytes()
    verify_readable(p)

    corrupt = bytearray(good)
    for i in range(4, min(len(corrupt) - 8, 400)):
        corrupt[i] = 0
    p.write_bytes(bytes(corrupt))

    with pytest.raises(RuntimeError, match=r"fhvhv_tripdata_2023-09\.parquet is corrupt"):
        verify_readable(p)


def test_null_predicate_row_is_dropped_not_kept() -> None:
    lf = pl.LazyFrame(
        [("HV0003", _t(1, 8), _t(1, 8, 8), _t(1, 8, 10), _t(1, 8, 30), 100, 200, None, 1200)],
        schema={
            "hvfhs_license_num": pl.String,
            "request_datetime": pl.Datetime("us", NYC_TZ),
            "on_scene_datetime": pl.Datetime("us", NYC_TZ),
            "pickup_datetime": pl.Datetime("us", NYC_TZ),
            "dropoff_datetime": pl.Datetime("us", NYC_TZ),
            "PULocationID": pl.Int64,
            "DOLocationID": pl.Int64,
            "trip_miles": pl.Float64,
            "trip_time": pl.Int64,
        },
        orient="row",
    )
    assert apply_filters(lf).collect().height == 0

    _, audits = audit_filters(lf)
    by_name = {a.name: a.rejected_alone for a in audits}
    assert by_name["distance_consistent"] == 1
    assert by_name["speed_plausible"] == 1
    assert by_name["timestamps_present"] == 0
