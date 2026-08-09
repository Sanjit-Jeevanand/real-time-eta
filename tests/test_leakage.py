from __future__ import annotations

import datetime as dt
from itertools import pairwise
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from eta.config import SegmentConfig, SplitConfig, get_settings
from eta.data.schema import EVAL_ONLY_COLUMNS, NYC_TZ, POST_HOC_COLUMNS, REQUEST_TIME_KNOWN
from eta.data.segments import SEGMENT_COLUMNS, assign_segments, zone_density_map
from eta.data.splits import (
    SPLIT_COLUMN,
    Split,
    assign_split,
    holdout_end,
    holdout_week_index,
    split_boundaries,
)

pytestmark = pytest.mark.leakage

TZ = ZoneInfo(NYC_TZ)

SPLITS = SplitConfig(
    train_start=dt.date(2023, 1, 1),
    train_end=dt.date(2023, 10, 1),
    cal_end=dt.date(2023, 11, 15),
    val_end=dt.date(2023, 12, 15),
    test_end=dt.date(2024, 1, 15),
    holdout_weeks=8,
)


def _frame(days: list[dt.date]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {"request_datetime": [dt.datetime(d.year, d.month, d.day, 12, tzinfo=TZ) for d in days]}
    )


def test_split_boundaries_are_contiguous_and_ordered() -> None:
    bounds = split_boundaries(SPLITS)
    for (_, _, end), (_, next_start, _) in pairwise(bounds):
        assert end == next_start
    assert bounds[0][1] == SPLITS.train_start
    assert bounds[-1][2] == holdout_end(SPLITS)


def test_every_row_lands_in_exactly_one_split() -> None:
    days = [
        dt.date(2023, 6, 1),
        dt.date(2023, 10, 15),
        dt.date(2023, 12, 1),
        dt.date(2024, 1, 1),
        dt.date(2024, 2, 1),
    ]
    out = assign_split(_frame(days), SPLITS).collect()
    assert out[SPLIT_COLUMN].to_list() == [
        Split.TRAIN,
        Split.CAL,
        Split.VAL,
        Split.TEST,
        Split.HOLDOUT,
    ]


def test_boundary_dates_belong_to_the_later_split() -> None:
    out = assign_split(
        _frame([SPLITS.train_end, SPLITS.cal_end, SPLITS.val_end, SPLITS.test_end]), SPLITS
    ).collect()
    assert out[SPLIT_COLUMN].to_list() == [Split.CAL, Split.VAL, Split.TEST, Split.HOLDOUT]


def test_splits_are_temporally_disjoint() -> None:
    days = [SPLITS.train_start + dt.timedelta(days=7 * i) for i in range(60)]
    out = assign_split(_frame(days), SPLITS).collect()
    ranges = (
        out.group_by(SPLIT_COLUMN)
        .agg(
            pl.col("request_datetime").min().alias("lo"),
            pl.col("request_datetime").max().alias("hi"),
        )
        .sort("lo")
    )
    highs = ranges["hi"].to_list()
    lows = ranges["lo"].to_list()
    for hi, lo in zip(highs[:-1], lows[1:], strict=True):
        assert hi < lo


def test_split_assignment_uses_request_time_not_completion() -> None:
    late = pl.LazyFrame(
        {
            "request_datetime": [dt.datetime(2023, 9, 30, 23, 30, tzinfo=TZ)],
            "dropoff_datetime": [dt.datetime(2023, 10, 1, 0, 30, tzinfo=TZ)],
        }
    )
    out = assign_split(late, SPLITS).collect()
    assert out[SPLIT_COLUMN][0] == Split.TRAIN


def test_calibration_split_is_non_empty_and_distinct() -> None:
    assert SPLITS.calibration_days > 0
    bounds = {name: (s, e) for name, s, e in split_boundaries(SPLITS)}
    cal_start, cal_end = bounds[Split.CAL]
    train_start, train_end = bounds[Split.TRAIN]
    test_start, _test_end = bounds[Split.TEST]
    assert cal_start >= train_end
    assert cal_end <= test_start
    assert not (train_start <= cal_start < train_end)


def test_holdout_covers_the_configured_number_of_weeks() -> None:
    days = [SPLITS.test_end + dt.timedelta(days=d) for d in range(0, 7 * SPLITS.holdout_weeks, 3)]
    lf = holdout_week_index(assign_split(_frame(days), SPLITS), SPLITS)
    out = lf.collect()
    weeks = out.filter(pl.col(SPLIT_COLUMN) == Split.HOLDOUT)["holdout_week"]
    assert weeks.min() == 0
    assert weeks.max() == SPLITS.holdout_weeks - 1
    assert weeks.n_unique() == SPLITS.holdout_weeks


def test_holdout_week_is_null_outside_the_holdout() -> None:
    lf = holdout_week_index(assign_split(_frame([dt.date(2023, 6, 1)]), SPLITS), SPLITS)
    assert lf.collect()["holdout_week"][0] is None


def test_request_time_and_post_hoc_sets_are_disjoint() -> None:
    assert REQUEST_TIME_KNOWN.isdisjoint(POST_HOC_COLUMNS)
    assert EVAL_ONLY_COLUMNS.isdisjoint(REQUEST_TIME_KNOWN)


def test_target_and_its_components_are_post_hoc() -> None:
    for col in ("total_time_s", "dispatch_approach_s", "curb_wait_s", "trip_duration_s"):
        assert col in POST_HOC_COLUMNS
        assert col not in REQUEST_TIME_KNOWN


def test_actual_distance_and_duration_are_not_request_time_known() -> None:
    for col in ("trip_miles", "trip_time"):
        assert col in POST_HOC_COLUMNS
        assert col not in REQUEST_TIME_KNOWN


def test_trip_length_segment_is_marked_evaluation_only() -> None:
    assert "seg_trip_length" in EVAL_ONLY_COLUMNS
    assert "seg_trip_length" not in REQUEST_TIME_KNOWN
    servable = set(SEGMENT_COLUMNS) - EVAL_ONLY_COLUMNS
    assert servable <= REQUEST_TIME_KNOWN


def test_servable_segments_use_only_request_time_inputs() -> None:
    cfg = SegmentConfig()
    lookup = pl.DataFrame(
        {
            "zone_id": [132, 161, 61],
            "borough": ["Queens", "Manhattan", "Brooklyn"],
            "zone_name": ["JFK Airport", "Midtown Center", "Crown Heights"],
        }
    )
    density = zone_density_map(lookup, cfg)

    at_request_only = pl.LazyFrame(
        {
            "request_datetime": [dt.datetime(2023, 6, 1, 18, tzinfo=TZ)],
            "pu_zone": pl.Series([161], dtype=pl.UInt16),
            "temp_c": [20.0],
            "precip_mm_h": [0.0],
            "trip_miles": [None],
        }
    )
    out = assign_segments(at_request_only, cfg, density).collect()
    assert out["seg_time"][0] == "peak_pm"
    assert out["seg_zone_density"][0] == "manhattan_core"
    assert out["seg_weather"][0] == "clear"
    assert out["seg_trip_length"][0] is None


def test_fixture_matches_the_checked_in_config() -> None:
    get_settings.cache_clear()
    live = get_settings().splits
    assert live.train_start == SPLITS.train_start
    assert live.train_end == SPLITS.train_end
    assert live.cal_end == SPLITS.cal_end
    assert live.val_end == SPLITS.val_end
    assert live.test_end == SPLITS.test_end
    assert live.holdout_weeks == SPLITS.holdout_weeks


def test_unknown_distance_is_not_bucketed() -> None:
    cfg = SegmentConfig()
    density = zone_density_map(
        pl.DataFrame({"zone_id": [161], "borough": ["Manhattan"], "zone_name": ["Midtown"]}), cfg
    )
    lf = pl.LazyFrame(
        {
            "request_datetime": [dt.datetime(2023, 6, 1, 12, tzinfo=TZ)],
            "pu_zone": pl.Series([161], dtype=pl.UInt16),
            "temp_c": [10.0],
            "precip_mm_h": [0.0],
            "trip_miles": [None],
        }
    )
    assert assign_segments(lf, cfg, density).collect()["seg_trip_length"][0] is None
