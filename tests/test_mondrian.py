from __future__ import annotations

import polars as pl
import pytest

from eta.calibration.mondrian import (
    GLOBAL_CELL,
    build_cell_plan,
    parent_cells,
    resolve_cell,
)

pytestmark = pytest.mark.calibration

AXES = ("seg_time", "seg_zone_density", "seg_weather")


def _cell(*pairs: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(pairs)


def test_parents_are_ordered_most_specific_first() -> None:
    cell = _cell(("a", "1"), ("b", "2"), ("c", "3"))
    parents = parent_cells(cell)
    assert [len(p) for p in parents] == [2, 2, 2, 1, 1, 1, 0]
    assert parents[-1] == GLOBAL_CELL


def test_exact_cell_is_used_when_it_meets_the_floor() -> None:
    cell = _cell(("a", "1"), ("b", "2"))
    res = resolve_cell(cell, {cell: 10_000}, floor=5_000)
    assert res.is_exact
    assert res.n == 10_000
    assert res.dropped_axes == ()


def test_thin_cell_borrows_from_the_nearest_sufficient_parent() -> None:
    cell = _cell(("time", "late_night"), ("density", "airport"), ("weather", "rain"))
    counts = {
        cell: 86,
        _cell(("density", "airport"), ("weather", "rain")): 400,
        _cell(("time", "late_night"), ("density", "airport")): 9_000,
        _cell(("density", "airport")): 50_000,
    }
    res = resolve_cell(cell, counts, floor=5_000)
    assert res.served_by == _cell(("time", "late_night"), ("density", "airport"))
    assert res.n == 9_000
    assert res.dropped_axes == ("weather",)
    assert not res.is_exact
    assert not res.is_global


def test_fallback_prefers_two_axes_over_one() -> None:
    cell = _cell(("a", "1"), ("b", "2"), ("c", "3"))
    counts = {cell: 10, _cell(("a", "1"), ("c", "3")): 6_000, _cell(("a", "1")): 90_000}
    res = resolve_cell(cell, counts, floor=5_000)
    assert len(res.served_by) == 2
    assert res.n == 6_000


def test_global_fallback_is_the_last_resort() -> None:
    cell = _cell(("a", "1"), ("b", "2"))
    counts = {cell: 3, _cell(("a", "1")): 10, _cell(("b", "2")): 20, GLOBAL_CELL: 100_000}
    res = resolve_cell(cell, counts, floor=5_000)
    assert res.is_global
    assert res.n == 100_000
    assert set(res.dropped_axes) == {"a", "b"}


def test_missing_cell_is_treated_as_empty_not_an_error() -> None:
    cell = _cell(("a", "1"), ("b", "2"))
    res = resolve_cell(cell, {GLOBAL_CELL: 50_000}, floor=5_000)
    assert res.is_global
    assert res.n == 50_000


def _frame(rows: list[tuple[str, str, str]], times: list[int]) -> pl.DataFrame:
    data: dict[str, list[str]] = {"seg_time": [], "seg_zone_density": [], "seg_weather": []}
    for (t, d, w), n in zip(rows, times, strict=True):
        data["seg_time"] += [t] * n
        data["seg_zone_density"] += [d] * n
        data["seg_weather"] += [w] * n
    return pl.DataFrame(data)


def test_plan_counts_exact_fallback_and_global() -> None:
    frame = _frame(
        [
            ("peak_pm", "manhattan_core", "clear"),
            ("peak_pm", "airport", "clear"),
            ("peak_pm", "airport", "rain"),
        ],
        [20_000, 6_000, 50],
    )
    plan = build_cell_plan(frame, AXES, floor=5_000)
    assert plan.exact == 2
    assert plan.fell_back == 1
    assert plan.global_fallback == 0

    thin = _cell(("seg_time", "peak_pm"), ("seg_zone_density", "airport"), ("seg_weather", "rain"))
    res = plan.resolution_for(thin)
    assert not res.is_exact
    assert res.served_by == _cell(("seg_time", "peak_pm"), ("seg_zone_density", "airport"))
    assert res.n == 6_050


def test_plan_never_serves_a_cell_below_the_floor_unless_global() -> None:
    frame = _frame(
        [("peak_am", "outer_borough", "clear"), ("late_night", "airport", "snow")],
        [30_000, 20],
    )
    plan = build_cell_plan(frame, AXES, floor=5_000)
    for res in plan.resolutions.values():
        assert res.n >= plan.floor or res.is_global


def test_every_full_cell_present_in_calibration_gets_a_resolution() -> None:
    frame = _frame(
        [
            ("peak_am", "manhattan_core", "clear"),
            ("peak_pm", "outer_borough", "rain"),
            ("off_peak", "airport", "snow"),
        ],
        [9_000, 8_000, 30],
    )
    plan = build_cell_plan(frame, AXES, floor=5_000)
    assert len(plan.resolutions) == 3
    assert all(r.served_by is not None for r in plan.resolutions.values())


def test_lower_floor_yields_more_exact_cells() -> None:
    frame = _frame(
        [("peak_pm", "airport", "rain"), ("peak_pm", "manhattan_core", "clear")],
        [1_000, 40_000],
    )
    strict = build_cell_plan(frame, AXES, floor=5_000)
    lenient = build_cell_plan(frame, AXES, floor=500)
    assert lenient.exact > strict.exact


def test_hierarchy_does_not_depend_on_the_data() -> None:
    cell = _cell(("a", "1"), ("b", "2"), ("c", "3"))
    assert parent_cells(cell) == parent_cells(cell)
    import inspect

    params = set(inspect.signature(parent_cells).parameters)
    assert params == {"cell"}, "the hierarchy must be a pure function of the cell"


def test_tie_break_is_deterministic_and_follows_axis_order() -> None:
    cell = _cell(("seg_time", "peak_pm"), ("seg_zone_density", "airport"), ("seg_weather", "rain"))
    counts = {
        cell: 10,
        _cell(("seg_time", "peak_pm"), ("seg_zone_density", "airport")): 9_000,
        _cell(("seg_time", "peak_pm"), ("seg_weather", "rain")): 9_000,
        _cell(("seg_zone_density", "airport"), ("seg_weather", "rain")): 9_000,
    }
    picks = {resolve_cell(cell, counts, floor=5_000).served_by for _ in range(20)}
    assert len(picks) == 1
    assert picks.pop() == _cell(("seg_time", "peak_pm"), ("seg_zone_density", "airport"))


def test_plan_is_reproducible_across_runs() -> None:
    frame = _frame(
        [("peak_pm", "airport", "rain"), ("peak_pm", "manhattan_core", "clear")],
        [50, 40_000],
    )
    a = build_cell_plan(frame, AXES, floor=5_000)
    b = build_cell_plan(frame, AXES, floor=5_000)
    assert {k: v.served_by for k, v in a.resolutions.items()} == {
        k: v.served_by for k, v in b.resolutions.items()
    }
