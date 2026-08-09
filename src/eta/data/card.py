from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, cast

import polars as pl

from eta.data.schema import EVAL_ONLY_COLUMNS, POST_HOC_COLUMNS, REQUEST_TIME_KNOWN
from eta.data.segments import SEGMENT_COLUMNS
from eta.data.splits import SPLIT_COLUMN, holdout_end, split_boundaries
from eta.data.target import COMPONENT_COLUMNS, TARGET_COLUMN
from eta.data.weather import WEATHER_COLUMNS
from eta.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from eta.config import Settings

__all__ = ["write_data_card"]

log = get_logger(__name__)

QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)


def _fmt(n: int) -> str:
    return f"{n:,}"


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f}"


def _overall(lf: pl.LazyFrame) -> dict[str, float]:
    aggs = [
        pl.len().alias("rows"),
        pl.col(TARGET_COLUMN).mean().alias("mean"),
        pl.col(TARGET_COLUMN).std().alias("std"),
        pl.col(TARGET_COLUMN).min().alias("min"),
        pl.col(TARGET_COLUMN).max().alias("max"),
    ]
    aggs += [pl.col(TARGET_COLUMN).quantile(q).alias(f"q{q}") for q in QUANTILES]
    row = lf.select(aggs).collect(engine="streaming").row(0, named=True)
    return {k: float(v) for k, v in row.items()}


def _missingness(lf: pl.LazyFrame, columns: tuple[str, ...]) -> dict[str, float]:
    exprs = [pl.col(c).is_null().mean().alias(c) for c in columns]
    row = lf.select(exprs).collect(engine="streaming").row(0, named=True)
    return {k: 100.0 * float(v) for k, v in row.items()}


def _by(lf: pl.LazyFrame, column: str) -> pl.DataFrame:
    return (
        lf.group_by(column)
        .agg(
            pl.len().alias("rows"),
            pl.col(TARGET_COLUMN).median().alias("p50"),
            pl.col(TARGET_COLUMN).quantile(0.9).alias("p90"),
        )
        .sort(column)
        .collect(engine="streaming")
    )


def _segment_table(lf: pl.LazyFrame, column: str, total: int) -> list[str]:
    df = _by(lf, column)
    lines = ["| bucket | rows | share | p50 (min) | p90 (min) |", "|---|---:|---:|---:|---:|"]
    for r in df.iter_rows(named=True):
        share = 100.0 * r["rows"] / total
        lines.append(
            f"| {r[column]} | {_fmt(r['rows'])} | {share:.1f}% | "
            f"{_minutes(r['p50'])} | {_minutes(r['p90'])} |"
        )
    return lines


def write_data_card(settings: Settings) -> Path:
    paths = settings.paths.resolve()
    src = paths["processed_dir"] / "enriched"
    lf = pl.scan_parquet(src / "enriched_*.parquet")

    stats = _overall(lf)
    total = int(stats["rows"])
    filters = json.loads((paths["reports_dir"] / "filters.json").read_text())

    split_df = _by(lf, SPLIT_COLUMN)
    comp_missing = _missingness(lf, COMPONENT_COLUMNS)
    weather_missing = _missingness(lf, WEATHER_COLUMNS)

    out: list[str] = []
    a = out.append

    a("# Data Card")
    a("")
    a(f"Generated {dt.datetime.now(tz=dt.UTC).date().isoformat()} by `make data-card`.")
    a("")
    a("## Source")
    a("")
    a("| | |")
    a("|---|---|")
    a("| Dataset | NYC TLC High Volume For-Hire Vehicle trip records |")
    a("| Operators | Uber (HV0003), Lyft (HV0005) |")
    a(f"| Window | {settings.splits.train_start} to {holdout_end(settings.splits)} |")
    a(f"| Months | {filters['months_audited']} |")
    a("| Weather | NOAA ISD hourly, Central Park / LGA / JFK |")
    a("| Licence | Public domain (NYC Open Data) |")
    a("")

    a("## Volume")
    a("")
    a("| | rows |")
    a("|---|---:|")
    a(f"| Raw | {_fmt(filters['rows_in'])} |")
    a(f"| Removed by filters | {_fmt(filters['rows_dropped'])} ({filters['pct_dropped']}%) |")
    a(f"| Kept | {_fmt(filters['rows_out'])} |")
    a(f"| Enriched | {_fmt(total)} |")
    a("")

    a("## Filter summary")
    a("")
    a("Each rule is evaluated independently, so the counts overlap and do not sum")
    a("to the net removal. A row with a null timestamp trips four rules at once.")
    a("")
    a("| rule | rejected alone | % | reason |")
    a("|---|---:|---:|---|")
    for r in sorted(filters["rules"], key=lambda x: -x["rejected_alone"]):
        a(f"| `{r['name']}` | {_fmt(r['rejected_alone'])} | {r['pct_alone']:.3f} | {r['reason']} |")
    independent = sum(r["rejected_alone"] for r in filters["rules"])
    a("")
    a(
        f"Independent rejections total {_fmt(independent)} against "
        f"{_fmt(filters['rows_dropped'])} net removals "
        f"({independent / filters['rows_dropped']:.2f}x overlap)."
    )
    a("")

    a("## Target")
    a("")
    a("`total_time_s = dropoff_datetime - request_datetime`, the user-facing promise.")
    a("Predicted from request-time information only.")
    a("")
    a("| statistic | minutes |")
    a("|---|---:|")
    a(f"| mean | {_minutes(stats['mean'])} |")
    a(f"| std | {_minutes(stats['std'])} |")
    a(f"| min | {_minutes(stats['min'])} |")
    for q in QUANTILES:
        a(f"| p{int(q * 100)} | {_minutes(stats[f'q{q}'])} |")
    a(f"| max | {_minutes(stats['max'])} |")
    a("")
    ratio = stats["mean"] / stats["q0.5"]
    a(
        f"Mean/median ratio {ratio:.3f}. The distribution has a long right tail: "
        f"p99 is {stats['q0.99'] / stats['q0.5']:.1f}x the median. This is why the "
        "project optimises a quantile of the predictive distribution rather than its mean."
    )
    a("")

    a("## Splits")
    a("")
    a("Wall-clock, never random. `cal` is held out from both training and test so")
    a("isotonic recalibration and CQR have data neither fitted nor evaluated on.")
    a("")
    a("| split | from | to | rows | share |")
    a("|---|---|---|---:|---:|")
    bounds = {name.value: (start, end) for name, start, end in split_boundaries(settings.splits)}
    for r in split_df.iter_rows(named=True):
        name = str(r[SPLIT_COLUMN])
        start, end = bounds.get(name, ("-", "-"))
        share = 100.0 * r["rows"] / total
        a(f"| {name} | {start} | {end} | {_fmt(r['rows'])} | {share:.1f}% |")
    a("")

    a("## Segments")
    a("")
    a("Frozen here and not re-cut afterwards: the Phase 7 miscalibration result and")
    a("the Phase 10 cost attribution are only comparable on a fixed grid.")
    a("")
    for col in SEGMENT_COLUMNS:
        a(f"### `{col}`")
        a("")
        if col in EVAL_ONLY_COLUMNS:
            a("**Evaluation only.** Derived from `trip_miles`, the distance actually")
            a("travelled, which is not known when the ETA is quoted. Safe for slicing")
            a("results; unusable as a model feature or as a Mondrian CQR conditioning")
            a("variable until Phase 3 supplies OSRM route distance as a request-time proxy.")
            a("")
        a("\n".join(_segment_table(lf, col, total)))
        a("")

    a("## Mondrian grid viability")
    a("")
    a("Segment-conditional CQR needs a calibration set per cell of the full grid.")
    a("Cells thinner than `calibration.min_segment_samples` fall back to the global map,")
    a("so the count below is what decides whether conditional coverage is reachable.")
    a("")
    grid = (
        lf.filter(pl.col(SPLIT_COLUMN) == "cal")
        .group_by(list(SEGMENT_COLUMNS))
        .agg(pl.len().alias("rows"))
        .collect(engine="streaming")
        .sort("rows")
    )
    floor = settings.calibration.min_segment_samples
    thin = grid.filter(pl.col("rows") < floor)
    a("| | |")
    a("|---|---:|")
    a(f"| Cells in the grid | {_fmt(grid.height)} |")
    a(f"| Calibration rows | {_fmt(int(grid['rows'].sum()))} |")
    a(f"| Cells below {_fmt(floor)} rows | {_fmt(thin.height)} |")
    a(f"| Rows in those cells | {_fmt(int(thin['rows'].sum()))} |")
    smallest = cast("int", grid["rows"].min() or 0)
    a(f"| Smallest cell | {_fmt(smallest)} |")
    a("")
    if thin.height:
        a("Thinnest cells:")
        a("")
        a("| " + " | ".join(SEGMENT_COLUMNS) + " | rows |")
        a("|" + "---|" * (len(SEGMENT_COLUMNS) + 1))
        for r in thin.head(8).iter_rows(named=True):
            cells = " | ".join(str(r[c]) for c in SEGMENT_COLUMNS)
            a(f"| {cells} | {_fmt(r['rows'])} |")
        a("")
    a("")

    a("## Missingness")
    a("")
    a("| column | null % | note |")
    a("|---|---:|---|")
    for c, pct in comp_missing.items():
        note = (
            "absent when `on_scene_datetime` is not reported"
            if c != "trip_duration_s"
            else "derived from pickup/dropoff only, so always present"
        )
        a(f"| `{c}` | {pct:.2f} | {note} |")
    for c, pct in weather_missing.items():
        a(f"| `{c}` | {pct:.2f} | NOAA gap beyond the 2h forward-fill |")
    a("")
    a(
        "`on_scene_datetime` is reported by Uber and effectively absent for Lyft, so the "
        "component decomposition covers only part of the data. The target itself is "
        "complete. Components are left null rather than zero-filled: a zero would read "
        "as an instant pickup and bias the decomposition toward the operator that "
        "reports the field."
    )
    a("")

    a("## Leakage boundary")
    a("")
    a(f"Known at request time ({len(REQUEST_TIME_KNOWN)} columns):")
    a("")
    a(", ".join(f"`{c}`" for c in sorted(REQUEST_TIME_KNOWN)))
    a("")
    a(f"Post-hoc, never a feature ({len(POST_HOC_COLUMNS)} columns):")
    a("")
    a(", ".join(f"`{c}`" for c in sorted(POST_HOC_COLUMNS)))
    a("")
    a("Enforced by `tests/test_leakage.py`, which runs in CI.")
    a("")

    a("## Known limitations")
    a("")
    a(
        f"- Unroutable zones (264 Unknown, 265 N/A) account for "
        f"{filters['rules'][-1]['pct_alone']:.2f}% of raw rows and are dropped. These are "
        "legitimate trips whose endpoint the feed never resolved, not corrupt records. "
        "Coverage limitation, not a cleaning statistic."
    )
    a(
        "- Weather station assignment is a placeholder: airport zones take their own "
        "station, everything else takes Central Park. Phase 3 replaces it with true "
        "nearest-centroid once zone geometry exists."
    )
    a(
        "- Zone-level geography only. TLC stopped publishing coordinates in 2016, so "
        "there is no sub-zone spatial resolution."
    )
    a(
        "- Quoted ETAs affect cancellation, and cancelled trips never enter this feed. "
        "The dataset is censored by the very predictions it is used to train."
    )
    a("")

    dest = paths["reports_dir"].parent / "DATA_CARD.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    log.info("data_card_written", path=str(dest), rows=total)
    return dest
