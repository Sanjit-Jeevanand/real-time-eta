from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from eta.data.download import download
from eta.data.schema import (
    MAX_VALID_ZONE_ID,
    NYC_TZ,
    TLC_BASE_URL,
    TLC_READ_COLUMNS,
    UNKNOWN_ZONE_IDS,
    ZONE_LOOKUP_URL,
)
from eta.logging import get_logger

__all__ = [
    "FILTER_RULES",
    "FilterAudit",
    "FilterRule",
    "IngestReport",
    "apply_filters",
    "audit_filters",
    "download_months",
    "ingest_month",
    "month_range",
    "scan_month",
    "verify_readable",
]

log = get_logger(__name__)

MAX_TRIP_HOURS = 6
MAX_DISPATCH_HOURS = 2
MAX_PLAUSIBLE_MPH = 80.0


@dataclass(frozen=True)
class FilterRule:
    name: str
    reason: str
    keep: pl.Expr


@dataclass(frozen=True)
class FilterAudit:
    name: str
    reason: str
    rejected_alone: int
    pct_alone: float


@dataclass(frozen=True)
class IngestReport:
    month: str
    rows_in: int
    rows_out: int
    rows_dropped: int
    pct_dropped: float
    audits: list[FilterAudit]

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        return d


FILTER_RULES: tuple[FilterRule, ...] = (
    FilterRule(
        name="timestamps_present",
        reason="request/pickup/dropoff must all exist to form the target",
        keep=pl.col("request_datetime").is_not_null()
        & pl.col("pickup_datetime").is_not_null()
        & pl.col("dropoff_datetime").is_not_null(),
    ),
    FilterRule(
        name="timestamps_ordered",
        reason="request <= pickup <= dropoff; anything else is a clock or join error",
        keep=(pl.col("request_datetime") <= pl.col("pickup_datetime"))
        & (pl.col("pickup_datetime") <= pl.col("dropoff_datetime")),
    ),
    FilterRule(
        name="positive_duration",
        reason="non-positive total_time cannot be a promise",
        keep=(pl.col("dropoff_datetime") - pl.col("request_datetime")).dt.total_seconds() > 0,
    ),
    FilterRule(
        name="duration_under_6h",
        reason=f"trips over {MAX_TRIP_HOURS}h are meter-left-running artefacts, not rides",
        keep=(pl.col("dropoff_datetime") - pl.col("request_datetime")).dt.total_seconds()
        <= MAX_TRIP_HOURS * 3600,
    ),
    FilterRule(
        name="dispatch_under_2h",
        reason=f"waiting over {MAX_DISPATCH_HOURS}h for a car is a stuck record",
        keep=(pl.col("pickup_datetime") - pl.col("request_datetime")).dt.total_seconds()
        <= MAX_DISPATCH_HOURS * 3600,
    ),
    FilterRule(
        name="distance_consistent",
        reason="zero distance with non-zero trip time: the ride never moved",
        keep=~((pl.col("trip_miles") <= 0) & (pl.col("trip_time") > 0)),
    ),
    FilterRule(
        name="speed_plausible",
        reason=f"implied speed over {MAX_PLAUSIBLE_MPH:.0f} mph is impossible in NYC",
        keep=(pl.col("trip_time") <= 0)
        | ((pl.col("trip_miles") / (pl.col("trip_time") / 3600.0)) <= MAX_PLAUSIBLE_MPH),
    ),
    FilterRule(
        name="zones_routable",
        reason="zones 264/265 are Unknown/N-A and have no centroid to route from",
        keep=pl.col("PULocationID").is_between(1, MAX_VALID_ZONE_ID)
        & pl.col("DOLocationID").is_between(1, MAX_VALID_ZONE_ID)
        & ~pl.col("PULocationID").is_in(UNKNOWN_ZONE_IDS)
        & ~pl.col("DOLocationID").is_in(UNKNOWN_ZONE_IDS),
    ),
)


def month_range(start: dt.date, end: dt.date) -> list[str]:
    months: list[str] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def download_months(months: list[str], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for month in months:
        name = f"fhvhv_tripdata_{month}.parquet"
        final = dest / name
        if final.exists():
            log.info("tlc_download_skipped", month=month, path=str(final))
            paths.append(final)
            continue
        paths.append(download(f"{TLC_BASE_URL}/{name}", final))
    return paths


def download_zone_lookup(dest: Path) -> Path:
    return download(ZONE_LOOKUP_URL, dest / "taxi_zone_lookup.csv")


def scan_month(path: Path) -> pl.LazyFrame:
    ts_cols = (
        "request_datetime",
        "on_scene_datetime",
        "pickup_datetime",
        "dropoff_datetime",
    )
    return (
        pl.scan_parquet(path)
        .select(TLC_READ_COLUMNS)
        .with_columns(
            [
                pl.col(c)
                .dt.replace_time_zone(NYC_TZ, ambiguous="earliest", non_existent="null")
                .alias(c)
                for c in ts_cols
            ]
        )
    )


def audit_filters(lf: pl.LazyFrame) -> tuple[int, list[FilterAudit]]:
    aggs = [pl.len().alias("__rows")]
    aggs += [(~rule.keep).fill_null(True).sum().alias(rule.name) for rule in FILTER_RULES]
    row = lf.select(aggs).collect(engine="streaming").row(0, named=True)

    rows = int(row["__rows"])
    audits = [
        FilterAudit(
            name=rule.name,
            reason=rule.reason,
            rejected_alone=int(row[rule.name]),
            pct_alone=round(100.0 * int(row[rule.name]) / rows, 4) if rows else 0.0,
        )
        for rule in FILTER_RULES
    ]
    return rows, audits


def apply_filters(lf: pl.LazyFrame) -> pl.LazyFrame:
    keep = pl.lit(value=True)
    for rule in FILTER_RULES:
        keep = keep & rule.keep.fill_null(value=False)
    return lf.filter(keep)


def verify_readable(path: Path) -> None:
    try:
        pl.scan_parquet(path).select(pl.col("request_datetime").max()).collect(engine="streaming")
    except Exception as exc:
        msg = (
            f"{path.name} is corrupt (footer valid, pages truncated). "
            f"Delete it and re-download: rm {path} && make data-download"
        )
        raise RuntimeError(msg) from exc


def ingest_month(src: Path, dest_dir: Path) -> IngestReport:
    from eta.data.target import TARGET_COLUMN, with_target

    month = src.stem.removeprefix("fhvhv_tripdata_")
    verify_readable(src)
    lf = scan_month(src)

    rows_in, audits = audit_filters(lf)

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"trips_{month}.parquet"
    filtered = (
        with_target(apply_filters(lf))
        .with_columns(
            pl.col("PULocationID").cast(pl.UInt16).alias("pu_zone"),
            pl.col("DOLocationID").cast(pl.UInt16).alias("do_zone"),
            pl.col(TARGET_COLUMN).cast(pl.Int32),
        )
        .drop("PULocationID", "DOLocationID")
    )

    filtered.sink_parquet(out, compression="zstd")

    rows_out = int(pl.scan_parquet(out).select(pl.len()).collect().item())
    dropped = rows_in - rows_out

    report = IngestReport(
        month=month,
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped=dropped,
        pct_dropped=round(100.0 * dropped / rows_in, 4) if rows_in else 0.0,
        audits=audits,
    )
    log.info(
        "tlc_month_ingested",
        month=month,
        rows_in=rows_in,
        rows_out=rows_out,
        pct_dropped=report.pct_dropped,
    )
    return report


def write_audit(
    reports: list[IngestReport],
    path: Path,
    *,
    missing_months: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    totals_in = sum(r.rows_in for r in reports)
    totals_out = sum(r.rows_out for r in reports)

    by_rule: dict[str, int] = {}
    for r in reports:
        for a in r.audits:
            by_rule[a.name] = by_rule.get(a.name, 0) + a.rejected_alone

    payload = {
        "generated_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "months_audited": len(reports),
        "months_missing": missing_months or [],
        "rows_in": totals_in,
        "rows_out": totals_out,
        "rows_dropped": totals_in - totals_out,
        "pct_dropped": round(100.0 * (totals_in - totals_out) / totals_in, 4) if totals_in else 0.0,
        "rules": [
            {
                "name": rule.name,
                "reason": rule.reason,
                "rejected_alone": by_rule.get(rule.name, 0),
                "pct_alone": round(100.0 * by_rule.get(rule.name, 0) / totals_in, 4)
                if totals_in
                else 0.0,
            }
            for rule in FILTER_RULES
        ],
        "per_month": [r.to_dict() for r in reports],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("filter_audit_written", path=str(path), rows_in=totals_in, rows_out=totals_out)
