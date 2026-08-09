from __future__ import annotations

import argparse
import datetime as dt
import sys

from eta.config import get_settings
from eta.data.card import write_data_card
from eta.data.enrich import enrich_all
from eta.data.ingest import (
    download_months,
    download_zone_lookup,
    ingest_month,
    month_range,
    write_audit,
)
from eta.data.weather import build_hourly_weather
from eta.logging import bind_request_id, configure_logging, get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eta.data", description="TLC + weather ingest")
    parser.add_argument(
        "stage",
        choices=("download", "trips", "weather", "enrich", "card", "all"),
        nargs="?",
        default="all",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level)
    bind_request_id()

    paths = settings.paths.resolve()
    raw_tlc = paths["raw_dir"] / "tlc"
    raw_weather = paths["raw_dir"] / "weather"
    processed = paths["processed_dir"]

    holdout_end = settings.splits.test_end + dt.timedelta(weeks=settings.splits.holdout_weeks)
    months = month_range(settings.splits.train_start, holdout_end)
    log.info(
        "ingest_plan",
        months=len(months),
        first=months[0],
        last=months[-1],
        holdout_end=holdout_end.isoformat(),
    )

    if args.stage in ("download", "all"):
        download_months(months, raw_tlc)
        download_zone_lookup(paths["raw_dir"])

    if args.stage in ("trips", "all"):
        present: list[str] = []
        missing: list[str] = []
        for m in months:
            src = raw_tlc / f"fhvhv_tripdata_{m}.parquet"
            (present if src.exists() else missing).append(m)
        if missing:
            log.warning("tlc_months_missing", missing=missing, present=len(present))

        reports = [
            ingest_month(raw_tlc / f"fhvhv_tripdata_{m}.parquet", processed / "trips")
            for m in present
        ]
        write_audit(reports, paths["reports_dir"] / "filters.json", missing_months=missing)

    if args.stage in ("weather", "all"):
        years = sorted({settings.splits.train_start.year, settings.splits.test_end.year})
        weather = build_hourly_weather(raw_weather, years)
        out = processed / "weather_hourly.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        weather.write_parquet(out, compression="zstd")
        log.info("weather_written", path=str(out), rows=weather.height)

    if args.stage in ("enrich", "all"):
        enrich_all(settings)

    if args.stage in ("card", "all"):
        write_data_card(settings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
