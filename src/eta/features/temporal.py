from __future__ import annotations

import datetime as dt
import math
from typing import Final

import polars as pl

__all__ = [
    "CYCLICAL_COLUMNS",
    "US_HOLIDAYS_2023_2024",
    "cyclical_encodings",
    "is_holiday",
]

HOURS_PER_DAY: Final = 24
DAYS_PER_WEEK: Final = 7

CYCLICAL_COLUMNS: Final = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

US_HOLIDAYS_2023_2024: Final = (
    "2023-01-01",
    "2023-01-16",
    "2023-02-20",
    "2023-05-29",
    "2023-06-19",
    "2023-07-04",
    "2023-09-04",
    "2023-10-09",
    "2023-11-10",
    "2023-11-23",
    "2023-12-25",
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
)


def _sin_cos(value: pl.Expr, period: int, name: str) -> list[pl.Expr]:
    angle = value * (2.0 * math.pi / period)
    return [
        angle.sin().cast(pl.Float32).alias(f"{name}_sin"),
        angle.cos().cast(pl.Float32).alias(f"{name}_cos"),
    ]


def cyclical_encodings(column: str = "request_datetime") -> list[pl.Expr]:
    ts = pl.col(column)
    hour_of_day = ts.dt.hour() + ts.dt.minute() / 60.0
    day_of_week = ts.dt.weekday() - 1
    return [
        *_sin_cos(hour_of_day, HOURS_PER_DAY, "hour"),
        *_sin_cos(day_of_week, DAYS_PER_WEEK, "dow"),
    ]


def is_holiday(column: str = "request_datetime") -> pl.Expr:
    dates = [dt.date.fromisoformat(d) for d in US_HOLIDAYS_2023_2024]
    return pl.col(column).dt.date().is_in(dates).alias("is_holiday")
