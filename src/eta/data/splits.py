from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl

from eta.types import polars_enum

if TYPE_CHECKING:
    from eta.config import SplitConfig

__all__ = [
    "SPLIT_COLUMN",
    "Split",
    "assign_split",
    "holdout_end",
    "holdout_week_index",
    "split_boundaries",
]

SPLIT_COLUMN = "split"


class Split(StrEnum):
    TRAIN = "train"
    CAL = "cal"
    VAL = "val"
    TEST = "test"
    HOLDOUT = "holdout"
    BEYOND = "beyond"


def holdout_end(splits: SplitConfig) -> dt.date:
    return splits.test_end + dt.timedelta(weeks=splits.holdout_weeks)


def split_boundaries(splits: SplitConfig) -> list[tuple[Split, dt.date, dt.date]]:
    return [
        (Split.TRAIN, splits.train_start, splits.train_end),
        (Split.CAL, splits.train_end, splits.cal_end),
        (Split.VAL, splits.cal_end, splits.val_end),
        (Split.TEST, splits.val_end, splits.test_end),
        (Split.HOLDOUT, splits.test_end, holdout_end(splits)),
    ]


def assign_split(lf: pl.LazyFrame, splits: SplitConfig) -> pl.LazyFrame:
    day = pl.col("request_datetime").dt.date()
    expr: Any = pl.when(day < splits.train_start).then(pl.lit(Split.BEYOND.value))
    for name, _start, end in split_boundaries(splits):
        expr = expr.when(day < end).then(pl.lit(name.value))
    final: pl.Expr = expr.otherwise(pl.lit(Split.BEYOND.value))
    return lf.with_columns(final.cast(polars_enum(Split)).alias(SPLIT_COLUMN))


def holdout_week_index(lf: pl.LazyFrame, splits: SplitConfig) -> pl.LazyFrame:
    days = (pl.col("request_datetime").dt.date() - splits.test_end).dt.total_days()
    return lf.with_columns(
        pl.when(pl.col(SPLIT_COLUMN) == Split.HOLDOUT.value)
        .then((days // 7).clip(0, 255).cast(pl.UInt8))
        .otherwise(None)
        .alias("holdout_week")
    )
