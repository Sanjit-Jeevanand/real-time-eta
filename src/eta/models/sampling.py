from __future__ import annotations

from enum import StrEnum
from typing import Final

import polars as pl

__all__ = [
    "HASH_COLUMN",
    "SAMPLE_KEY",
    "Tier",
    "assign_hash",
    "tier_filter",
]

HASH_COLUMN: Final = "sample_hash"

SAMPLE_KEY: Final = (
    "request_datetime",
    "pu_zone",
    "do_zone",
    "hvfhs_license_num",
)

HASH_MODULUS: Final = 10_000


class Tier(StrEnum):
    TUNE = "tune"
    ABLATION = "ablation"
    FULL = "full"


TIER_BASIS_POINTS: Final[dict[Tier, int]] = {
    Tier.TUNE: 300,
    Tier.ABLATION: 2_500,
    Tier.FULL: 10_000,
}


def assign_hash(lf: pl.LazyFrame, columns: tuple[str, ...] = SAMPLE_KEY) -> pl.LazyFrame:
    present = [c for c in columns if c in lf.collect_schema().names()]
    if not present:
        msg = f"none of the sampling key columns are present: {columns}"
        raise ValueError(msg)
    return lf.with_columns(
        (pl.struct(present).hash(seed=0) % HASH_MODULUS).cast(pl.UInt16).alias(HASH_COLUMN)
    )


def tier_filter(tier: Tier) -> pl.Expr:
    return pl.col(HASH_COLUMN) < TIER_BASIS_POINTS[tier]
