from __future__ import annotations

from typing import TYPE_CHECKING, Final

import polars as pl

from eta.data.segments import SEGMENT_COLUMNS
from eta.data.splits import SPLIT_COLUMN, Split
from eta.data.target import COMPONENT_COLUMNS
from eta.features.congestion import build_zone_state
from eta.features.context import load_static_tables
from eta.features.families import register_all
from eta.features.pipeline import assemble
from eta.features.registry import REGISTRY
from eta.logging import get_logger
from eta.models.sampling import HASH_COLUMN, Tier, assign_hash, tier_filter

if TYPE_CHECKING:
    from pathlib import Path

    from eta.config import Settings

__all__ = ["TARGET", "build_matrix", "feature_names", "load_split", "population_digest"]

log = get_logger(__name__)

TARGET: Final = "total_time_s"

CARRY_COLUMNS: Final = (
    TARGET,
    "request_datetime",
    "pu_zone",
    "do_zone",
    "hvfhs_license_num",
    SPLIT_COLUMN,
    *SEGMENT_COLUMNS,
    # Carried for the Phase 6 decomposition ablation. Null on Lyft rows, which is
    # exactly why that ablation is Uber-only and says so.
    *COMPONENT_COLUMNS,
)


def feature_names() -> list[str]:
    register_all()
    return list(REGISTRY.names)


def build_matrix(settings: Settings, tier: Tier = Tier.TUNE) -> Path:
    register_all()
    paths = settings.paths.resolve()
    processed = paths["processed_dir"]
    enriched = processed / "enriched" / "enriched_*.parquet"
    out = processed / f"model_matrix_{tier.value}.parquet"

    static = load_static_tables(processed)
    zone_state = build_zone_state(enriched)

    requests = assign_hash(pl.scan_parquet(enriched)).filter(
        tier_filter(tier) & pl.col(SPLIT_COLUMN).is_in([s.value for s in Split][:4])
    )

    assembled = assemble(requests, static, zone_state)
    ctx_frame = assembled
    from eta.features.context import BatchContext

    features = REGISTRY.batch_frame(BatchContext(frame=ctx_frame))
    combined = pl.concat(
        [ctx_frame.select([c for c in CARRY_COLUMNS if c != HASH_COLUMN]), features],
        how="horizontal",
    )
    combined.sink_parquet(out, compression="zstd")

    rows = int(pl.scan_parquet(out).select(pl.len()).collect().item())
    log.info(
        "model_matrix_built",
        tier=tier.value,
        rows=rows,
        features=len(REGISTRY),
        path=str(out),
        bytes=out.stat().st_size,
    )
    return out


def load_split(path: Path, split: Split) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col(SPLIT_COLUMN) == split.value)
        .collect(engine="streaming")
    )


def population_digest(frame: pl.DataFrame) -> str:
    import hashlib

    from eta.types import stat_float

    keyed = frame.select(
        pl.col("request_datetime").cast(pl.Int64),
        pl.col("pu_zone").cast(pl.Int64),
        pl.col("do_zone").cast(pl.Int64),
    ).sort("request_datetime", "pu_zone", "do_zone")
    digest = hashlib.sha256()
    digest.update(str(keyed.height).encode())
    for col in keyed.columns:
        for stat in (keyed[col].sum(), keyed[col].min(), keyed[col].max()):
            digest.update(str(stat_float(stat)).encode())
    return digest.hexdigest()[:16]
