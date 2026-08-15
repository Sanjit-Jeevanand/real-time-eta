from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD

from eta.data.splits import SPLIT_COLUMN, Split
from eta.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["build_zone_embeddings", "embedding_columns"]

log = get_logger(__name__)


def embedding_columns(dims: int) -> list[str]:
    return [f"zone_emb_{i}" for i in range(dims)]


def _cooccurrence(lf: pl.LazyFrame, zone_ids: list[int]) -> np.ndarray:
    counts = (
        lf.filter(pl.col(SPLIT_COLUMN) == Split.TRAIN.value)
        .group_by("pu_zone", "do_zone")
        .agg(pl.len().alias("trips"))
        .collect(engine="streaming")
    )
    index = {z: i for i, z in enumerate(zone_ids)}
    n = len(zone_ids)
    matrix = np.zeros((n, n), dtype=np.float64)
    for pu, do, trips in counts.iter_rows():
        i, j = index.get(int(pu)), index.get(int(do))
        if i is not None and j is not None:
            matrix[i, j] = float(trips)
    return matrix


def build_zone_embeddings(
    enriched_glob: Path, zones: pl.DataFrame, dims: int, seed: int = 0
) -> pl.DataFrame:
    zone_ids = [int(z) for z in zones["zone_id"].to_list()]
    matrix = _cooccurrence(pl.scan_parquet(enriched_glob), zone_ids)

    weighted = np.log1p(matrix)
    symmetric = np.hstack([weighted, weighted.T])

    svd = TruncatedSVD(n_components=dims, random_state=seed)
    embedded = svd.fit_transform(symmetric)
    explained = float(svd.explained_variance_ratio_.sum())

    embedded = embedded - embedded.mean(axis=0, keepdims=True)

    log.info(
        "zone_embeddings_built",
        zones=len(zone_ids),
        dims=dims,
        explained_variance=round(explained, 4),
        empty_rows=int((matrix.sum(axis=1) == 0).sum()),
        centred=True,
    )

    out = pl.DataFrame(
        {"zone_id": pl.Series(zone_ids, dtype=pl.UInt16)}
        | {
            name: pl.Series(embedded[:, i].astype(np.float32))
            for i, name in enumerate(embedding_columns(dims))
        }
    )
    return out
