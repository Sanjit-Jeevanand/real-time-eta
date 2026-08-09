from __future__ import annotations

import argparse
import sys

import polars as pl

from eta.config import get_settings
from eta.logging import bind_request_id, configure_logging, get_logger
from eta.routing.detour import build_detour_ratios
from eta.routing.embeddings import build_zone_embeddings
from eta.routing.manifest import write_manifest
from eta.routing.matrix import build_matrix, validate_matrix
from eta.routing.zones import build_zone_table, validate_zone_table

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eta.routing")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("zones", "matrix", "embeddings", "detour", "all"),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level)
    bind_request_id()

    paths = settings.paths.resolve()
    processed = paths["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    enriched = processed / "enriched" / "enriched_*.parquet"
    zones_path = processed / "zones.parquet"
    matrix_path = processed / "zone_pair_matrix.parquet"

    if args.stage in ("zones", "all"):
        zones = build_zone_table(paths["raw_dir"], settings.segments)
        validate_zone_table(zones)
        zones.write_parquet(zones_path)
        log.info("zones_written", path=str(zones_path), zones=zones.height)

    if args.stage in ("matrix", "all"):
        zones = pl.read_parquet(zones_path)
        out = build_matrix(settings, zones, enriched)
        validate_matrix(pl.read_parquet(out), zones.height)
        write_manifest(settings, out, zones_path)

    if args.stage in ("embeddings", "all"):
        zones = pl.read_parquet(zones_path)
        emb = build_zone_embeddings(enriched, zones, settings.routing.zone_embedding_dims)
        dest = processed / "zone_embeddings.parquet"
        emb.write_parquet(dest)
        log.info("embeddings_written", path=str(dest), rows=emb.height)

    if args.stage in ("detour", "all"):
        matrix = pl.read_parquet(matrix_path)
        detour = build_detour_ratios(enriched, matrix)
        dest = processed / "detour_ratios.parquet"
        detour.write_parquet(dest)
        log.info("detour_written", path=str(dest), rows=detour.height)

    return 0


if __name__ == "__main__":
    sys.exit(main())
