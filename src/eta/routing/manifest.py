from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from eta.logging import get_logger

if TYPE_CHECKING:
    from eta.config import Settings

__all__ = ["MANIFEST_NAME", "build_manifest", "write_manifest"]

log = get_logger(__name__)

MANIFEST_NAME: Final = "routing_manifest.json"
OSRM_IMAGE: Final = "ghcr.io/project-osrm/osrm-backend:v6.0.0"
OSRM_PROFILE: Final = "/opt/car.lua"
OSRM_ALGORITHM: Final = "mld"
OSM_SOURCE_URL: Final = "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"


def _sha256(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
            read += len(chunk)
            if read >= limit_bytes:
                digest.update(b"__truncated__")
                break
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def _osm_snapshot(osrm_dir: Path) -> dict[str, Any]:
    pbf = osrm_dir / "new-york-latest.osm.pbf"
    downloaded = (
        dt.datetime.fromtimestamp(pbf.stat().st_mtime, tz=dt.UTC).isoformat(timespec="seconds")
        if pbf.exists()
        else None
    )
    return {
        "source_url": OSM_SOURCE_URL,
        "file": pbf.name,
        "bytes": pbf.stat().st_size if pbf.exists() else None,
        "sha256_first_64mb": _sha256(pbf),
        "downloaded_utc": downloaded,
        "note": (
            "Geofabrik publishes a rolling extract; sha256 and downloaded_utc pin the exact "
            "snapshot. The .osrm.timestamp file is an OSRM binary fingerprint, not an OSM date."
        ),
    }


def build_manifest(settings: Settings, matrix_path: Path, zones_path: Path) -> dict[str, Any]:
    paths = settings.paths.resolve()
    shapefile = paths["raw_dir"] / "taxi_zones" / "taxi_zones.shp"
    return {
        "generated_utc": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "router": {
            "image": OSRM_IMAGE,
            "profile": OSRM_PROFILE,
            "algorithm": OSRM_ALGORITHM,
            "url": settings.routing.osrm_url,
        },
        "osm": _osm_snapshot(paths["raw_dir"].parent / "osrm"),
        "zone_geometry": {
            "source": "NYC TLC taxi_zones.zip",
            "crs_source": "EPSG:2263",
            "crs_target": "EPSG:4326",
            "shapefile_sha256": _sha256(shapefile),
            "zones": settings.routing.n_zones,
        },
        "matrix": {
            "file": matrix_path.name,
            "bytes": matrix_path.stat().st_size if matrix_path.exists() else None,
            "sha256": _sha256(matrix_path),
        },
        "zones_table": {
            "file": zones_path.name,
            "sha256": _sha256(zones_path),
        },
        "provenance": {
            "intra_zone_distance": (
                "observed median of same-zone trips, TRAIN split only; zones below the "
                "observation floor use an area-based fallback that is an extrapolation, "
                "not cross-validated on those zones"
            ),
            "structural_detour_ratio": "observed median distance / route distance, TRAIN split only",
            "zone_embeddings": "SVD on zone x zone trip counts, TRAIN split only",
        },
    }


def write_manifest(settings: Settings, matrix_path: Path, zones_path: Path) -> Path:
    manifest = build_manifest(settings, matrix_path, zones_path)
    dest = matrix_path.parent / MANIFEST_NAME
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("routing_manifest_written", path=str(dest), commit=manifest["git_commit"])
    return dest
