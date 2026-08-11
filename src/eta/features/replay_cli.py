from __future__ import annotations

import argparse
import sys

from eta.config import get_settings
from eta.features.replay import DictStore, replay_to_store
from eta.logging import bind_request_id, configure_logging, get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eta.features.replay")
    parser.add_argument("--month", default="*")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level)
    bind_request_id()

    processed = settings.paths.resolve()["processed_dir"]
    glob = processed / "enriched" / f"enriched_{args.month}.parquet"

    store = DictStore()
    stream = replay_to_store(glob, store)
    log.info(
        "replay_store_summary",
        keys=len(store.data),
        writes=store.writes,
        buckets=stream.buckets_seen,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
