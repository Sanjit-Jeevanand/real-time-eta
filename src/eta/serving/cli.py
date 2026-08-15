from __future__ import annotations

import argparse
import sys

from eta.config import get_settings
from eta.logging import bind_request_id, configure_logging, get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eta.serving")
    parser.add_argument("stage", nargs="?", default="bench", choices=("bench", "compile"))
    parser.add_argument("--requests", type=int, default=2_000)
    parser.add_argument("--redis", action="store_true", help="use a real Redis over TCP")
    parser.add_argument("--treelite", action="store_true", help="use the compiled models")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level)
    bind_request_id()

    reports = settings.paths.resolve()["reports_dir"]

    if args.stage == "compile":
        from eta.serving.bench import run_compile_comparison

        out = run_compile_comparison(settings, reports)
        print(out["markdown"])  # noqa: T201
        return 0

    from eta.serving.bench import run_bench

    out = run_bench(
        settings,
        reports,
        requests=args.requests,
        use_redis=args.redis,
        use_treelite=args.treelite,
    )
    print(out["markdown"])  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
