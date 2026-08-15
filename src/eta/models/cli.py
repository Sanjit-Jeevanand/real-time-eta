from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from eta.config import get_settings
from eta.data.splits import Split
from eta.logging import bind_request_id, configure_logging, get_logger
from eta.models.baselines.models import HistoricalMean, LgbmL2, OsrmMultiplier
from eta.models.dataset import (
    TARGET,
    build_matrix,
    feature_names,
    load_split,
    population_digest,
)
from eta.models.evaluate import evaluate, segment_table, summarise_seeds
from eta.models.sampling import Tier

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eta.models")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("matrix", "baselines", "quantile", "calibrate", "ablation", "explain", "all"),
    )
    parser.add_argument("--tier", default=Tier.TUNE.value, choices=[t.value for t in Tier])
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Optuna trials per quantile level; defaults to model.optuna_trials",
    )
    parser.add_argument(
        "--skip-nn",
        action="store_true",
        help="skip the torch comparator and the softplus-ordered net",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level)
    bind_request_id()

    tier = Tier(args.tier)
    paths = settings.paths.resolve()
    matrix_path = paths["processed_dir"] / f"model_matrix_{tier.value}.parquet"

    if args.stage in ("matrix", "all") or not matrix_path.exists():
        matrix_path = build_matrix(settings, tier)

    if args.stage not in ("baselines", "quantile", "calibrate", "ablation", "explain", "all"):
        return 0

    train = load_split(matrix_path, Split.TRAIN)
    val = load_split(matrix_path, Split.VAL)
    test = load_split(matrix_path, Split.TEST)
    cal = load_split(matrix_path, Split.CAL)
    population = population_digest(test)
    log.info(
        "splits_loaded",
        train=train.height,
        val=val.height,
        cal=cal.height,
        test=test.height,
        tier=tier.value,
        population=population,
    )

    features = feature_names()
    actual = test[TARGET].to_numpy().astype(float)
    segments = test.select(
        [
            c
            for c in ("seg_time", "seg_trip_length", "seg_zone_density", "seg_weather")
            if c in test.columns
        ]
    )

    reports = paths["reports_dir"]

    if args.stage == "quantile":
        from eta.models.quantile.run import run_quantile_phase

        _, summary = run_quantile_phase(
            settings,
            train,
            val,
            test,
            features,
            population,
            reports,
            trials=args.trials if args.trials is not None else settings.model.optuna_trials,
            baseline_seeds=reports / "baseline_seeds.json",
            skip_nn=args.skip_nn,
        )
        print((reports / "quantile.md").read_text())  # noqa: T201
        print((reports / "crossing.md").read_text())  # noqa: T201
        log.info("quantile_phase_complete", strategies=list(summary["crossing"]))
        return 0

    if args.stage == "calibrate":
        from eta.calibration.run import run_calibration_phase

        out = run_calibration_phase(settings, train, cal, test, features, reports)
        print(out["markdown"])  # noqa: T201
        log.info("calibration_phase_complete", target=out["target_coverage"])
        return 0

    if args.stage == "ablation":
        from eta.models.ablation import run_ablation_phase

        out = run_ablation_phase(settings, train, test, features, reports)
        print(out["markdown"])  # noqa: T201
        return 0

    if args.stage == "explain":
        from eta.models.explain import run_explain_phase

        out = run_explain_phase(settings, train, test, features, reports)
        print(out["markdown"])  # noqa: T201
        return 0

    results = []
    for seed in settings.model.seeds:
        for model in (
            HistoricalMean(),
            OsrmMultiplier(),
            LgbmL2(features=features),
        ):
            if isinstance(model, LgbmL2):
                model.fit(train, seed=seed, valid=val)
            else:
                model.fit(train, seed=seed)
            promised = model.predict(test)
            result = evaluate(
                model.name, seed, actual, promised, settings.cost, segments, population
            )
            results.append(result)
            log.info(
                "baseline_evaluated",
                model=model.name,
                seed=seed,
                business_cost=round(result.business_cost, 2),
                mae_min=round(result.mae / 60, 2),
                late_rate=round(result.late_rate, 4),
            )

    board = summarise_seeds(results)
    reports.mkdir(parents=True, exist_ok=True)
    board.to_frame().write_parquet(reports / "baselines.parquet")
    (reports / "baselines.md").write_text(board.markdown(baseline="lgbm_l2") + "\n")
    (reports / "baseline_seeds.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str) + "\n"
    )
    seg = segment_table(results, "lgbm_l2")
    if seg.height:
        seg.write_parquet(reports / "baseline_segments.parquet")

    print(board.markdown(baseline="lgbm_l2"))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
