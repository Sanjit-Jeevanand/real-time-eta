"""Phase 6 steps 1-4: quantile models, the crossing rate, and the three fixes.

Every strategy here produces the same object -- an (n_rows, n_alphas) matrix of
quantile predictions on the test split -- so crossing and business cost are measured
the same way for all of them, and the served promise is always the column at q*.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.logging import get_logger
from eta.models.cost import business_cost
from eta.models.dataset import TARGET
from eta.models.evaluate import evaluate, load_seed_results, summarise_seeds
from eta.models.quantile.composition import MonotonicComposition
from eta.models.quantile.crossing import (
    CrossingReport,
    comparison_table,
    crossing_report,
    sort_rows,
)
from eta.models.quantile.lgbm import QuantileBundle, tune_on_validation
from eta.models.quantile.nn import MultiHeadQuantileNet
from eta.models.quantile.robustness import markdown as robustness_markdown
from eta.models.quantile.robustness import paired_blocks, temporal_windows
from eta.models.quantile.selection import select_champion

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl
    from numpy.typing import NDArray

    from eta.config import Settings
    from eta.models.evaluate import SeedResult

__all__ = ["StrategyRun", "run_quantile_phase"]

#: Crossing treatments eligible to be *served*. The neural nets are comparators --
#: they are untuned relative to the boosters, so letting them win a selection would
#: be comparing a tuned model against an unfunded one.
CANDIDATES: tuple[str, ...] = ("lgbm_quantile", "lgbm_sorted", "lgbm_composed")

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StrategyRun:
    """One (strategy, seed) pair: what it predicted and whether it crossed."""

    strategy: str
    seed: int
    crossing_rate: float
    crossing_rows: int
    worst_gap_s: float
    enforced: str
    business_cost: float
    late_rate: float


def _q_star_column(alphas: Sequence[float], q_star: float) -> int:
    for i, a in enumerate(alphas):
        if abs(a - q_star) < 1e-9:
            return i
    msg = f"q*={q_star} is not among the trained quantiles {alphas}"
    raise ValueError(msg)


def run_quantile_phase(
    settings: Settings,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    features: Sequence[str],
    population: str,
    reports: Path,
    trials: int,
    baseline_seeds: Path | None = None,
    skip_nn: bool = False,
) -> tuple[list[SeedResult], dict[str, Any]]:
    alphas = tuple(settings.model.reported_quantiles)
    q_star = settings.cost.optimal_quantile
    served = _q_star_column(alphas, q_star)
    actual = test[TARGET].to_numpy().astype(np.float64)
    val_actual = val[TARGET].to_numpy().astype(np.float64)
    segments = test.select(
        [
            c
            for c in ("seg_time", "seg_trip_length", "seg_zone_density", "seg_weather")
            if c in test.columns
        ]
    )

    log.info(
        "quantile_phase_started",
        alphas=list(alphas),
        q_star=q_star,
        served_column=served,
        train=train.height,
        val=val.height,
        test=test.height,
        population=population,
        trials=trials,
    )

    # --- step 1: tune each level on the validation split, never cal or test --------
    params_by_alpha: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        params_by_alpha[float(alpha)] = tune_on_validation(
            train, val, features, alpha, trials=trials
        )

    results: list[SeedResult] = []
    runs: list[StrategyRun] = []
    # Validation costs drive selection; test costs are recorded but never consulted
    # until the champion is frozen.
    val_costs: dict[str, list[float]] = {}
    test_promises: dict[str, NDArray[np.float64]] = {}
    clamp_rows: list[dict[str, Any]] = []
    clamped_mask: NDArray[np.bool_] | None = None
    # Every seed, not just the first: an unconstrained model's crossing rate moves a
    # lot between seeds, and that spread is part of what step 3 is asking for.
    crossing_reports: dict[str, list[CrossingReport]] = {}
    enforced_by: dict[str, str] = {}

    def record_val(strategy: str, matrix: NDArray[np.float64]) -> None:
        """Selection signal. This is the only cost that may inform the choice."""
        promised = matrix[:, served]
        cost = float(np.mean(business_cost(val_actual, promised, settings.cost)))
        val_costs.setdefault(strategy, []).append(cost)
        log.info(
            "val_cost",
            strategy=strategy,
            seed_index=len(val_costs[strategy]) - 1,
            val_cost=round(cost, 3),
        )

    def record(
        strategy: str,
        seed: int,
        matrix: NDArray[np.float64],
        enforced: str,
    ) -> None:
        report = crossing_report(matrix, alphas)
        crossing_reports.setdefault(strategy, []).append(report)
        enforced_by[strategy] = enforced
        promised = matrix[:, served]
        if seed == settings.model.seeds[0]:
            test_promises[strategy] = promised
        result = evaluate(strategy, seed, actual, promised, settings.cost, segments, population)
        results.append(result)
        runs.append(
            StrategyRun(
                strategy=strategy,
                seed=seed,
                crossing_rate=report.crossing_rate,
                crossing_rows=report.crossing_rows,
                worst_gap_s=report.worst_gap_s,
                enforced=enforced,
                business_cost=result.business_cost,
                late_rate=result.late_rate,
            )
        )
        log.info(
            "strategy_evaluated",
            strategy=strategy,
            seed=seed,
            crossing_rate=round(report.crossing_rate, 6),
            business_cost=round(result.business_cost, 2),
            late_rate=round(result.late_rate, 4),
            enforced=enforced,
        )

    for seed in settings.model.seeds:
        # --- steps 1 + 3: independent boosters, then measure what they do ---------
        bundle = QuantileBundle(alphas=alphas, features=features, params_by_alpha=params_by_alpha)
        bundle.fit(train, seed=seed, valid=val)

        raw_val = bundle.predict_matrix(val)
        record_val("lgbm_quantile", raw_val)
        record_val("lgbm_sorted", sort_rows(raw_val))

        raw = bundle.predict_matrix(test)
        record("lgbm_quantile", seed, raw, enforced="nothing -- independent fits")

        # --- step 4, fix 1: post-hoc sorting --------------------------------------
        record("lgbm_sorted", seed, sort_rows(raw), enforced="post-hoc sort")

        # --- step 4, fix 2: monotonic composition ---------------------------------
        composed = MonotonicComposition(
            alphas=alphas, features=features, params_by_alpha=params_by_alpha
        )
        composed.fit(train, seed=seed, valid=val)
        record_val("lgbm_composed", composed.predict_matrix(val))

        composed_test = composed.predict_matrix(test)
        if seed == settings.model.seeds[0]:
            clamped_mask = composed.clamped_any.copy()
        # How much the clamp actually rewrote -- a crossing fix and a different model
        # formulation are different claims, and the numbers decide which this is.
        for cs in composed.clamp_stats:
            clamp_rows.append({"seed": seed, **asdict(cs)})
        record("lgbm_composed", seed, composed_test, enforced="clamped increments")

        if skip_nn:
            continue

        # --- step 2: the neural comparator, free to cross -------------------------
        net = MultiHeadQuantileNet(alphas=alphas, features=features, ordered=False)
        net.fit(train, seed=seed, valid=val)
        record(
            "nn_multihead_untuned",
            seed,
            net.predict_matrix(test),
            enforced="nothing -- joint loss",
        )

        # --- step 4, fix 3: softplus-ordered heads --------------------------------
        ordered = MultiHeadQuantileNet(alphas=alphas, features=features, ordered=True)
        ordered.fit(train, seed=seed, valid=val)
        record(
            "nn_ordered_untuned",
            seed,
            ordered.predict_matrix(test),
            enforced="softplus increments",
        )

    # --- freeze the choice on validation, THEN look at test ----------------------
    selection = select_champion({k: v for k, v in val_costs.items() if k in CANDIDATES})
    log.info(
        "champion_selected",
        champion=selection.champion,
        val_cost=round(selection.val_cost, 2),
        margin=round(selection.margin, 3),
        margin_exceeds_seed_spread=selection.margin_exceeds_seed_spread,
        selected_on="validation split only",
    )

    # --- temporal robustness of the frozen champion against the L2 baseline -------
    from eta.models.baselines.models import LgbmL2

    if True:  # always measured: this is the headline's real error bar
        l2 = LgbmL2(features=features)
        l2.fit(train, seed=settings.model.seeds[0], valid=val)
        l2_promised = l2.predict(test)
        champ_promised = test_promises[selection.champion]

        windows = temporal_windows(
            test["request_datetime"], actual, champ_promised, l2_promised, settings.cost
        )
        blocks = paired_blocks(
            test["request_datetime"], actual, champ_promised, l2_promised, settings.cost
        )
        robustness_md = robustness_markdown(windows, blocks, selection.champion, "lgbm_l2")

        # Does the win survive on rows the clamp never touched? If the improvement
        # lives only where the constraint bound, the claim is about the constraint;
        # if it holds on untouched rows too, it is about the parameterisation.
        if clamped_mask is not None and selection.champion == "lgbm_composed":
            champ_c = business_cost(actual, champ_promised, settings.cost)
            base_c = business_cost(actual, l2_promised, settings.cost)
            untouched = ~clamped_mask
            split = {
                "clamped_rows": int(clamped_mask.sum()),
                "clamped_share": float(clamped_mask.mean()),
                "champion_cost_clamped": float(champ_c[clamped_mask].mean())
                if clamped_mask.any()
                else float("nan"),
                "baseline_cost_clamped": float(base_c[clamped_mask].mean())
                if clamped_mask.any()
                else float("nan"),
                "champion_cost_untouched": float(champ_c[untouched].mean())
                if untouched.any()
                else float("nan"),
                "baseline_cost_untouched": float(base_c[untouched].mean())
                if untouched.any()
                else float("nan"),
            }
            split["improvement_clamped_pct"] = (
                100.0
                * (split["champion_cost_clamped"] - split["baseline_cost_clamped"])
                / split["baseline_cost_clamped"]
            )
            split["improvement_untouched_pct"] = (
                100.0
                * (split["champion_cost_untouched"] - split["baseline_cost_untouched"])
                / split["baseline_cost_untouched"]
            )
            (reports / "clamp_split.json").write_text(json.dumps(split, indent=2) + "\n")
            log.info("clamp_split", **{k: round(v, 4) for k, v in split.items()})
        (reports / "robustness.md").write_text(robustness_md + "\n")
        (reports / "robustness.json").write_text(
            json.dumps(
                {"windows": [asdict(w) for w in windows], "blocks": asdict(blocks)}, indent=2
            )
            + "\n"
        )

    # The Phase 5 baselines join the board without refitting; the population digest
    # is what makes that safe, and it raises here if the matrix has moved underneath.
    if baseline_seeds is not None and baseline_seeds.exists():
        prior = load_seed_results(baseline_seeds)
        results.extend(prior)
        log.info("baselines_loaded", models=sorted({r.model for r in prior}), rows=len(prior))

    board = summarise_seeds(results)
    reports.mkdir(parents=True, exist_ok=True)
    board.to_frame().write_parquet(reports / "quantile.parquet")
    (reports / "quantile.md").write_text(board.markdown(baseline="lgbm_l2") + "\n")
    (reports / "quantile_seeds.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str) + "\n"
    )

    (reports / "selection.md").write_text(
        "## Crossing strategy selected on the validation split\n\n"
        + selection.markdown()
        + "\n\nThe test split was not consulted for this choice. Test numbers below are "
        "reported for the frozen champion and, for transparency, for the strategies that "
        "lost -- but the decision was already made.\n"
    )
    if clamp_rows:
        import statistics as _st
        from collections import defaultdict as _dd

        by_alpha: dict[float, list[dict[str, Any]]] = _dd(list)
        for r in clamp_rows:
            by_alpha[float(r["alpha"])].append(r)
        lines = [
            "## What the non-negativity clamp actually changed",
            "",
            "| level | rows clamped | share | mean adj | p95 adj | max adj |",
            "|---|---|---|---|---|---|",
        ]
        for a in sorted(by_alpha):
            g = by_alpha[a]
            lines.append(
                f"| P{round(a * 100)} | {_st.fmean(r['clamped_rows'] for r in g):,.0f} | "
                f"**{_st.fmean(r['clamped_share'] for r in g):.2%}** | "
                f"{_st.fmean(r['mean_adjustment_s'] for r in g):.1f}s | "
                f"{_st.fmean(r['p95_adjustment_s'] for r in g):.1f}s | "
                f"{max(r['max_adjustment_s'] for r in g):.1f}s |"
            )
        lines += [
            "",
            "Compare against the raw crossing rate. If the clamp rewrites far more rows than "
            "actually crossed, the composition is not a crossing repair -- it is a different "
            "model formulation that happens to be monotone, and must be compared as one.",
        ]
        (reports / "clamp.md").write_text("\n".join(lines) + "\n")
        (reports / "clamp.json").write_text(json.dumps(clamp_rows, indent=2) + "\n")

    crossing_md = "\n\n".join(
        [
            "## Crossing rate, before and after each fix",
            "",
            comparison_table(crossing_reports, enforced_by),
            "",
            "### Per-strategy detail (first seed)",
            *[f"\n#### {name}\n\n{reps[0].markdown()}" for name, reps in crossing_reports.items()],
        ]
    )
    (reports / "crossing.md").write_text(crossing_md + "\n")
    (reports / "crossing_runs.json").write_text(
        json.dumps([asdict(r) for r in runs], indent=2) + "\n"
    )

    summary = {
        "alphas": [float(a) for a in alphas],
        "q_star": float(q_star),
        "population": population,
        "tuning_trials": trials,
        "tuned_params": params_by_alpha,
        "crossing": {k: [asdict(r) for r in reps] for k, reps in crossing_reports.items()},
        "enforced": enforced_by,
        "selection": asdict(selection),
        # Per-seed, so the ordering can be checked seed by seed rather than only on
        # the mean -- a champion that wins on average but loses on a seed is a
        # near-tie however clean the mean looks.
        "val_costs": val_costs,
        "clamp": clamp_rows,
    }
    (reports / "quantile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results, summary
