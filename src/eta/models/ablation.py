"""Phase 6 steps 5-7: decomposition ablation, SHAP by segment, and the headline.

The decomposition question has a statistical trap in it that the plan does not
mention, and measuring it is most of the value here.

`total_time = dispatch_approach + curb_wait + trip_duration` holds *exactly* for
every row. It does not follow that the q* quantiles add:

    Q_q(A) + Q_q(B) + Q_q(C)  >=  Q_q(A + B + C)

with equality only when the components are perfectly comonotonic -- when the same
trips are simultaneously worst on all three. Real components are far from that, so
summing per-component P75s systematically **overshoots** the P75 of the total. The
decomposed model is not merely a different estimator of the same quantity; under an
asymmetric cost it is a more conservative one, and its late rate will land below
target for a structural reason rather than a modelling one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from eta.data.target import COMPONENT_COLUMNS
from eta.logging import get_logger
from eta.models.cost import business_cost, late_rate
from eta.models.dataset import TARGET
from eta.models.quantile.lgbm import QuantileBundle, default_params

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from eta.config import Settings

__all__ = ["AblationRow", "run_ablation_phase"]

log = get_logger(__name__)

SEGMENT_AXES = ("seg_time", "seg_trip_length", "seg_zone_density", "seg_weather")


@dataclass(frozen=True, slots=True)
class AblationRow:
    approach: str
    seed: int
    rows: int
    business_cost: float
    late_rate: float
    mae: float


def component_complete(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows where every component is observable -- Uber only, by construction."""
    expr = pl.col(COMPONENT_COLUMNS[0]).is_not_null()
    for c in COMPONENT_COLUMNS[1:]:
        expr = expr & pl.col(c).is_not_null()
    # Components must actually reconcile with the target, or the sum is not the total.
    total = pl.sum_horizontal(*[pl.col(c) for c in COMPONENT_COLUMNS])
    return frame.filter(expr & ((total - pl.col(TARGET)).abs() <= 1.0))


def _tuned(reports: Path, alpha: float) -> dict[str, Any]:
    path = reports / "quantile_summary.json"
    if not path.exists():
        return default_params()
    tuned = json.loads(path.read_text()).get("tuned_params", {})
    by_alpha: dict[float, dict[str, Any]] = {float(k): v for k, v in tuned.items()}
    return by_alpha.get(float(alpha), default_params())


def _fit_q(
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: Sequence[str],
    target: str,
    alpha: float,
    seed: int,
    params: dict[str, Any],
) -> NDArray[np.float64]:
    """One quantile booster on an arbitrary target column."""
    renamed_train = train.with_columns(pl.col(target).alias(TARGET)) if target != TARGET else train
    bundle = QuantileBundle(
        alphas=(alpha,), features=features, params_by_alpha={float(alpha): params}
    )
    bundle.fit(renamed_train, seed=seed)
    return bundle.predict_matrix(test)[:, 0]


def run_ablation_phase(
    settings: Settings,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: Sequence[str],
    reports: Path,
) -> dict[str, Any]:
    q_star = float(settings.cost.optimal_quantile)
    params = _tuned(reports, q_star)

    train_c = component_complete(train)
    test_c = component_complete(test)
    log.info(
        "ablation_population",
        train_all=train.height,
        train_component_complete=train_c.height,
        test_all=test.height,
        test_component_complete=test_c.height,
        share=round(test_c.height / test.height, 4),
    )

    actual = test_c[TARGET].to_numpy().astype(np.float64)
    rows: list[AblationRow] = []
    comonotonicity: dict[str, float] = {}
    # The plan predicts decomposition wins at peak. Sliced, so that is testable
    # rather than a claim inherited from the plan.
    by_segment: dict[str, dict[str, list[float]]] = {}

    for seed in settings.model.seeds:
        direct = _fit_q(train_c, test_c, features, TARGET, q_star, seed, params)
        parts = {
            c: _fit_q(train_c, test_c, features, c, q_star, seed, params) for c in COMPONENT_COLUMNS
        }
        summed = np.sum(np.column_stack(list(parts.values())), axis=1)

        for name, promised in (("direct", direct), ("decomposed", summed)):
            cost = float(np.mean(business_cost(actual, promised, settings.cost)))
            rows.append(
                AblationRow(
                    approach=name,
                    seed=seed,
                    rows=int(actual.size),
                    business_cost=cost,
                    late_rate=late_rate(actual, promised),
                    mae=float(np.mean(np.abs(promised - actual))),
                )
            )
            log.info(
                "ablation_evaluated",
                approach=name,
                seed=seed,
                business_cost=round(cost, 2),
                late_rate=round(late_rate(actual, promised), 4),
            )
            per_row = business_cost(actual, promised, settings.cost)
            for axis in SEGMENT_AXES:
                if axis not in test_c.columns:
                    continue
                values = test_c[axis].to_numpy()
                for bucket in np.unique(values):
                    if bucket is None:
                        continue
                    mask = values == bucket
                    if int(mask.sum()) < 1_000:
                        continue
                    key = f"{axis}={bucket}"
                    by_segment.setdefault(key, {}).setdefault(name, []).append(
                        float(np.mean(per_row[mask]))
                    )

        if seed == settings.model.seeds[0]:
            # How much the sum overshoots, in seconds and as a share of the promise.
            gap = summed - direct
            comonotonicity = {
                "mean_overshoot_s": float(np.mean(gap)),
                "median_overshoot_s": float(np.median(gap)),
                "share_overshooting": float(np.mean(gap > 0)),
                "mean_direct_s": float(np.mean(direct)),
                "mean_summed_s": float(np.mean(summed)),
            }
            log.info(
                "decomposition_overshoot", **{k: round(v, 3) for k, v in comonotonicity.items()}
            )

    summary = _summarise(rows, test_c.height, test.height, comonotonicity, by_segment)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ablation.md").write_text(summary + "\n")
    (reports / "ablation_runs.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n"
    )
    return {"rows": [asdict(r) for r in rows], "markdown": summary, "overshoot": comonotonicity}


def _summarise(
    rows: Sequence[AblationRow],
    n_used: int,
    n_all: int,
    overshoot: dict[str, float],
    by_segment: dict[str, dict[str, list[float]]] | None = None,
) -> str:
    import statistics as st
    from collections import defaultdict

    grouped: dict[str, list[AblationRow]] = defaultdict(list)
    for r in rows:
        grouped[r.approach].append(r)

    lines = [
        "## Component ablation: decomposed vs direct",
        "",
        f"**Uber rows only** -- {n_used:,} of {n_all:,} test rows ({n_used / n_all:.1%}). "
        "Lyft does not report `on_scene`, so two of the three components do not exist "
        "there. Both approaches are scored on this same restricted population.",
        "",
        "| approach | business cost | late rate | MAE (min) | seeds |",
        "|---|---|---|---|---|",
    ]
    for name, group in grouped.items():
        cost = st.fmean(r.business_cost for r in group)
        sd = st.stdev([r.business_cost for r in group]) if len(group) > 1 else 0.0
        lines.append(
            f"| {name} | {cost:.1f} ± {sd:.1f} | {st.fmean(r.late_rate for r in group):.1%} | "
            f"{st.fmean(r.mae for r in group) / 60:.2f} | {len(group)} |"
        )

    if overshoot:
        lines += [
            "",
            "### Why the decomposed promise runs long",
            "",
            "Summing per-component P75s does not give the P75 of the total. "
            "Q(A)+Q(B)+Q(C) >= Q(A+B+C), with equality only under perfect comonotonicity "
            "-- the same trips being worst on all three components at once. They are not.",
            "",
            f"- mean direct promise: **{overshoot['mean_direct_s'] / 60:.2f} min**",
            f"- mean summed promise: **{overshoot['mean_summed_s'] / 60:.2f} min**",
            f"- mean overshoot: **{overshoot['mean_overshoot_s'] / 60:.2f} min** "
            f"({overshoot['share_overshooting']:.1%} of rows overshoot)",
        ]

    if by_segment:
        lines += [
            "",
            "### Does decomposition win at peak?",
            "",
            "The plan expects it to. Cost per segment, direct vs decomposed:",
            "",
            "| segment | direct | decomposed | winner |",
            "|---|---|---|---|",
        ]
        for key, per_approach in sorted(by_segment.items()):
            d = st.fmean(per_approach.get("direct", [float("nan")]))
            c = st.fmean(per_approach.get("decomposed", [float("nan")]))
            winner = "**direct**" if d <= c else "**decomposed**"
            lines.append(f"| {key} | {d:.1f} | {c:.1f} | {winner} ({abs(d - c):.1f}) |")
    return "\n".join(lines)
