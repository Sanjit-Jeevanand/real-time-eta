from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
import polars as pl

from eta.data.segments import SEGMENT_COLUMNS
from eta.logging import get_logger
from eta.models.cost import business_cost, late_rate, pinball_loss

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from eta.config import CostConfig

__all__ = [
    "Leaderboard",
    "SeedResult",
    "SegmentResult",
    "evaluate",
    "load_seed_results",
    "summarise_seeds",
]

log = get_logger(__name__)

MIN_SEGMENT_ROWS: Final = 1_000


@dataclass(frozen=True, slots=True)
class SegmentResult:
    axis: str
    bucket: str
    rows: int
    business_cost: float
    mae: float
    late_rate: float


@dataclass(frozen=True, slots=True)
class SeedResult:
    model: str
    seed: int
    rows: int
    business_cost: float
    mae: float
    rmse: float
    late_rate: float
    pinball_at_q_star: float
    population: str = ""
    segments: list[SegmentResult] = field(default_factory=list)


def evaluate(
    model: str,
    seed: int,
    actual: NDArray[np.float64],
    promised: NDArray[np.float64],
    cfg: CostConfig,
    segments: pl.DataFrame | None = None,
    population: str = "",
) -> SeedResult:
    cost = business_cost(actual, promised, cfg)
    error = promised - actual
    q_star = cfg.optimal_quantile

    seg_results: list[SegmentResult] = []
    if segments is not None:
        frame = segments.with_columns(
            pl.Series("__cost", cost),
            pl.Series("__abs_err", np.abs(error)),
            pl.Series("__late", actual > promised),
        )
        for axis in SEGMENT_COLUMNS:
            if axis not in frame.columns:
                continue
            grouped = (
                frame.group_by(axis)
                .agg(
                    pl.len().alias("rows"),
                    pl.col("__cost").mean().alias("cost"),
                    pl.col("__abs_err").mean().alias("mae"),
                    pl.col("__late").mean().alias("late"),
                )
                .sort(axis)
            )
            for row in grouped.iter_rows(named=True):
                if row["rows"] < MIN_SEGMENT_ROWS or row[axis] is None:
                    continue
                seg_results.append(
                    SegmentResult(
                        axis=axis,
                        bucket=str(row[axis]),
                        rows=int(row["rows"]),
                        business_cost=float(row["cost"]),
                        mae=float(row["mae"]),
                        late_rate=float(row["late"]),
                    )
                )

    return SeedResult(
        model=model,
        seed=seed,
        rows=int(actual.size),
        business_cost=float(np.mean(cost)),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        late_rate=late_rate(actual, promised),
        pinball_at_q_star=float(np.mean(pinball_loss(actual, promised, q_star))),
        population=population,
        segments=seg_results,
    )


def load_seed_results(path: Path) -> list[SeedResult]:
    """Rehydrate results written by an earlier run.

    Used to put the Phase 5 baselines on the same leaderboard as the quantile models
    without refitting them. This is safe precisely because `population` travels with
    each result: if the earlier run scored a different population, `summarise_seeds`
    raises rather than printing a comparison that looks fine.
    """
    import json

    raw = json.loads(path.read_text())
    return [
        SeedResult(
            **{k: v for k, v in row.items() if k != "segments"},
            segments=[SegmentResult(**s) for s in row.get("segments", [])],
        )
        for row in raw
    ]


@dataclass(frozen=True, slots=True)
class Leaderboard:
    rows: list[dict[str, object]]

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows).sort("business_cost_mean")

    def markdown(self, baseline: str | None = None) -> str:
        frame = self.to_frame()
        ref = None
        if baseline is not None:
            match = frame.filter(pl.col("model") == baseline)
            if match.height:
                ref = float(match["business_cost_mean"][0])

        header = "| model | business cost | MAE (min) | RMSE (min) | late rate | seeds |"
        if ref is not None:
            header = (
                "| model | business cost | vs baseline | MAE (min) | RMSE (min) | late rate |"
                " seeds |"
            )
        lines = [header, "|" + "---|" * (header.count("|") - 1)]
        for r in frame.iter_rows(named=True):
            cost = f"{r['business_cost_mean']:.1f} ± {r['business_cost_std']:.1f}"
            mae = f"{r['mae_mean'] / 60:.2f} ± {r['mae_std'] / 60:.2f}"
            rmse = f"{r['rmse_mean'] / 60:.2f}"
            late = f"{r['late_rate_mean']:.1%}"
            if ref is not None:
                delta = 100.0 * (r["business_cost_mean"] - ref) / ref
                lines.append(
                    f"| {r['model']} | {cost} | {delta:+.1f}% | {mae} | {rmse} | {late} |"
                    f" {r['seeds']} |"
                )
            else:
                lines.append(f"| {r['model']} | {cost} | {mae} | {rmse} | {late} | {r['seeds']} |")
        return "\n".join(lines)


def summarise_seeds(results: Sequence[SeedResult]) -> Leaderboard:
    by_model: dict[str, list[SeedResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    populations = {r.population for r in results if r.population}
    if len(populations) > 1:
        msg = (
            "models were evaluated on different populations, so their costs are not "
            f"comparable: {sorted(populations)}"
        )
        raise ValueError(msg)

    rows: list[dict[str, object]] = []
    for model, runs in by_model.items():
        if len(runs) < 3:
            log.warning("fewer_than_three_seeds", model=model, seeds=len(runs))
        row: dict[str, object] = {"model": model, "seeds": len(runs), "rows": runs[0].rows}
        for metric in ("business_cost", "mae", "rmse", "late_rate", "pinball_at_q_star"):
            values = [getattr(r, metric) for r in runs]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return Leaderboard(rows=rows)


def segment_table(results: Sequence[SeedResult], model: str) -> pl.DataFrame:
    runs = [r for r in results if r.model == model]
    flat = [asdict(s) | {"seed": r.seed} for r in runs for s in r.segments]
    if not flat:
        return pl.DataFrame()
    return (
        pl.DataFrame(flat)
        .group_by("axis", "bucket")
        .agg(
            pl.col("rows").first(),
            pl.col("business_cost").mean().alias("business_cost"),
            pl.col("mae").mean().alias("mae"),
            pl.col("late_rate").mean().alias("late_rate"),
        )
        .sort("axis", "bucket")
    )
