"""Phase 7: measure raw calibration, then fix it three ways and compare.

Order matters here. The quantile models are fitted on **train** only; every
correction is fitted on **cal** only; every number reported is measured on **test**.
Nothing that touches test is ever fitted, which is what makes the coverage claim at
the end a claim rather than a description of the fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.calibration.coverage import (
    coverage_report,
    interval_coverage,
    interval_coverage_by_segment,
    worst_interval_gap,
)
from eta.calibration.cqr import MondrianCQR, SplitCQR
from eta.calibration.isotonic import IsotonicCalibrator
from eta.logging import get_logger
from eta.models.dataset import TARGET
from eta.models.quantile.lgbm import QuantileBundle, default_params

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl
    from numpy.typing import NDArray

    from eta.config import Settings

__all__ = ["CalibrationRow", "run_calibration_phase"]

log = get_logger(__name__)

SEGMENT_AXES = ("seg_time", "seg_trip_length", "seg_zone_density", "seg_weather")


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    approach: str
    seed: int
    aggregate_gap_pp: float
    worst_point_gap_pp: float
    worst_point_segment: str
    worst_interval_gap_pp: float
    worst_interval_segment: str
    mean_width_s: float
    interval_coverage: float


def _segments(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select([c for c in SEGMENT_AXES if c in frame.columns])


def _tuned_params(reports: Path, alphas: Sequence[float]) -> dict[float, dict[str, Any]]:
    """Reuse Phase 6's validation-tuned parameters where they exist."""
    path = reports / "quantile_summary.json"
    if not path.exists():
        return {}
    tuned = json.loads(path.read_text()).get("tuned_params", {})
    out = {float(k): v for k, v in tuned.items()}
    missing = [a for a in alphas if float(a) not in out]
    if missing:
        log.info("tuned_params_missing", levels=missing, using="defaults")
        for a in missing:
            out[float(a)] = default_params()
    return out


def run_calibration_phase(
    settings: Settings,
    train: pl.DataFrame,
    cal: pl.DataFrame,
    test: pl.DataFrame,
    features: Sequence[str],
    reports: Path,
) -> dict[str, Any]:
    alphas = tuple(settings.model.quantiles)
    target = settings.calibration.target_coverage
    alpha_risk = 1.0 - target
    lo_level, hi_level = (1.0 - target) / 2.0, 1.0 - (1.0 - target) / 2.0
    lo_idx = min(range(len(alphas)), key=lambda i: abs(alphas[i] - lo_level))
    hi_idx = min(range(len(alphas)), key=lambda i: abs(alphas[i] - hi_level))

    cal_actual = cal[TARGET].to_numpy().astype(np.float64)
    test_actual = test[TARGET].to_numpy().astype(np.float64)
    cal_seg, test_seg = _segments(cal), _segments(test)
    params = _tuned_params(reports, alphas)

    log.info(
        "calibration_phase_started",
        quantiles=list(alphas),
        target_coverage=target,
        interval=[alphas[lo_idx], alphas[hi_idx]],
        train=train.height,
        cal=cal.height,
        test=test.height,
    )

    rows: list[CalibrationRow] = []
    reliability: dict[str, Any] = {}

    for seed in settings.model.seeds:
        bundle = QuantileBundle(alphas=alphas, features=features, params_by_alpha=params)
        bundle.fit(train, seed=seed)
        cal_matrix = bundle.predict_matrix(cal)
        test_matrix = bundle.predict_matrix(test)

        def record(
            name: str,
            matrix: NDArray[np.float64],
            lo: NDArray[np.float64],
            hi: NDArray[np.float64],
            seed: int = seed,
        ) -> None:
            rep = coverage_report(test_actual, matrix, alphas, test_seg)
            worst = rep.worst(nominal=settings.cost.optimal_quantile)
            # The interval is scored separately from the point quantiles: CQR moves
            # the band and leaves the columns alone, so scoring it on the columns
            # would report marginal and Mondrian CQR as identical.
            band_points = interval_coverage_by_segment(test_actual, lo, hi, test_seg, target)
            worst_band = worst_interval_gap(band_points)
            row = CalibrationRow(
                approach=name,
                seed=seed,
                aggregate_gap_pp=rep.aggregate_gap_pp(settings.cost.optimal_quantile),
                worst_point_gap_pp=abs(worst.gap_pp),
                worst_point_segment=f"{worst.axis}={worst.bucket}",
                worst_interval_gap_pp=abs(worst_band.gap_pp),
                worst_interval_segment=f"{worst_band.axis}={worst_band.bucket}",
                mean_width_s=float(np.mean(hi - lo)),
                interval_coverage=interval_coverage(test_actual, lo, hi),
            )
            rows.append(row)
            if seed == settings.model.seeds[0]:
                reliability[name] = [asdict(p) for p in rep.points]
            log.info(
                "calibration_measured",
                approach=name,
                seed=seed,
                aggregate_gap_pp=round(row.aggregate_gap_pp, 2),
                worst_point_gap_pp=round(row.worst_point_gap_pp, 2),
                worst_interval_gap_pp=round(row.worst_interval_gap_pp, 2),
                worst_interval_segment=row.worst_interval_segment,
                interval_coverage=round(row.interval_coverage, 4),
                mean_width_min=round(row.mean_width_s / 60, 2),
            )

        # --- A: raw, uncalibrated ------------------------------------------------
        record("raw", test_matrix, test_matrix[:, lo_idx], test_matrix[:, hi_idx])

        # --- B: per-segment isotonic, fitted on cal ------------------------------
        # One calibrator per axis; the worst axis in the raw report is the one the
        # project claims to fix, so every axis is fitted and reported separately.
        for axis in cal_seg.columns:
            iso = IsotonicCalibrator(
                alphas=tuple(float(a) for a in alphas),
                axis=axis,
                min_rows=settings.calibration.min_segment_samples,
            ).fit(cal_matrix, cal_actual, cal_seg)
            adjusted = iso.apply(test_matrix, test_seg)
            record(f"isotonic[{axis}]", adjusted, adjusted[:, lo_idx], adjusted[:, hi_idx])

        # --- C: marginal CQR -----------------------------------------------------
        split = SplitCQR(alpha=alpha_risk).fit(
            cal_actual, cal_matrix[:, lo_idx], cal_matrix[:, hi_idx]
        )
        band = split.apply(test_matrix[:, lo_idx], test_matrix[:, hi_idx])
        record("cqr", test_matrix, band.lo, band.hi)

        # --- D: Mondrian (segment-conditional) CQR -------------------------------
        mondrian = MondrianCQR(
            alpha=alpha_risk,
            axes=tuple(cal_seg.columns),
            floor=settings.calibration.min_segment_samples,
        ).fit(cal_actual, cal_matrix[:, lo_idx], cal_matrix[:, hi_idx], cal_seg)
        mband = mondrian.apply(test_matrix[:, lo_idx], test_matrix[:, hi_idx], test_seg)
        record("cqr_mondrian", test_matrix, mband.lo, mband.hi)

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "calibration_runs.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n"
    )
    (reports / "reliability.json").write_text(json.dumps(reliability, indent=2) + "\n")

    summary = _summarise(rows, target)
    (reports / "calibration.md").write_text(summary + "\n")
    return {
        "rows": [asdict(r) for r in rows],
        "target_coverage": target,
        "interval_levels": [alphas[lo_idx], alphas[hi_idx]],
        "markdown": summary,
    }


def _summarise(rows: Sequence[CalibrationRow], target: float) -> str:
    import statistics as st
    from collections import defaultdict

    grouped: dict[str, list[CalibrationRow]] = defaultdict(list)
    for r in rows:
        grouped[r.approach].append(r)

    lines = [
        "## Calibration: the aggregate is not the number that matters",
        "",
        "Point-quantile columns score the served P75. Interval columns score the "
        f"{target:.0%} band -- CQR moves the band, not the columns, so the two are "
        "reported separately rather than conflated.",
        "",
        "| approach | agg gap (P75) | worst-segment gap (P75) | **worst-segment gap "
        f"(interval)** | worst segment | interval coverage (target {target:.0%}) | "
        "mean width |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, group in grouped.items():
        agg = st.fmean(r.aggregate_gap_pp for r in group)
        wp = st.fmean(r.worst_point_gap_pp for r in group)
        wi = st.fmean(r.worst_interval_gap_pp for r in group)
        wi_sd = st.stdev([r.worst_interval_gap_pp for r in group]) if len(group) > 1 else 0.0
        cov = st.fmean(r.interval_coverage for r in group)
        width = st.fmean(r.mean_width_s for r in group) / 60.0
        seg = max(
            {r.worst_interval_segment for r in group},
            key=[x.worst_interval_segment for x in group].count,
        )
        lines.append(
            f"| {name} | {agg:.1f}pp | {wp:.1f}pp | **{wi:.1f} ± {wi_sd:.1f}pp** | {seg} | "
            f"{cov:.2%} | {width:.1f} min |"
        )
    return "\n".join(lines)
