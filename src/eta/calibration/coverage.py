"""Empirical coverage, globally and along each segment axis.

Phase 7A. The aggregate number is expected to look acceptable while individual
segments are badly broken -- so the aggregate is reported *next to* the worst
segment, never on its own. A calibration claim backed only by a marginal average
is the thing this module exists to make impossible to state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from numpy.typing import NDArray

__all__ = [
    "CoveragePoint",
    "CoverageReport",
    "coverage",
    "coverage_report",
    "interval_coverage",
    "interval_coverage_by_segment",
    "worst_interval_gap",
]


def coverage(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """P(actual <= predicted) -- the empirical coverage of a one-sided quantile."""
    return float(np.mean(actual <= predicted))


def interval_coverage(
    actual: NDArray[np.float64], lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> float:
    return float(np.mean((actual >= lo) & (actual <= hi)))


@dataclass(frozen=True, slots=True)
class CoveragePoint:
    axis: str
    bucket: str
    rows: int
    nominal: float
    empirical: float

    @property
    def gap_pp(self) -> float:
        """Signed, in percentage points. Negative means under-covering."""
        return 100.0 * (self.empirical - self.nominal)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    points: list[CoveragePoint]

    def global_points(self) -> list[CoveragePoint]:
        return [p for p in self.points if p.axis == "global"]

    def segment_points(self) -> list[CoveragePoint]:
        return [p for p in self.points if p.axis != "global"]

    def worst(self, nominal: float | None = None) -> CoveragePoint:
        """The segment furthest from its nominal level, by absolute gap."""
        pool = self.segment_points()
        if nominal is not None:
            pool = [p for p in pool if abs(p.nominal - nominal) < 1e-9]
        return max(pool, key=lambda p: abs(p.gap_pp))

    def worst_gap_pp(self, nominal: float | None = None) -> float:
        return abs(self.worst(nominal).gap_pp)

    def aggregate_gap_pp(self, nominal: float) -> float:
        for p in self.global_points():
            if abs(p.nominal - nominal) < 1e-9:
                return abs(p.gap_pp)
        msg = f"no global coverage recorded at {nominal}"
        raise ValueError(msg)

    def markdown(self, nominal: float) -> str:
        pts = sorted(
            (p for p in self.segment_points() if abs(p.nominal - nominal) < 1e-9),
            key=lambda p: p.gap_pp,
        )
        lines = [
            "| segment | rows | nominal | empirical | gap |",
            "|---|---|---|---|---|",
        ]
        for p in pts:
            lines.append(
                f"| {p.axis}={p.bucket} | {p.rows:,} | {p.nominal:.0%} | "
                f"{p.empirical:.1%} | **{p.gap_pp:+.1f}pp** |"
            )
        return "\n".join(lines)


def interval_coverage_by_segment(
    actual: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    segments: pl.DataFrame,
    target: float,
    min_rows: int = 1_000,
) -> list[CoveragePoint]:
    """Conditional coverage of an *interval*, per segment bucket.

    Needed because CQR moves the interval, not the point quantiles: scoring a
    conformalised approach on its untouched quantile columns would show marginal
    and Mondrian CQR as identical, which is precisely the comparison 7D exists to
    make.
    """
    points: list[CoveragePoint] = []
    for axis in segments.columns:
        values = segments[axis].to_numpy()
        for bucket in np.unique(values):
            if bucket is None:
                continue
            mask = values == bucket
            n = int(mask.sum())
            if n < min_rows:
                continue
            points.append(
                CoveragePoint(
                    axis=axis,
                    bucket=str(bucket),
                    rows=n,
                    nominal=target,
                    empirical=interval_coverage(actual[mask], lo[mask], hi[mask]),
                )
            )
    return points


def worst_interval_gap(points: Sequence[CoveragePoint]) -> CoveragePoint:
    return max(points, key=lambda p: abs(p.gap_pp))


def coverage_report(
    actual: NDArray[np.float64],
    matrix: NDArray[np.float64],
    alphas: Sequence[float],
    segments: pl.DataFrame,
    min_rows: int = 1_000,
) -> CoverageReport:
    """Coverage at every nominal level, globally and per segment bucket."""
    points: list[CoveragePoint] = []
    for j, q in enumerate(alphas):
        points.append(
            CoveragePoint(
                "global", "all", int(actual.size), float(q), coverage(actual, matrix[:, j])
            )
        )
        for axis in segments.columns:
            values = segments[axis].to_numpy()
            for bucket in np.unique(values):
                if bucket is None:
                    continue
                mask = values == bucket
                n = int(mask.sum())
                if n < min_rows:
                    continue
                points.append(
                    CoveragePoint(
                        axis=axis,
                        bucket=str(bucket),
                        rows=n,
                        nominal=float(q),
                        empirical=coverage(actual[mask], matrix[mask, j]),
                    )
                )
    return CoverageReport(points=points)
