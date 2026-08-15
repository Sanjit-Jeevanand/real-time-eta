"""Conformalized Quantile Regression -- Phase 7C and 7D.

The guarantee is finite-sample, distribution-free, and assumption-light: given
exchangeability between the calibration split and the test split, the interval

    [ q_lo(x) - Q ,  q_hi(x) + Q ]

covers the truth with probability at least 1 - alpha, where Q is the
ceil((n+1)(1-alpha))/n empirical quantile of the conformity scores

    E_i = max( q_lo(x_i) - y_i ,  y_i - q_hi(x_i) )

on the calibration split. Nothing is assumed about the shape of the residual
distribution or about the quantile models being any good -- a bad model still gets
coverage, it just pays for it in width. That is why width is reported next to
coverage everywhere in this module: coverage alone is trivially achievable.

The `+1` in the numerator is the finite-sample correction, and it is what makes the
guarantee hold at the stated level rather than approximately. It matters most for
Mondrian cells, where n is small by construction.

Mondrian CQR (7D) computes a separate Q per segment cell, which buys *conditional*
coverage instead of merely marginal. Thin cells borrow their score set from the most
specific sufficient parent via the Phase 3 cell plan, rather than collapsing to
global.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from eta.logging import get_logger

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

__all__ = [
    "ConformalInterval",
    "MondrianCQR",
    "SplitCQR",
    "conformal_quantile",
    "conformity_scores",
]

log = get_logger(__name__)


def conformity_scores(
    actual: NDArray[np.float64], lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> NDArray[np.float64]:
    """How far outside the interval each calibration point fell (negative = inside)."""
    return np.maximum(lo - actual, actual - hi)


def conformal_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile, with the finite-sample correction."""
    n = scores.size
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        # Too few points to certify this level at all; widen to the observed maximum
        # rather than quietly returning a quantile that does not carry the guarantee.
        return float(np.max(scores))
    return float(np.sort(scores)[rank - 1])


@dataclass(frozen=True, slots=True)
class ConformalInterval:
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]

    @property
    def width(self) -> NDArray[np.float64]:
        return self.hi - self.lo

    def covers(self, actual: NDArray[np.float64]) -> NDArray[np.bool_]:
        out: NDArray[np.bool_] = (actual >= self.lo) & (actual <= self.hi)
        return out


@dataclass(slots=True)
class SplitCQR:
    """Marginal CQR: one Q for the whole population."""

    alpha: float
    _q: float = 0.0
    _n: int = 0

    def fit(
        self, actual: NDArray[np.float64], lo: NDArray[np.float64], hi: NDArray[np.float64]
    ) -> SplitCQR:
        scores = conformity_scores(actual, lo, hi)
        self._q = conformal_quantile(scores, self.alpha)
        self._n = int(scores.size)
        log.info(
            "cqr_fitted",
            alpha=self.alpha,
            calibration_rows=self._n,
            q_seconds=round(self._q, 1),
            already_inside=round(float(np.mean(scores <= 0)), 4),
        )
        return self

    def apply(self, lo: NDArray[np.float64], hi: NDArray[np.float64]) -> ConformalInterval:
        return ConformalInterval(lo=lo - self._q, hi=hi + self._q)

    @property
    def q(self) -> float:
        return self._q


@dataclass(slots=True)
class MondrianCQR:
    """Segment-conditional CQR: one Q per cell, with parent fallback for thin cells."""

    alpha: float
    axes: tuple[str, ...]
    floor: int = 1_000
    _q_by_cell: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)
    _n_by_cell: dict[tuple[tuple[str, str], ...], int] = field(default_factory=dict)
    _global_q: float = 0.0
    _fallbacks: int = 0
    _exact: int = 0

    def _cells(self, segments: pl.DataFrame) -> list[tuple[tuple[str, str], ...]]:
        cols = [segments[a].to_numpy() for a in self.axes]
        return [
            tuple((a, str(v)) for a, v in zip(self.axes, row, strict=True))
            for row in zip(*cols, strict=True)
        ]

    def fit(
        self,
        actual: NDArray[np.float64],
        lo: NDArray[np.float64],
        hi: NDArray[np.float64],
        segments: pl.DataFrame,
    ) -> MondrianCQR:
        from eta.calibration.mondrian import build_cell_plan

        scores = conformity_scores(actual, lo, hi)
        self._global_q = conformal_quantile(scores, self.alpha)

        plan = build_cell_plan(segments, self.axes, self.floor)
        cells = self._cells(segments)
        by_cell: dict[tuple[tuple[str, str], ...], list[float]] = {}
        for cell, score in zip(cells, scores, strict=True):
            by_cell.setdefault(cell, []).append(float(score))

        # Score pools for every margin, so a thin cell can borrow from a parent.
        pools: dict[tuple[tuple[str, str], ...], list[float]] = {}
        for cell, vals in by_cell.items():
            for size in range(len(cell) + 1):
                from itertools import combinations

                for subset in combinations(cell, size):
                    pools.setdefault(subset, []).extend(vals)

        self._exact = 0
        self._fallbacks = 0
        for cell in by_cell:
            resolution = plan.resolution_for(cell)
            served = resolution.served_by
            pool = pools.get(served, [])
            self._q_by_cell[cell] = conformal_quantile(np.asarray(pool), self.alpha)
            self._n_by_cell[cell] = len(pool)
            if resolution.is_exact:
                self._exact += 1
            else:
                self._fallbacks += 1

        log.info(
            "mondrian_cqr_fitted",
            alpha=self.alpha,
            cells=len(by_cell),
            exact=self._exact,
            fell_back=self._fallbacks,
            floor=self.floor,
            global_q_seconds=round(self._global_q, 1),
            median_cell_q=round(float(np.median(list(self._q_by_cell.values()))), 1),
        )
        return self

    def apply(
        self, lo: NDArray[np.float64], hi: NDArray[np.float64], segments: pl.DataFrame
    ) -> ConformalInterval:
        q = np.asarray(
            [self._q_by_cell.get(cell, self._global_q) for cell in self._cells(segments)],
            dtype=np.float64,
        )
        return ConformalInterval(lo=lo - q, hi=hi + q)

    @property
    def cells(self) -> int:
        return len(self._q_by_cell)

    @property
    def fell_back(self) -> int:
        return self._fallbacks
