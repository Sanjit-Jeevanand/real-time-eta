"""Per-segment isotonic recalibration -- Phase 7B.

Fitted on the held-out calibration split, one map per (segment, quantile level).

The map is built from binned residual quantiles rather than from a direct isotonic
fit of y on the prediction. The reason is that isotonic regression minimises squared
error, so fitting it to y would drag the output toward the conditional *mean* --
which is precisely the estimator this project spent Phase 5 arguing against. Instead
each bin contributes the residual quantile that would make coverage exact in that
bin, and `IsotonicRegression` is used for what it is actually good for: forcing the
resulting correction to be monotone in the prediction, so a recalibrated P75 can
never invert.

Segments below the sample floor fall back to the global map, and which ones did is
reported rather than hidden -- a per-segment correction fitted on 200 rows is noise
wearing the costume of a fix.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from numpy.typing import NDArray

__all__ = ["IsotonicCalibrator", "SegmentMap"]

log = get_logger(__name__)

GLOBAL_KEY = ("global", "all")


@dataclass(frozen=True, slots=True)
class SegmentMap:
    key: tuple[str, str]
    rows: int
    model: Any
    fell_back: bool

    def apply(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(values), dtype=np.float64)


@dataclass(slots=True)
class IsotonicCalibrator:
    """One monotone correction per (axis, bucket, quantile level)."""

    alphas: tuple[float, ...]
    axis: str
    min_rows: int = 5_000
    bins: int = 20
    _maps: dict[tuple[tuple[str, str], int], SegmentMap] = field(default_factory=dict)
    _fallbacks: list[str] = field(default_factory=list)

    def _fit_one(self, pred: NDArray[np.float64], actual: NDArray[np.float64], alpha: float) -> Any:
        from sklearn.isotonic import IsotonicRegression

        order = np.argsort(pred)
        p, a = pred[order], actual[order]
        edges = np.linspace(0, p.size, min(self.bins, max(p.size // 200, 2)) + 1).astype(int)

        xs: list[float] = []
        ys: list[float] = []
        for lo, hi in itertools.pairwise(edges):
            if hi - lo < 20:
                continue
            chunk_p, chunk_a = p[lo:hi], a[lo:hi]
            centre = float(np.median(chunk_p))
            # The shift that would make coverage exactly `alpha` inside this bin.
            shift = float(np.quantile(chunk_a - chunk_p, alpha))
            xs.append(centre)
            ys.append(centre + shift)

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        if len(xs) < 2:
            iso.fit(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
            return iso
        iso.fit(np.asarray(xs), np.asarray(ys))
        return iso

    def fit(
        self,
        matrix: NDArray[np.float64],
        actual: NDArray[np.float64],
        segments: pl.DataFrame,
    ) -> IsotonicCalibrator:
        self._maps = {}
        self._fallbacks = []

        for j, alpha in enumerate(self.alphas):
            self._maps[(GLOBAL_KEY, j)] = SegmentMap(
                GLOBAL_KEY, int(actual.size), self._fit_one(matrix[:, j], actual, alpha), False
            )

        values = segments[self.axis].to_numpy()
        for bucket in np.unique(values):
            if bucket is None:
                continue
            mask = values == bucket
            n = int(mask.sum())
            key = (self.axis, str(bucket))
            if n < self.min_rows:
                self._fallbacks.append(str(bucket))
                for j in range(len(self.alphas)):
                    g = self._maps[(GLOBAL_KEY, j)]
                    self._maps[(key, j)] = SegmentMap(key, n, g.model, True)
                continue
            for j, alpha in enumerate(self.alphas):
                self._maps[(key, j)] = SegmentMap(
                    key, n, self._fit_one(matrix[mask, j], actual[mask], alpha), False
                )

        log.info(
            "isotonic_fitted",
            axis=self.axis,
            buckets=len(np.unique(values)),
            levels=len(self.alphas),
            min_rows=self.min_rows,
            fell_back_to_global=self._fallbacks,
        )
        return self

    def apply(self, matrix: NDArray[np.float64], segments: pl.DataFrame) -> NDArray[np.float64]:
        out = np.empty_like(matrix)
        values = segments[self.axis].to_numpy()
        for j in range(len(self.alphas)):
            column = matrix[:, j]
            corrected = self._maps[(GLOBAL_KEY, j)].apply(column)
            for bucket in np.unique(values):
                if bucket is None:
                    continue
                key = ((self.axis, str(bucket)), j)
                if key not in self._maps:
                    continue
                mask = values == bucket
                corrected[mask] = self._maps[key].apply(column[mask])
            out[:, j] = corrected
        # Recalibrating each level independently can re-introduce crossing; the
        # running max restores order without touching the lowest level.
        return np.maximum.accumulate(out, axis=1)

    @property
    def fallback_buckets(self) -> Sequence[str]:
        return tuple(self._fallbacks)
