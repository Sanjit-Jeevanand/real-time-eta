"""Quantile crossing: measuring it, and the cheapest of the three fixes.

A set of independently-trained quantile models has no mechanism forcing
Q_0.5(x) <= Q_0.75(x) <= Q_0.9(x). Each booster minimises its own pinball loss and
nothing couples them, so on some rows the estimates come back out of order. That is
not a bug in the fit -- it is a missing constraint, and it has to be measured before
it can be argued about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "CrossingReport",
    "comparison_table",
    "crossing_report",
    "is_monotone",
    "sort_rows",
]


def is_monotone(matrix: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Per-row: are the columns non-decreasing left to right?"""
    out: NDArray[np.bool_] = np.all(np.diff(matrix, axis=1) >= 0.0, axis=1)
    return out


def sort_rows(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Post-hoc sorting -- fix 1.

    Cheap and total: sorting each row makes crossing impossible by construction.
    The cost is that it silently reassigns values between levels. If the P90 model
    was the one that went wrong on a row, sorting can hand its value to P75 and
    quietly move the promise the service actually serves.
    """
    return np.sort(matrix, axis=1)


@dataclass(frozen=True, slots=True)
class CrossingReport:
    rows: int
    crossing_rows: int
    crossing_rate: float
    pairwise: dict[str, float]
    worst_gap_s: float
    mean_gap_s: float

    def markdown(self) -> str:
        lines = [
            f"rows: {self.rows:,}",
            f"rows with >=1 crossing: {self.crossing_rows:,} (**{self.crossing_rate:.2%}**)",
            f"worst inversion: {self.worst_gap_s:.1f}s   mean inversion (crossing rows only): "
            f"{self.mean_gap_s:.1f}s",
            "",
            "| pair | crossing rate |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v:.2%} |" for k, v in self.pairwise.items()]
        return "\n".join(lines)


def comparison_table(reports: dict[str, list[CrossingReport]], enforced: dict[str, str]) -> str:
    """Crossing before and after each fix, across every seed.

    Reporting one seed would be misleading: the crossing rate of an unconstrained
    model varies a lot between seeds, and that variation is part of the finding.
    """
    import statistics as st

    lines = [
        "| strategy | what enforces ordering | crossing rate | worst inversion | seeds |",
        "|---|---|---|---|---|",
    ]
    for name, reps in reports.items():
        rates = [r.crossing_rate for r in reps]
        mean = st.fmean(rates)
        std = st.stdev(rates) if len(rates) > 1 else 0.0
        worst = max(r.worst_gap_s for r in reps) if reps else 0.0
        rate = "**none**" if mean == 0.0 and std == 0.0 else f"{mean:.2%} ± {std:.2%}"
        gap = "--" if mean == 0.0 and std == 0.0 else f"{worst:.0f}s"
        lines.append(f"| {name} | {enforced.get(name, '?')} | {rate} | {gap} | {len(reps)} |")
    return "\n".join(lines)


def crossing_report(matrix: NDArray[np.float64], alphas: Sequence[float]) -> CrossingReport:
    """Measure crossing on an (n_rows, n_alphas) matrix with ascending alphas."""
    if matrix.ndim != 2 or matrix.shape[1] != len(alphas):
        msg = f"matrix {matrix.shape} does not match {len(alphas)} alphas"
        raise ValueError(msg)
    if list(alphas) != sorted(alphas):
        msg = f"alphas must be ascending to talk about crossing: {alphas}"
        raise ValueError(msg)

    diffs = np.diff(matrix, axis=1)
    violated = diffs < 0.0
    per_row = violated.any(axis=1)

    pairwise = {
        f"P{round(alphas[i] * 100)} > P{round(alphas[i + 1] * 100)}": float(np.mean(violated[:, i]))
        for i in range(len(alphas) - 1)
    }

    inversions = -diffs[violated] if violated.any() else np.array([0.0])
    return CrossingReport(
        rows=int(matrix.shape[0]),
        crossing_rows=int(per_row.sum()),
        crossing_rate=float(np.mean(per_row)),
        pairwise=pairwise,
        worst_gap_s=float(np.max(inversions)),
        mean_gap_s=float(np.mean(inversions)),
    )
