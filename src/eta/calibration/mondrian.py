from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Final

import polars as pl

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "GLOBAL_CELL",
    "CellPlan",
    "CellResolution",
    "build_cell_plan",
    "parent_cells",
    "resolve_cell",
]

log = get_logger(__name__)

GLOBAL_CELL: Final = ()

type Cell = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CellResolution:
    requested: Cell
    served_by: Cell
    n: int
    dropped_axes: tuple[str, ...]

    @property
    def is_exact(self) -> bool:
        return self.requested == self.served_by

    @property
    def is_global(self) -> bool:
        return self.served_by == GLOBAL_CELL


@dataclass(frozen=True, slots=True)
class CellPlan:
    axes: tuple[str, ...]
    counts: Mapping[Cell, int]
    floor: int
    resolutions: Mapping[Cell, CellResolution]

    def resolution_for(self, cell: Cell) -> CellResolution:
        return self.resolutions[cell]

    @property
    def exact(self) -> int:
        return sum(1 for r in self.resolutions.values() if r.is_exact)

    @property
    def fell_back(self) -> int:
        return sum(1 for r in self.resolutions.values() if not r.is_exact and not r.is_global)

    @property
    def global_fallback(self) -> int:
        return sum(1 for r in self.resolutions.values() if r.is_global)


def parent_cells(cell: Cell) -> list[Cell]:
    parents: list[Cell] = []
    for keep in range(len(cell) - 1, -1, -1):
        for subset in combinations(cell, keep):
            parents.append(tuple(subset))
    return parents


def resolve_cell(cell: Cell, counts: Mapping[Cell, int], floor: int) -> CellResolution:
    if counts.get(cell, 0) >= floor:
        return CellResolution(cell, cell, counts.get(cell, 0), ())
    for parent in parent_cells(cell):
        if counts.get(parent, 0) >= floor:
            dropped = tuple(axis for axis, _ in cell if axis not in dict(parent))
            return CellResolution(cell, parent, counts[parent], dropped)
    return CellResolution(cell, GLOBAL_CELL, counts.get(GLOBAL_CELL, 0), tuple(a for a, _ in cell))


def _counts_for_all_margins(df: pl.DataFrame, axes: Sequence[str]) -> dict[Cell, int]:
    counts: dict[Cell, int] = {GLOBAL_CELL: df.height}
    for size in range(1, len(axes) + 1):
        for subset in combinations(axes, size):
            grouped = df.group_by(list(subset)).agg(pl.len().alias("n"))
            for row in grouped.iter_rows(named=True):
                key = tuple((axis, str(row[axis])) for axis in subset)
                counts[key] = int(row["n"])
    return counts


def build_cell_plan(calibration: pl.DataFrame, axes: Sequence[str], floor: int) -> CellPlan:
    counts = _counts_for_all_margins(calibration, axes)
    full = [c for c in counts if len(c) == len(axes)]
    resolutions = {cell: resolve_cell(cell, counts, floor) for cell in full}

    plan = CellPlan(tuple(axes), counts, floor, resolutions)
    log.info(
        "mondrian_plan_built",
        axes=list(axes),
        cells=len(full),
        exact=plan.exact,
        fell_back=plan.fell_back,
        global_fallback=plan.global_fallback,
        floor=floor,
    )
    return plan
