"""Treelite compilation -- Phase 8 step 5.

LightGBM's own `predict` is built for batches: it dispatches through a generic tree
walker and pays interpreter and allocation overhead that a single row cannot
amortise. Treelite compiles the ensemble to C, so a one-row prediction becomes a
sequence of comparisons in a shared object.

The claim that compilation is faster is checked here rather than assumed, and the
compiled model is checked to predict the *same values* first -- a faster model that
disagrees with the trained one is not an optimisation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = ["CompiledEnsemble", "compile_boosters"]

log = get_logger(__name__)


@dataclass(slots=True)
class CompiledEnsemble:
    """Treelite-compiled quantile models behind the same call shape as the raw ones."""

    predictors: list[Any]

    def __call__(self, row: NDArray[np.float64]) -> NDArray[np.float64]:
        import tl2cgen

        batch = tl2cgen.DMatrix(np.ascontiguousarray(row, dtype=np.float32))
        return np.column_stack([p.predict(batch).reshape(-1) for p in self.predictors])


def compile_boosters(
    boosters: Sequence[Any], out_dir: Path, toolchain: str = "clang"
) -> CompiledEnsemble:
    """Compile each booster to a shared object and load it back."""
    import tl2cgen
    import treelite

    out_dir.mkdir(parents=True, exist_ok=True)
    predictors = []
    total_bytes = 0

    for i, booster in enumerate(boosters):
        model = treelite.frontend.from_lightgbm(booster)
        lib = out_dir / f"quantile_{i}.so"
        tl2cgen.export_lib(
            model,
            toolchain=toolchain,
            libpath=str(lib),
            params={"parallel_comp": 4},
            verbose=False,
        )
        total_bytes += lib.stat().st_size
        predictors.append(tl2cgen.Predictor(str(lib)))

    log.info(
        "boosters_compiled",
        models=len(predictors),
        toolchain=toolchain,
        total_bytes=total_bytes,
        total_mb=round(total_bytes / 1e6, 2),
    )
    return CompiledEnsemble(predictors=predictors)


def compare_predictions(
    raw: NDArray[np.float64], compiled: NDArray[np.float64], tol: float = 1e-4
) -> dict[str, float]:
    """A faster model that disagrees with the trained one is not an optimisation."""
    diff = np.abs(raw - compiled)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "within_tolerance": float(np.mean(diff <= tol)),
    }


def time_single_row(
    predict: Any, row: NDArray[np.float64], repeats: int = 2_000
) -> dict[str, float]:
    """Single-row latency, which is the only shape the service ever calls."""
    predict(row)  # warm any lazy allocation
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(row)
        samples.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(samples)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }
