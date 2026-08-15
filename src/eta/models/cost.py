from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from eta.config import CostConfig
    from eta.types import Quantile

__all__ = [
    "business_cost",
    "business_cost_expr",
    "late_rate",
    "optimal_quantile",
    "pinball_loss",
    "pinball_to_cost_factor",
]


def optimal_quantile(lambda_late: float, lambda_early: float) -> Quantile:
    return lambda_late / (lambda_late + lambda_early)


def pinball_to_cost_factor(lambda_late: float, lambda_early: float) -> float:
    return lambda_late + lambda_early


def business_cost(
    actual: NDArray[np.float64], promised: NDArray[np.float64], cfg: CostConfig
) -> NDArray[np.float64]:
    delta = actual - promised
    return cfg.lambda_late * np.maximum(delta, 0.0) + cfg.lambda_early * np.maximum(-delta, 0.0)


def pinball_loss(
    actual: NDArray[np.float64], promised: NDArray[np.float64], q: float
) -> NDArray[np.float64]:
    delta = actual - promised
    return np.maximum(q * delta, (q - 1.0) * delta)


def business_cost_expr(actual: str, promised: str, cfg: CostConfig) -> pl.Expr:
    delta = pl.col(actual) - pl.col(promised)
    return (
        cfg.lambda_late * pl.max_horizontal(delta, pl.lit(0.0))
        + cfg.lambda_early * pl.max_horizontal(-delta, pl.lit(0.0))
    ).alias("business_cost")


def late_rate(actual: NDArray[np.float64], promised: NDArray[np.float64]) -> float:
    return float(np.mean(actual > promised))
