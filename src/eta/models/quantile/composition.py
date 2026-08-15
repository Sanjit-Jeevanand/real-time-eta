"""Monotonic composition -- fix 2: a base quantile plus non-negative increments.

The identity this rests on is that quantiles are equivariant under adding any
function of x:

    Q_a(y | x) = m(x) + Q_a(y - m(x) | x)

So fitting a booster at level `a` on the residual `y - Q_prev(x)` and adding it back
targets the same quantile as fitting on `y` directly. That identity holds at the
**population** level; a finite-sample fit recovers it only approximately, and the
clamp then modifies the learned increment wherever it came out negative. Both
caveats are real -- the claim is that the construction targets the right quantity,
not that it reproduces it exactly.

What the reparameterisation buys: the increment is now the modelled object, so
clamping it at zero enforces ordering without touching the base prediction.

That is the asymmetry against post-hoc sorting. Sorting can move any level, including
the one being served. This only ever raises a level that came out too low, and never
revises P50.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.logging import get_logger
from eta.models.quantile.lgbm import BASE_PARAMS, TARGET, default_params

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import polars as pl
    from numpy.typing import NDArray

    from eta.types import Quantile

__all__ = ["ClampStats", "MonotonicComposition"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ClampStats:
    """How much the non-negativity clamp actually changed, per level.

    Reported because "a crossing fix" and "a different model" are different claims.
    If the raw crossing rate is 0.58% but the clamp rewrites 15% of predictions, the
    composition is not repairing crossings -- it is a distinct parameterisation that
    happens to be monotone, and it has to be compared as one.
    """

    alpha: float
    rows: int
    clamped_rows: int
    clamped_share: float
    mean_adjustment_s: float
    p95_adjustment_s: float
    max_adjustment_s: float


@dataclass(slots=True)
class MonotonicComposition:
    """Chained quantile models: base at the lowest alpha, then clamped increments."""

    alphas: tuple[Quantile, ...]
    features: Sequence[str]
    params_by_alpha: dict[float, dict[str, Any]] = field(default_factory=dict)
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    name: str = "lgbm_composed"
    _boosters: list[Any] = field(default_factory=list)
    _clamp_stats: list[ClampStats] = field(default_factory=list)
    _clamped_any: Any = None

    def _params(self, alpha: Quantile, seed: int) -> dict[str, Any]:
        tuned = self.params_by_alpha.get(float(alpha), default_params())
        return BASE_PARAMS | tuned | {"alpha": float(alpha), "seed": seed}

    def fit(self, train: pl.DataFrame, seed: int, valid: pl.DataFrame | None = None) -> None:
        import lightgbm as lgb

        x = train.select(self.features).to_numpy()
        y = train[TARGET].to_numpy().astype(np.float64)
        vx = vy = None
        if valid is not None:
            vx = valid.select(self.features).to_numpy()
            vy = valid[TARGET].to_numpy().astype(np.float64)

        self._boosters = []
        # Running prediction of the previous level, which each increment is relative to.
        running = np.zeros(y.shape, dtype=np.float64)
        running_valid = np.zeros(vy.shape, dtype=np.float64) if vy is not None else None

        for i, alpha in enumerate(self.alphas):
            residual = y - running
            dataset = lgb.Dataset(x, label=residual, feature_name=list(self.features))
            valid_sets = None
            callbacks: list[Callable[..., Any]] = []
            if vx is not None and vy is not None and running_valid is not None:
                valid_sets = [lgb.Dataset(vx, label=vy - running_valid, reference=dataset)]
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

            booster = lgb.train(
                self._params(alpha, seed),
                dataset,
                num_boost_round=self.num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )
            self._boosters.append(booster)

            step = np.asarray(booster.predict(x), dtype=np.float64)
            # The base level is free to be any value; every step after it must be >= 0.
            running = running + (step if i == 0 else np.maximum(step, 0.0))
            if running_valid is not None and vx is not None:
                vstep = np.asarray(booster.predict(vx), dtype=np.float64)
                running_valid = running_valid + (vstep if i == 0 else np.maximum(vstep, 0.0))

            log.info(
                "composition_stage_fitted",
                alpha=alpha,
                stage="base" if i == 0 else "increment",
                seed=seed,
                iterations=int(booster.current_iteration()),
                clamped_fraction=(0.0 if i == 0 else float(np.mean(step < 0.0))),
            )

    def predict_matrix(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        if not self._boosters:
            msg = "monotonic composition has not been fitted"
            raise RuntimeError(msg)
        x = frame.select(self.features).to_numpy()
        running = np.zeros(frame.height, dtype=np.float64)
        columns: list[NDArray[np.float64]] = []
        self._clamp_stats = []
        clamped_any = np.zeros(frame.height, dtype=bool)
        for i, booster in enumerate(self._boosters):
            step = np.asarray(booster.predict(x), dtype=np.float64)
            if i == 0:
                running = running + step
            else:
                negative = step < 0.0
                clamped_any |= negative
                # The clamp moves the prediction by exactly |step| where step < 0.
                adjust = np.where(negative, -step, 0.0)
                self._clamp_stats.append(
                    ClampStats(
                        alpha=float(self.alphas[i]),
                        rows=int(step.size),
                        clamped_rows=int(negative.sum()),
                        clamped_share=float(np.mean(negative)),
                        mean_adjustment_s=float(adjust[negative].mean()) if negative.any() else 0.0,
                        p95_adjustment_s=(
                            float(np.percentile(adjust[negative], 95)) if negative.any() else 0.0
                        ),
                        max_adjustment_s=float(adjust.max()) if negative.any() else 0.0,
                    )
                )
                running = running + np.maximum(step, 0.0)
            columns.append(running.copy())
        self._clamped_any = clamped_any
        return np.column_stack(columns)

    @property
    def clamped_any(self) -> NDArray[np.bool_]:
        """Per-row: was any level's increment clamped in the last prediction?

        Exposed so the cost improvement can be split into rows the clamp touched
        and rows it did not. If the win lives entirely on clamped rows, the claim
        is about the constraint; if it survives on untouched rows, it is about the
        residual parameterisation.
        """
        if self._clamped_any is None:
            msg = "predict_matrix has not been called"
            raise RuntimeError(msg)
        out: NDArray[np.bool_] = self._clamped_any
        return out

    @property
    def clamp_stats(self) -> list[ClampStats]:
        """Populated by the most recent `predict_matrix` call."""
        return list(self._clamp_stats)

    def predict(self, frame: pl.DataFrame, alpha: Quantile) -> NDArray[np.float64]:
        return self.predict_matrix(frame)[:, self.alphas.index(alpha)]
