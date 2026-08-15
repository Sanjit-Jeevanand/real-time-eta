"""LightGBM quantile models, and the tuner that never sees cal or test."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import polars as pl
    from numpy.typing import NDArray

    from eta.types import Quantile

__all__ = [
    "TUNABLE",
    "LgbmQuantile",
    "QuantileBundle",
    "default_params",
    "sampler_seed_for",
    "tune_on_validation",
]

log = get_logger(__name__)

TARGET: Final = "total_time_s"

#: Fixed across every quantile so the comparison is about the loss, not the budget.
BASE_PARAMS: Final[dict[str, Any]] = {
    "objective": "quantile",
    "metric": "quantile",
    "verbosity": -1,
    "num_threads": 0,
}

TUNABLE: Final = (
    "learning_rate",
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l2",
)


def sampler_seed_for(seed: int, alpha: float) -> int:
    """Distinct Optuna sampler seed per (seed, quantile level).

    Sharing one seed across levels makes TPE's random startup phase draw identical
    configurations for every alpha, so a small budget returns identical "tuned"
    parameters for all of them. Separating them is what makes the three searches
    independent.
    """
    return seed * 1000 + round(alpha * 100)


def default_params() -> dict[str, Any]:
    """The L2 baseline's settings, so an untuned quantile model is not a strawman."""
    return {
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 0.0,
    }


@dataclass(slots=True)
class LgbmQuantile:
    """One booster trained at one quantile level with the pinball objective."""

    alpha: Quantile
    features: Sequence[str]
    params: dict[str, Any] = field(default_factory=default_params)
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    _booster: Any = None
    _best_iteration: int = 0

    @property
    def name(self) -> str:
        return f"lgbm_q{round(self.alpha * 100)}"

    def fit(self, train: pl.DataFrame, seed: int, valid: pl.DataFrame | None = None) -> None:
        import lightgbm as lgb

        x = train.select(self.features).to_numpy()
        y = train[TARGET].to_numpy().astype(np.float64)
        dataset = lgb.Dataset(x, label=y, feature_name=list(self.features))

        params = BASE_PARAMS | self.params | {"alpha": float(self.alpha), "seed": seed}
        valid_sets = None
        callbacks: list[Callable[..., Any]] = []
        if valid is not None:
            vx = valid.select(self.features).to_numpy()
            vy = valid[TARGET].to_numpy().astype(np.float64)
            valid_sets = [lgb.Dataset(vx, label=vy, reference=dataset)]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self._booster = lgb.train(
            params,
            dataset,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self._best_iteration = int(self._booster.current_iteration())
        log.info(
            "lgbm_quantile_fitted",
            alpha=self.alpha,
            seed=seed,
            rows=train.height,
            iterations=self._best_iteration,
        )

    def predict(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        if self._booster is None:
            msg = f"{self.name} has not been fitted"
            raise RuntimeError(msg)
        x = frame.select(self.features).to_numpy()
        return np.asarray(self._booster.predict(x), dtype=np.float64)


@dataclass(slots=True)
class QuantileBundle:
    """A set of independently-trained quantile models -- the thing that can cross.

    Independence is the point: each level is fitted without reference to the others,
    which is what makes the crossing rate in `eta.models.quantile.crossing` a real
    measurement rather than a property that was designed away.
    """

    alphas: tuple[Quantile, ...]
    features: Sequence[str]
    params_by_alpha: dict[float, dict[str, Any]] = field(default_factory=dict)
    num_boost_round: int = 600
    name: str = "lgbm_quantile"
    _models: list[LgbmQuantile] = field(default_factory=list)

    def fit(self, train: pl.DataFrame, seed: int, valid: pl.DataFrame | None = None) -> None:
        self._models = []
        for alpha in self.alphas:
            model = LgbmQuantile(
                alpha=alpha,
                features=self.features,
                params=self.params_by_alpha.get(float(alpha), default_params()),
                num_boost_round=self.num_boost_round,
            )
            model.fit(train, seed=seed, valid=valid)
            self._models.append(model)

    def predict_matrix(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        """Return an (n_rows, n_alphas) matrix, columns in ascending alpha order."""
        if not self._models:
            msg = "quantile bundle has not been fitted"
            raise RuntimeError(msg)
        return np.column_stack([m.predict(frame) for m in self._models])

    def predict(self, frame: pl.DataFrame, alpha: Quantile) -> NDArray[np.float64]:
        idx = self.alphas.index(alpha)
        return self.predict_matrix(frame)[:, idx]


def tune_on_validation(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    features: Sequence[str],
    alpha: Quantile,
    trials: int,
    seed: int = 0,
    num_boost_round: int = 400,
) -> dict[str, Any]:
    """Optuna search for one quantile level.

    The only frames this function accepts are train and validation. Calibration and
    test are not parameters, so a tuning run cannot read them even by mistake -- the
    discipline is enforced by the signature, not by a comment. Selection is on
    validation pinball loss at the same alpha the model is trained on.

    The sampler seed is derived from `alpha`, not shared across levels. With one seed
    for every level, TPE's random startup phase draws the *same* configurations for
    each alpha, and on a small budget the winner tends to fall inside that shared
    phase -- which returns bit-identical parameters for all three levels and makes
    "tuned per level" mean nothing. Measured: with a shared seed and 20 trials, all
    three levels came back identical to nine decimal places.
    """
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler_seed = sampler_seed_for(seed, float(alpha))

    x = train.select(features).to_numpy()
    y = train[TARGET].to_numpy().astype(np.float64)
    vx = valid.select(features).to_numpy()
    vy = valid[TARGET].to_numpy().astype(np.float64)

    def objective(trial: optuna.Trial) -> float:
        params = BASE_PARAMS | {
            "alpha": float(alpha),
            "seed": seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        dataset = lgb.Dataset(x, label=y, feature_name=list(features))
        valid_set = lgb.Dataset(vx, label=vy, reference=dataset)
        booster = lgb.train(
            params,
            dataset,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = np.asarray(booster.predict(vx), dtype=np.float64)
        delta = vy - pred
        return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=sampler_seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = default_params() | dict(study.best_params)
    best["bagging_freq"] = 1
    log.info(
        "quantile_tuned",
        alpha=alpha,
        trials=trials,
        sampler_seed=sampler_seed,
        best_val_pinball=round(study.best_value, 4),
        tuned_on="validation split only",
        **{k: best[k] for k in TUNABLE if k in best},
    )
    return best
