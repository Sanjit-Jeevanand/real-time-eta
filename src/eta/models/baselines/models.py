from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import polars as pl

from eta.logging import get_logger
from eta.types import stat_float

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

__all__ = [
    "Baseline",
    "HistoricalMean",
    "LgbmL2",
    "OsrmMultiplier",
]

log = get_logger(__name__)

TARGET: Final = "total_time_s"
GLOBAL_KEY: Final = -1


class Baseline(Protocol):
    name: str

    def fit(self, train: pl.DataFrame, seed: int) -> None: ...
    def predict(self, frame: pl.DataFrame) -> NDArray[np.float64]: ...


@dataclass(slots=True)
class HistoricalMean:
    name: str = "historical_mean"
    _by_pair_hour: dict[tuple[int, int, int], float] = field(default_factory=dict)
    _by_pair: dict[tuple[int, int], float] = field(default_factory=dict)
    _global: float = 0.0
    min_rows: int = 30

    def fit(self, train: pl.DataFrame, seed: int) -> None:
        del seed
        hour = pl.col("request_datetime").dt.hour()
        pair_hour = (
            train.with_columns(hour.alias("hour"))
            .group_by("pu_zone", "do_zone", "hour")
            .agg(pl.col(TARGET).mean().alias("mean"), pl.len().alias("n"))
            .filter(pl.col("n") >= self.min_rows)
        )
        self._by_pair_hour = {
            (int(r["pu_zone"]), int(r["do_zone"]), int(r["hour"])): float(r["mean"])
            for r in pair_hour.iter_rows(named=True)
        }
        pair = (
            train.group_by("pu_zone", "do_zone")
            .agg(pl.col(TARGET).mean().alias("mean"), pl.len().alias("n"))
            .filter(pl.col("n") >= self.min_rows)
        )
        self._by_pair = {
            (int(r["pu_zone"]), int(r["do_zone"])): float(r["mean"])
            for r in pair.iter_rows(named=True)
        }
        self._global = stat_float(train[TARGET].mean())
        log.info(
            "historical_mean_fitted",
            pair_hour_cells=len(self._by_pair_hour),
            pair_cells=len(self._by_pair),
            global_mean_s=round(self._global, 1),
        )

    def predict(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        hours = frame["request_datetime"].dt.hour().to_list()
        pu = frame["pu_zone"].to_list()
        do = frame["do_zone"].to_list()
        out = np.empty(frame.height, dtype=np.float64)
        for i, (a, b, h) in enumerate(zip(pu, do, hours, strict=True)):
            key = (int(a), int(b), int(h))
            value = self._by_pair_hour.get(key)
            if value is None:
                value = self._by_pair.get((int(a), int(b)), self._global)
            out[i] = value
        return out


@dataclass(slots=True)
class OsrmMultiplier:
    name: str = "osrm_multiplier"
    _multiplier: float = 1.0
    _by_hour: dict[int, float] = field(default_factory=dict)
    _global_mean: float = 0.0
    use_hourly: bool = True

    def fit(self, train: pl.DataFrame, seed: int) -> None:
        del seed
        free_flow = train["free_flow_duration_s"].to_numpy().astype(np.float64)
        actual = train[TARGET].to_numpy().astype(np.float64)
        usable = free_flow > 0
        self._multiplier = float(
            np.sum(actual[usable] * free_flow[usable]) / np.sum(free_flow[usable] ** 2)
        )
        self._global_mean = float(actual.mean())

        if self.use_hourly:
            hours = train["request_datetime"].dt.hour().to_numpy()
            for h in range(24):
                mask = usable & (hours == h)
                if mask.sum() >= 100:
                    self._by_hour[h] = float(
                        np.sum(actual[mask] * free_flow[mask]) / np.sum(free_flow[mask] ** 2)
                    )
        log.info(
            "osrm_multiplier_fitted",
            multiplier=round(self._multiplier, 4),
            hourly=len(self._by_hour),
            spread=(
                round(max(self._by_hour.values()) - min(self._by_hour.values()), 4)
                if self._by_hour
                else 0.0
            ),
        )

    def predict(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        free_flow = frame["free_flow_duration_s"].to_numpy().astype(np.float64)
        hours = frame["request_datetime"].dt.hour().to_numpy()
        mult = np.full(free_flow.shape, self._multiplier, dtype=np.float64)
        for h, m in self._by_hour.items():
            mult[hours == h] = m
        out = free_flow * mult
        return np.where(free_flow > 0, out, self._global_mean)


@dataclass(slots=True)
class LgbmL2:
    features: Sequence[str]
    name: str = "lgbm_l2"
    num_boost_round: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 127
    min_data_in_leaf: int = 200
    _booster: object = None

    def fit(self, train: pl.DataFrame, seed: int, valid: pl.DataFrame | None = None) -> None:
        import lightgbm as lgb

        x = train.select(self.features).to_numpy()
        y = train[TARGET].to_numpy().astype(np.float64)
        dataset = lgb.Dataset(x, label=y, feature_name=list(self.features))

        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": seed,
            "verbosity": -1,
            "num_threads": 0,
        }
        valid_sets = None
        callbacks: list[Callable[..., Any]] = []
        if valid is not None:
            vx = valid.select(self.features).to_numpy()
            vy = valid[TARGET].to_numpy().astype(np.float64)
            valid_sets = [lgb.Dataset(vx, label=vy, reference=dataset)]
            callbacks.append(lgb.early_stopping(50, verbose=False))

        self._booster = lgb.train(
            params,
            dataset,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        log.info(
            "lgbm_l2_fitted",
            seed=seed,
            rows=train.height,
            features=len(self.features),
            iterations=self._booster.current_iteration(),
        )

    def predict(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        if self._booster is None:
            msg = "lgbm_l2 has not been fitted"
            raise RuntimeError(msg)
        x = frame.select(self.features).to_numpy()
        out: NDArray[np.float64] = self._booster.predict(x)  # type: ignore[attr-defined]
        return np.asarray(out, dtype=np.float64)
