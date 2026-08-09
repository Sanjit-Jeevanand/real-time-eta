from __future__ import annotations

import datetime as dt
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field, computed_field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from eta.types import Quantile

__all__ = ["Settings", "get_settings"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

PositiveFloat = Annotated[float, Field(gt=0)]


class CostConfig(BaseModel):
    model_config = {"frozen": True}

    lambda_late: PositiveFloat = 3.0
    lambda_early: PositiveFloat = 1.0

    @computed_field
    @property
    def optimal_quantile(self) -> Quantile:
        return self.lambda_late / (self.lambda_late + self.lambda_early)

    @computed_field
    @property
    def late_early_ratio(self) -> float:
        return self.lambda_late / self.lambda_early

    def pinball_loss(self, actual: float, predicted: float) -> float:
        q = self.optimal_quantile
        delta = actual - predicted
        return q * delta if delta >= 0 else (q - 1.0) * delta

    def business_cost(self, actual: float, promised: float) -> float:
        return self.lambda_late * max(0.0, actual - promised) + self.lambda_early * max(
            0.0, promised - actual
        )


class SplitConfig(BaseModel):
    model_config = {"frozen": True}

    train_start: dt.date
    train_end: dt.date
    cal_end: dt.date
    val_end: dt.date
    test_end: dt.date
    holdout_weeks: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _strictly_increasing(self) -> Self:
        bounds = [
            ("train_start", self.train_start),
            ("train_end", self.train_end),
            ("cal_end", self.cal_end),
            ("val_end", self.val_end),
            ("test_end", self.test_end),
        ]
        for (lname, lval), (rname, rval) in pairwise(bounds):
            if lval >= rval:
                raise ValueError(
                    f"split boundaries must strictly increase: {lname}={lval} >= {rname}={rval}"
                )
        return self

    @computed_field
    @property
    def calibration_days(self) -> int:
        return (self.cal_end - self.train_end).days


class SegmentConfig(BaseModel):
    model_config = {"frozen": True}

    peak_am_hours: tuple[int, int] = (6, 10)
    peak_pm_hours: tuple[int, int] = (16, 20)
    late_night_hours: tuple[int, int] = (23, 5)
    short_trip_max_miles: float = Field(default=2.0, gt=0)
    long_trip_min_miles: float = Field(default=8.0, gt=0)
    rain_mm_threshold: float = Field(default=0.2, ge=0)
    snow_mm_threshold: float = Field(default=0.5, ge=0)
    airport_zone_ids: tuple[int, ...] = (1, 132, 138)
    manhattan_core_borough: str = "Manhattan"

    @model_validator(mode="after")
    def _trip_length_buckets_ordered(self) -> Self:
        if self.short_trip_max_miles >= self.long_trip_min_miles:
            raise ValueError(
                "short_trip_max_miles must be below long_trip_min_miles; "
                f"got {self.short_trip_max_miles} >= {self.long_trip_min_miles}"
            )
        return self


class ModelConfig(BaseModel):
    model_config = {"frozen": True}

    quantiles: tuple[Quantile, ...] = (0.05, 0.5, 0.75, 0.9, 0.95)
    reported_quantiles: tuple[Quantile, ...] = (0.5, 0.75, 0.9)
    seeds: tuple[int, ...] = (0, 1, 2)
    optuna_trials: int = Field(default=100, ge=1)
    early_stopping_rounds: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def _quantiles_sorted_and_bounded(self) -> Self:
        if len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError(f"duplicate quantiles: {self.quantiles}")
        if list(self.quantiles) != sorted(self.quantiles):
            raise ValueError(f"quantiles must be ascending: {self.quantiles}")
        if not all(0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError(f"quantiles must lie strictly in (0, 1): {self.quantiles}")
        if len(self.seeds) < 3:
            raise ValueError("at least 3 seeds are required; every headline number is mean +/- std")
        missing = set(self.reported_quantiles) - set(self.quantiles)
        if missing:
            raise ValueError(f"reported_quantiles not trained: {sorted(missing)}")
        return self


class CalibrationConfig(BaseModel):
    model_config = {"frozen": True}

    target_coverage: float = Field(default=0.90, gt=0, lt=1)
    min_segment_samples: int = Field(default=5_000, ge=1)
    max_segment_coverage_gap_pp: float = Field(default=5.0, gt=0)
    mondrian: bool = True


class RoutingConfig(BaseModel):
    model_config = {"frozen": True}

    osrm_url: str = "http://127.0.0.1:5000"
    osrm_timeout_s: float = Field(default=10.0, gt=0)
    n_zones: int = Field(default=263, ge=1)
    zone_matrix_path: Path = Path("data/processed/zone_pair_matrix.parquet")
    zone_embedding_dims: int = Field(default=16, ge=1)


class RedisConfig(BaseModel):
    model_config = {"frozen": True}

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    socket_timeout_s: float = Field(default=0.05, gt=0)
    congestion_windows_min: tuple[int, ...] = (15, 30, 60)


class ServingConfig(BaseModel):
    model_config = {"frozen": True}

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=2, ge=1)
    p99_target_ms: float = Field(default=25.0, gt=0)
    degrade_on_redis_failure: bool = True


class PathsConfig(BaseModel):
    model_config = {"frozen": True}

    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")

    def resolve(self, root: Path = REPO_ROOT) -> dict[str, Path]:
        return {name: root / p for name, p in self.model_dump().items()}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ETA_",
        env_nested_delimiter="__",
        yaml_file=DEFAULT_CONFIG_PATH,
        extra="forbid",
        frozen=True,
    )

    env: str = "dev"
    log_level: str = "INFO"
    cost: CostConfig = CostConfig()
    splits: SplitConfig
    segments: SegmentConfig = SegmentConfig()
    model: ModelConfig = ModelConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    routing: RoutingConfig = RoutingConfig()
    redis: RedisConfig = RedisConfig()
    serving: ServingConfig = ServingConfig()
    paths: PathsConfig = PathsConfig()

    @model_validator(mode="after")
    def _served_quantile_is_trained(self) -> Self:
        q_star = self.cost.optimal_quantile
        if not any(abs(q - q_star) < 1e-9 for q in self.model.quantiles):
            raise ValueError(
                f"cost ratio {self.cost.lambda_late}:{self.cost.lambda_early} implies "
                f"q*={q_star:.4f}, which is not in model.quantiles={self.model.quantiles}. "
                "Either train that quantile or change the cost ratio -- do not serve the "
                "nearest available level."
            )
        return self

    @computed_field
    @property
    def cqr_bracket(self) -> tuple[Quantile, Quantile]:
        alpha = 1.0 - self.calibration.target_coverage
        return (round(alpha / 2.0, 10), round(1.0 - alpha / 2.0, 10))

    @model_validator(mode="after")
    def _target_coverage_is_reachable(self) -> Self:
        lo, hi = self.cqr_bracket
        trained = self.model.quantiles
        if not any(abs(q - lo) < 1e-9 for q in trained) or not any(
            abs(q - hi) < 1e-9 for q in trained
        ):
            raise ValueError(
                f"target_coverage={self.calibration.target_coverage} needs quantiles "
                f"[{lo:.3f}, {hi:.3f}] trained; model.quantiles={trained} does not contain them. "
                "CQR would have no interval to conformalise."
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)
