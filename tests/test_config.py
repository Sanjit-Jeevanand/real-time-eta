from __future__ import annotations

import datetime as dt
import math

import pytest
from pydantic import ValidationError

from eta.config import CostConfig, ModelConfig, Settings, SplitConfig, get_settings

BASE_SPLITS = {
    "train_start": dt.date(2023, 1, 1),
    "train_end": dt.date(2023, 10, 1),
    "cal_end": dt.date(2023, 11, 15),
    "val_end": dt.date(2023, 12, 15),
    "test_end": dt.date(2024, 1, 15),
}


@pytest.mark.parametrize(
    ("late", "early", "expected"),
    [(2.0, 1.0, 2 / 3), (3.0, 1.0, 0.75), (5.0, 1.0, 5 / 6), (1.0, 1.0, 0.5)],
)
def test_optimal_quantile_follows_cost_ratio(late: float, early: float, expected: float) -> None:
    assert CostConfig(lambda_late=late, lambda_early=early).optimal_quantile == pytest.approx(
        expected
    )


def test_pinball_minimised_at_the_derived_quantile() -> None:
    samples = [10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 22.0, 30.0, 55.0]
    cost = CostConfig(lambda_late=3.0, lambda_early=1.0)

    def total(promised: float) -> float:
        return sum(cost.business_cost(a, promised) for a in samples)

    best = min((float(x) for x in range(10, 56)), key=total)

    empirical_q_star = sorted(samples)[math.ceil(cost.optimal_quantile * len(samples)) - 1]
    assert best == pytest.approx(empirical_q_star)

    assert total(best) < total(sum(samples) / len(samples))
    assert total(best) < total(sorted(samples)[len(samples) // 2])


def test_business_cost_is_asymmetric() -> None:
    cost = CostConfig(lambda_late=3.0, lambda_early=1.0)
    assert cost.business_cost(actual=100.0, promised=90.0) == pytest.approx(30.0)
    assert cost.business_cost(actual=90.0, promised=100.0) == pytest.approx(10.0)


def test_cost_config_rejects_non_positive_weights() -> None:
    with pytest.raises(ValidationError):
        CostConfig(lambda_late=0.0, lambda_early=1.0)


def test_served_quantile_must_be_trained() -> None:
    with pytest.raises(ValidationError, match=r"not in model\.quantiles"):
        Settings(
            splits=SplitConfig(**BASE_SPLITS),
            cost=CostConfig(lambda_late=5.0, lambda_early=1.0),
        )


def test_matching_ratio_and_quantiles_load() -> None:
    s = Settings(
        splits=SplitConfig(**BASE_SPLITS),
        cost=CostConfig(lambda_late=5.0, lambda_early=1.0),
        model=ModelConfig(quantiles=(0.05, 0.5, 5 / 6, 0.95), reported_quantiles=(0.5, 5 / 6)),
    )
    assert s.cost.optimal_quantile == pytest.approx(5 / 6)


def test_target_coverage_must_be_bracketed() -> None:
    with pytest.raises(ValidationError, match="no interval to conformalise"):
        Settings(
            splits=SplitConfig(**BASE_SPLITS),
            calibration={"target_coverage": 0.99},
        )


def test_split_boundaries_must_strictly_increase() -> None:
    bad = {**BASE_SPLITS, "cal_end": dt.date(2023, 9, 1)}
    with pytest.raises(ValidationError, match="strictly increase"):
        SplitConfig(**bad)


def test_calibration_window_is_non_empty() -> None:
    assert SplitConfig(**BASE_SPLITS).calibration_days == 45


def test_multi_seed_reporting_is_enforced() -> None:
    with pytest.raises(ValidationError, match="at least 3 seeds"):
        ModelConfig(seeds=(0,))


def test_quantiles_must_be_ascending_and_unique() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        ModelConfig(quantiles=(0.9, 0.5))
    with pytest.raises(ValidationError, match="duplicate"):
        ModelConfig(quantiles=(0.5, 0.5))


def test_trip_length_buckets_must_not_overlap() -> None:
    from eta.config import SegmentConfig

    with pytest.raises(ValidationError, match="below long_trip_min_miles"):
        SegmentConfig(short_trip_max_miles=9.0, long_trip_min_miles=8.0)


def test_repo_config_yaml_loads_and_derives_075() -> None:
    get_settings.cache_clear()
    s = get_settings()
    assert s.cost.optimal_quantile == pytest.approx(0.75)
    assert 0.75 in s.model.quantiles
    assert s.cqr_bracket == pytest.approx((0.05, 0.95))
    assert s.splits.train_start < s.splits.test_end
    assert s.calibration.max_segment_coverage_gap_pp == 5.0


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("ETA_COST__LAMBDA_LATE", "1.0")
    s = Settings()
    assert s.cost.optimal_quantile == pytest.approx(0.5)
    get_settings.cache_clear()


def test_settings_are_frozen() -> None:
    get_settings.cache_clear()
    s = get_settings()
    with pytest.raises(ValidationError):
        s.cost.lambda_late = 99.0
