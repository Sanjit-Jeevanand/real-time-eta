from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from eta.config import CostConfig
from eta.models.cost import (
    business_cost,
    business_cost_expr,
    late_rate,
    optimal_quantile,
    pinball_loss,
    pinball_to_cost_factor,
)
from eta.models.evaluate import evaluate, summarise_seeds
from eta.models.sampling import HASH_COLUMN, Tier, assign_hash, tier_filter

RNG = np.random.default_rng(0)


def _sample(n: int = 20_000) -> np.ndarray:
    return RNG.lognormal(mean=7.0, sigma=0.5, size=n)


# ------------------------------------------------------- the derivation -----
@pytest.mark.parametrize(
    ("late", "early", "expected"),
    [(2.0, 1.0, 2 / 3), (3.0, 1.0, 0.75), (5.0, 1.0, 5 / 6), (1.0, 1.0, 0.5)],
)
def test_optimal_quantile_formula(late: float, early: float, expected: float) -> None:
    assert optimal_quantile(late, early) == pytest.approx(expected)


def test_pinball_at_q_star_equals_business_cost_up_to_a_constant() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample()
    promised = actual * RNG.uniform(0.6, 1.4, size=actual.size)

    cost = business_cost(actual, promised, cfg)
    pinball = pinball_loss(actual, promised, cfg.optimal_quantile)
    factor = pinball_to_cost_factor(cfg.lambda_late, cfg.lambda_early)

    assert np.allclose(cost, pinball * factor, rtol=1e-12)


def test_empirical_minimiser_is_the_q_star_quantile() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample(50_000)
    grid = np.quantile(actual, np.linspace(0.5, 0.95, 91))
    costs = [business_cost(actual, np.full(actual.size, p), cfg).mean() for p in grid]
    best = float(grid[int(np.argmin(costs))])
    target = float(np.quantile(actual, cfg.optimal_quantile))
    assert best == pytest.approx(target, rel=0.01)


def test_the_mean_is_not_the_cost_minimiser_under_asymmetry() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample(50_000)
    mean_cost = business_cost(actual, np.full(actual.size, actual.mean()), cfg).mean()
    q_star = float(np.quantile(actual, cfg.optimal_quantile))
    q_cost = business_cost(actual, np.full(actual.size, q_star), cfg).mean()
    assert q_cost < mean_cost
    assert q_star > actual.mean(), "under a 3:1 penalty the optimum sits above the mean"


def test_symmetric_cost_collapses_to_the_median() -> None:
    cfg = CostConfig(lambda_late=1.0, lambda_early=1.0)
    assert cfg.optimal_quantile == pytest.approx(0.5)
    actual = _sample(50_000)
    median = float(np.median(actual))
    med_cost = business_cost(actual, np.full(actual.size, median), cfg).mean()
    mean_cost = business_cost(actual, np.full(actual.size, actual.mean()), cfg).mean()
    assert med_cost <= mean_cost


def test_higher_late_penalty_pushes_the_promise_later() -> None:
    actual = _sample(50_000)
    previous = 0.0
    for ratio in (2.0, 3.0, 5.0):
        cfg = CostConfig(lambda_late=ratio, lambda_early=1.0)
        p = float(np.quantile(actual, cfg.optimal_quantile))
        assert p > previous
        previous = p


# ------------------------------------------------------------ mechanics -----
def test_business_cost_is_asymmetric_in_the_right_direction() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    late = business_cost(np.array([100.0]), np.array([90.0]), cfg)[0]
    early = business_cost(np.array([90.0]), np.array([100.0]), cfg)[0]
    assert late == pytest.approx(30.0)
    assert early == pytest.approx(10.0)
    assert late == 3 * early


def test_polars_and_numpy_cost_agree() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample(5_000)
    promised = actual * RNG.uniform(0.7, 1.3, size=actual.size)
    frame = pl.DataFrame({"a": actual, "p": promised}).with_columns(
        business_cost_expr("a", "p", cfg)
    )
    assert np.allclose(frame["business_cost"].to_numpy(), business_cost(actual, promised, cfg))


def test_late_rate_counts_strictly_late() -> None:
    actual = np.array([10.0, 10.0, 10.0])
    promised = np.array([9.0, 10.0, 11.0])
    assert late_rate(actual, promised) == pytest.approx(1 / 3)


# ------------------------------------------------------------ sampling ------
def test_hash_is_deterministic_and_order_independent() -> None:
    frame = pl.DataFrame(
        {
            "request_datetime": pl.Series(
                ["2023-06-01T08:00:00", "2023-06-01T09:00:00", "2023-06-02T10:00:00"]
            ).str.to_datetime(),
            "pu_zone": pl.Series([1, 2, 3], dtype=pl.UInt16),
            "do_zone": pl.Series([4, 5, 6], dtype=pl.UInt16),
            "hvfhs_license_num": ["HV0003", "HV0005", "HV0003"],
        }
    )
    a = assign_hash(frame.lazy()).collect()
    b = assign_hash(frame.reverse().lazy()).collect().reverse()
    assert a[HASH_COLUMN].to_list() == b[HASH_COLUMN].to_list()

    again = assign_hash(frame.lazy()).collect()
    assert a[HASH_COLUMN].to_list() == again[HASH_COLUMN].to_list()


def test_tiers_are_nested() -> None:
    frame = pl.DataFrame({HASH_COLUMN: pl.Series(range(10_000), dtype=pl.UInt16)})
    tune = frame.filter(tier_filter(Tier.TUNE))
    ablation = frame.filter(tier_filter(Tier.ABLATION))
    full = frame.filter(tier_filter(Tier.FULL))
    assert set(tune[HASH_COLUMN]) <= set(ablation[HASH_COLUMN]) <= set(full[HASH_COLUMN])
    assert tune.height == 300
    assert ablation.height == 2_500
    assert full.height == 10_000


def test_sampling_needs_a_key_column() -> None:
    with pytest.raises(ValueError, match="sampling key columns"):
        assign_hash(pl.DataFrame({"unrelated": [1]}).lazy()).collect()


# ------------------------------------------------------------- harness ------
def test_evaluate_reports_cost_mae_and_late_rate() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = np.array([600.0, 900.0, 1200.0, 1500.0])
    promised = np.array([700.0, 800.0, 1200.0, 1400.0])
    result = evaluate("m", 0, actual, promised, cfg)
    assert result.rows == 4
    assert result.late_rate == pytest.approx(0.5)
    assert result.mae == pytest.approx(75.0)
    assert result.business_cost == pytest.approx(business_cost(actual, promised, cfg).mean())


def test_evaluate_slices_by_segment() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    n = 4_000
    actual = np.full(n, 600.0)
    promised = np.full(n, 500.0)
    segments = pl.DataFrame({"seg_time": ["peak_pm"] * (n // 2) + ["off_peak"] * (n // 2)})
    result = evaluate("m", 0, actual, promised, cfg, segments)
    axes = {s.axis for s in result.segments}
    assert axes == {"seg_time"}
    assert {s.bucket for s in result.segments} == {"peak_pm", "off_peak"}


def test_segments_below_the_row_floor_are_dropped() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    n = 1_200
    actual = np.full(n, 600.0)
    promised = np.full(n, 500.0)
    segments = pl.DataFrame({"seg_time": ["peak_pm"] * 1_100 + ["off_peak"] * 100})
    result = evaluate("m", 0, actual, promised, cfg, segments)
    assert {s.bucket for s in result.segments} == {"peak_pm"}


def test_summary_reports_mean_and_std_across_seeds() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample(2_000)
    results = [evaluate("m", seed, actual, actual * (1.0 + 0.01 * seed), cfg) for seed in (0, 1, 2)]
    board = summarise_seeds(results)
    row = board.to_frame().row(0, named=True)
    assert row["seeds"] == 3
    assert row["business_cost_std"] > 0
    assert "business cost" in board.markdown()


def test_leaderboard_expresses_the_delta_against_a_named_baseline() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample(2_000)
    good = [evaluate("good", s, actual, actual * 1.02, cfg) for s in (0, 1, 2)]
    poor = [evaluate("poor", s, actual, actual * 0.7, cfg) for s in (0, 1, 2)]
    board = summarise_seeds([*good, *poor])
    text = board.markdown(baseline="poor")
    assert "vs baseline" in text
    assert "-" in text


# ------------------------------------------------- population enforcement ---
def _digest_frame(n: int, offset: int = 0) -> pl.DataFrame:
    import datetime as dt

    return pl.DataFrame(
        {
            "request_datetime": [
                dt.datetime(2023, 6, 1) + dt.timedelta(minutes=i + offset) for i in range(n)
            ],
            "pu_zone": pl.Series([1 + i % 20 for i in range(n)], dtype=pl.UInt16),
            "do_zone": pl.Series([1 + (i * 7) % 20 for i in range(n)], dtype=pl.UInt16),
        }
    )


def test_population_digest_is_stable_and_order_independent() -> None:
    from eta.models.dataset import population_digest

    frame = _digest_frame(500)
    assert population_digest(frame) == population_digest(frame)
    assert population_digest(frame) == population_digest(frame.reverse())


def test_population_digest_changes_with_the_population() -> None:
    from eta.models.dataset import population_digest

    assert population_digest(_digest_frame(500)) != population_digest(_digest_frame(501))
    assert population_digest(_digest_frame(500)) != population_digest(_digest_frame(500, offset=1))


def test_comparing_models_across_populations_raises() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample(1_000)
    a = [evaluate("a", s, actual, actual * 1.05, cfg, population="aaaa") for s in (0, 1, 2)]
    b = [evaluate("b", s, actual, actual * 1.10, cfg, population="bbbb") for s in (0, 1, 2)]
    with pytest.raises(ValueError, match="different populations"):
        summarise_seeds([*a, *b])


def test_same_population_compares_fine() -> None:
    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    actual = _sample(1_000)
    a = [evaluate("a", s, actual, actual * 1.05, cfg, population="same") for s in (0, 1, 2)]
    b = [evaluate("b", s, actual, actual * 1.10, cfg, population="same") for s in (0, 1, 2)]
    board = summarise_seeds([*a, *b])
    assert board.to_frame().height == 2
