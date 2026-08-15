from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from eta.calibration.coverage import coverage, coverage_report, interval_coverage
from eta.calibration.cqr import (
    MondrianCQR,
    SplitCQR,
    conformal_quantile,
    conformity_scores,
)
from eta.calibration.isotonic import IsotonicCalibrator

RNG = np.random.default_rng(0)
ALPHAS = (0.05, 0.5, 0.75, 0.9, 0.95)


# ------------------------------------------------------------- coverage -----
def test_coverage_is_the_fraction_at_or_below() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(actual, np.array([1.0, 2.0, 3.0, 4.0])) == 1.0
    assert coverage(actual, np.array([0.0, 0.0, 0.0, 0.0])) == 0.0
    assert coverage(actual, np.array([1.0, 2.0, 0.0, 0.0])) == pytest.approx(0.5)


def test_a_perfect_quantile_covers_its_own_level() -> None:
    actual = RNG.normal(600.0, 100.0, 40_000)
    for q in (0.25, 0.5, 0.9):
        predicted = np.full(actual.size, float(np.quantile(actual, q)))
        assert coverage(actual, predicted) == pytest.approx(q, abs=0.01)


def _segments(n: int) -> pl.DataFrame:
    return pl.DataFrame({"seg_time": ["peak_pm"] * (n // 2) + ["off_peak"] * (n - n // 2)})


def test_the_aggregate_can_look_fine_while_a_segment_is_broken() -> None:
    """The exact failure Phase 7 exists to expose: marginal coverage hides it."""
    n = 20_000
    actual = np.concatenate([RNG.normal(900.0, 60.0, n // 2), RNG.normal(300.0, 60.0, n // 2)])
    # One flat prediction: over-covers the cheap half, under-covers the expensive half.
    flat = np.full(n, 600.0)
    matrix = np.column_stack([flat] * len(ALPHAS))

    rep = coverage_report(actual, matrix, ALPHAS, _segments(n))
    aggregate = rep.aggregate_gap_pp(0.5)
    worst = rep.worst_gap_pp(0.5)

    assert aggregate < 10.0, "the marginal number looks unremarkable"
    assert worst > 40.0, "while a segment is catastrophically off"


def test_worst_is_reported_per_nominal_level() -> None:
    n = 6_000
    actual = RNG.normal(600.0, 100.0, n)
    matrix = np.column_stack([np.full(n, float(np.quantile(actual, q))) for q in ALPHAS])
    rep = coverage_report(actual, matrix, ALPHAS, _segments(n))
    assert rep.worst(nominal=0.75).nominal == pytest.approx(0.75)


# ----------------------------------------------------------------- CQR ------
def test_conformity_score_is_negative_inside_the_interval() -> None:
    actual = np.array([50.0, 150.0, 100.0])
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0])
    scores = conformity_scores(actual, lo, hi)
    assert scores[0] < 0, "well inside"
    assert scores[1] == pytest.approx(50.0), "50 above the top"
    assert scores[2] == pytest.approx(0.0), "exactly on the boundary"


def test_conformal_quantile_uses_the_finite_sample_correction() -> None:
    scores = np.arange(1.0, 101.0, dtype=np.float64)  # n = 100
    # ceil((100+1) * 0.9) = 91 -> the 91st order statistic, not the 90th.
    assert conformal_quantile(scores, 0.1) == pytest.approx(91.0)


def test_conformal_quantile_widens_when_it_cannot_certify() -> None:
    """With n too small for the level, return the max rather than a false guarantee."""
    scores = np.arange(1.0, 6.0, dtype=np.float64)  # n = 5, ceil(6 * 0.99) = 6 > 5
    assert conformal_quantile(scores, 0.01) == pytest.approx(5.0)


@pytest.mark.parametrize("target", [0.80, 0.90, 0.95])
def test_cqr_attains_its_guarantee_on_unseen_data(target: float) -> None:
    """The point of CQR: coverage holds even when the quantile model is bad."""
    n = 20_000
    cal_y = RNG.lognormal(6.5, 0.6, n)
    test_y = RNG.lognormal(6.5, 0.6, n)
    # Deliberately terrible, far too narrow.
    cal_lo, cal_hi = np.full(n, 600.0), np.full(n, 700.0)
    test_lo, test_hi = np.full(n, 600.0), np.full(n, 700.0)

    cqr = SplitCQR(alpha=1.0 - target).fit(cal_y, cal_lo, cal_hi)
    band = cqr.apply(test_lo, test_hi)
    got = interval_coverage(test_y, band.lo, band.hi)

    assert got >= target - 0.01, f"guarantee broken: {got:.4f} < {target}"


def test_cqr_pays_for_a_bad_model_in_width_not_coverage() -> None:
    n = 20_000
    cal_y, test_y = RNG.normal(600.0, 100.0, n), RNG.normal(600.0, 100.0, n)

    tight = SplitCQR(alpha=0.1).fit(cal_y, np.full(n, 590.0), np.full(n, 610.0))
    good_lo = np.full(n, float(np.quantile(cal_y, 0.05)))
    good_hi = np.full(n, float(np.quantile(cal_y, 0.95)))
    good = SplitCQR(alpha=0.1).fit(cal_y, good_lo, good_hi)

    bad_band = tight.apply(np.full(n, 590.0), np.full(n, 610.0))
    good_band = good.apply(good_lo, good_hi)

    assert interval_coverage(test_y, bad_band.lo, bad_band.hi) >= 0.89
    assert interval_coverage(test_y, good_band.lo, good_band.hi) >= 0.89
    assert bad_band.width.mean() > good_band.width.mean(), (
        "the bad model must be the wider one -- coverage alone proves nothing"
    )


def test_mondrian_beats_marginal_on_the_worst_segment() -> None:
    """Marginal CQR can hit 90% overall while a segment sits far off. Mondrian is the fix."""
    n = 40_000
    half = n // 2
    seg = pl.DataFrame({"seg_time": ["peak_pm"] * half + ["off_peak"] * half})

    # Heteroscedastic by segment: one is four times as noisy as the other.
    def draw() -> np.ndarray:
        return np.concatenate([RNG.normal(600.0, 240.0, half), RNG.normal(600.0, 60.0, half)])

    cal_y, test_y = draw(), draw()
    lo, hi = np.full(n, 540.0), np.full(n, 660.0)

    marginal = SplitCQR(alpha=0.1).fit(cal_y, lo, hi).apply(lo, hi)
    mondrian = (
        MondrianCQR(alpha=0.1, axes=("seg_time",), floor=1_000)
        .fit(cal_y, lo, hi, seg)
        .apply(lo, hi, seg)
    )

    peak = np.zeros(n, dtype=bool)
    peak[:half] = True

    marg_worst = abs(interval_coverage(test_y[peak], marginal.lo[peak], marginal.hi[peak]) - 0.9)
    mond_worst = abs(interval_coverage(test_y[peak], mondrian.lo[peak], mondrian.hi[peak]) - 0.9)
    assert mond_worst < marg_worst, "segment-conditional must beat marginal where it matters"


def test_mondrian_keeps_its_marginal_guarantee_too() -> None:
    n = 30_000
    half = n // 2
    seg = pl.DataFrame({"seg_time": ["peak_pm"] * half + ["off_peak"] * half})
    cal_y = np.concatenate([RNG.normal(600, 200, half), RNG.normal(600, 50, half)])
    test_y = np.concatenate([RNG.normal(600, 200, half), RNG.normal(600, 50, half)])
    lo, hi = np.full(n, 550.0), np.full(n, 650.0)

    band = (
        MondrianCQR(alpha=0.1, axes=("seg_time",), floor=1_000)
        .fit(cal_y, lo, hi, seg)
        .apply(lo, hi, seg)
    )
    assert interval_coverage(test_y, band.lo, band.hi) >= 0.89


# ------------------------------------------------------------- isotonic -----
def _iso_frames(n: int) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    half = n // 2
    seg = pl.DataFrame({"seg_time": ["peak_pm"] * half + ["off_peak"] * half})
    actual = np.concatenate([RNG.normal(900.0, 90.0, half), RNG.normal(400.0, 90.0, half)])
    matrix = np.column_stack([np.full(n, 600.0 + 40.0 * i) for i in range(len(ALPHAS))])
    return actual, matrix, seg


def test_isotonic_closes_a_segment_gap_it_was_fitted_on() -> None:
    n = 40_000
    cal_actual, cal_matrix, cal_seg = _iso_frames(n)
    test_actual, test_matrix, test_seg = _iso_frames(n)

    before = coverage_report(test_actual, test_matrix, ALPHAS, test_seg).worst_gap_pp(0.75)
    iso = IsotonicCalibrator(alphas=ALPHAS, axis="seg_time", min_rows=1_000).fit(
        cal_matrix, cal_actual, cal_seg
    )
    after_matrix = iso.apply(test_matrix, test_seg)
    after = coverage_report(test_actual, after_matrix, ALPHAS, test_seg).worst_gap_pp(0.75)

    assert after < before, f"worst-segment gap did not improve: {before:.1f} -> {after:.1f}"
    assert after < 10.0


def test_isotonic_output_cannot_cross() -> None:
    n = 20_000
    cal_actual, cal_matrix, cal_seg = _iso_frames(n)
    iso = IsotonicCalibrator(alphas=ALPHAS, axis="seg_time", min_rows=1_000).fit(
        cal_matrix, cal_actual, cal_seg
    )
    out = iso.apply(cal_matrix, cal_seg)
    assert np.all(np.diff(out, axis=1) >= 0.0), "recalibration re-introduced crossing"


def test_thin_segments_fall_back_to_the_global_map_and_say_so() -> None:
    n = 12_000
    actual = RNG.normal(600.0, 100.0, n)
    matrix = np.column_stack([np.full(n, 500.0 + 40.0 * i) for i in range(len(ALPHAS))])
    # 'rare' is far below the floor and must not get its own map.
    seg = pl.DataFrame({"seg_time": ["common"] * (n - 300) + ["rare"] * 300})

    iso = IsotonicCalibrator(alphas=ALPHAS, axis="seg_time", min_rows=5_000).fit(
        matrix, actual, seg
    )
    assert "rare" in iso.fallback_buckets
    assert "common" not in iso.fallback_buckets


def test_interval_coverage_is_scored_per_segment_not_via_the_columns() -> None:
    """CQR moves the band and leaves the quantile columns untouched.

    Scoring a conformalised approach on its columns would report marginal and
    Mondrian CQR as identical, which is exactly the comparison Phase 7D exists to
    make. This asserts the interval scorer actually sees the band.
    """
    from eta.calibration.coverage import interval_coverage_by_segment, worst_interval_gap

    n = 20_000
    half = n // 2
    seg = pl.DataFrame({"seg_time": ["peak_pm"] * half + ["off_peak"] * half})
    actual = np.concatenate([RNG.normal(600.0, 300.0, half), RNG.normal(600.0, 30.0, half)])
    lo, hi = np.full(n, 550.0), np.full(n, 650.0)

    points = interval_coverage_by_segment(actual, lo, hi, seg, target=0.9)
    by_bucket = {p.bucket: p for p in points}

    assert by_bucket["off_peak"].empirical > by_bucket["peak_pm"].empirical
    assert worst_interval_gap(points).bucket == "peak_pm"

    # Widening only the noisy half must move its number and nothing else.
    wide_lo, wide_hi = lo.copy(), hi.copy()
    wide_lo[:half] -= 1_000.0
    wide_hi[:half] += 1_000.0
    after = {p.bucket: p for p in interval_coverage_by_segment(actual, wide_lo, wide_hi, seg, 0.9)}
    assert after["peak_pm"].empirical > by_bucket["peak_pm"].empirical
    assert after["off_peak"].empirical == pytest.approx(by_bucket["off_peak"].empirical)
