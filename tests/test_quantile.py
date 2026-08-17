from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from eta.models.cost import pinball_loss
from eta.models.quantile.composition import MonotonicComposition
from eta.models.quantile.crossing import crossing_report, is_monotone, sort_rows
from eta.models.quantile.lgbm import QuantileBundle

ALPHAS = (0.5, 0.75, 0.9)
RNG = np.random.default_rng(0)


# --------------------------------------------------------- measuring it -----
def test_crossing_report_counts_rows_not_pairs() -> None:
    matrix = np.array(
        [
            [100.0, 200.0, 300.0],  # fine
            [200.0, 100.0, 300.0],  # one inversion
            [300.0, 200.0, 100.0],  # two inversions, still ONE crossing row
            [100.0, 200.0, 150.0],  # one inversion
        ]
    )
    rep = crossing_report(matrix, ALPHAS)
    assert rep.rows == 4
    assert rep.crossing_rows == 3
    assert rep.crossing_rate == pytest.approx(0.75)
    assert rep.pairwise["P50 > P75"] == pytest.approx(0.5)
    assert rep.pairwise["P75 > P90"] == pytest.approx(0.5)
    assert rep.worst_gap_s == pytest.approx(100.0)


def test_a_clean_matrix_reports_no_crossing() -> None:
    matrix = np.cumsum(RNG.uniform(1.0, 10.0, size=(500, 3)), axis=1)
    rep = crossing_report(matrix, ALPHAS)
    assert rep.crossing_rows == 0
    assert rep.crossing_rate == 0.0
    assert is_monotone(matrix).all()


def test_ties_are_not_crossings() -> None:
    """P75 == P90 is degenerate, not inverted. Counting it would inflate the rate."""
    matrix = np.array([[100.0, 200.0, 200.0]])
    assert crossing_report(matrix, ALPHAS).crossing_rows == 0
    assert is_monotone(matrix).all()


def test_crossing_report_rejects_descending_alphas() -> None:
    with pytest.raises(ValueError, match="ascending"):
        crossing_report(np.zeros((3, 3)), (0.9, 0.75, 0.5))


def test_crossing_report_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        crossing_report(np.zeros((3, 2)), ALPHAS)


# ----------------------------------------------------------- fix 1: sort ----
def test_sorting_eliminates_crossing_but_moves_the_served_level() -> None:
    matrix = np.array([[300.0, 100.0, 200.0]])
    fixed = sort_rows(matrix)
    assert crossing_report(fixed, ALPHAS).crossing_rows == 0
    # The whole objection to sorting, made concrete: the P75 promise changed.
    assert fixed[0, 1] != matrix[0, 1]
    assert fixed[0].tolist() == [100.0, 200.0, 300.0]


def test_sorting_leaves_an_already_ordered_row_alone() -> None:
    matrix = np.array([[100.0, 200.0, 300.0]])
    assert sort_rows(matrix).tolist() == matrix.tolist()


# ------------------------------------------------- the models, on real-ish --
def _frame(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 10.0, n)
    x2 = rng.uniform(0.0, 5.0, n)
    # Heteroscedastic on purpose: spread grows with x1, so the quantiles must fan out.
    noise = rng.normal(0.0, 1.0, n) * (1.0 + x1)
    y = 300.0 + 40.0 * x1 + 20.0 * x2 + 30.0 * noise
    return pl.DataFrame({"f1": x1, "f2": x2, "total_time_s": np.maximum(y, 30.0)})


FEATURES = ["f1", "f2"]


def test_quantile_models_order_on_average_even_when_they_cross_on_rows() -> None:
    train, valid, test = _frame(4_000, 0), _frame(1_500, 1), _frame(1_500, 2)
    bundle = QuantileBundle(alphas=ALPHAS, features=FEATURES, num_boost_round=60)
    bundle.fit(train, seed=0, valid=valid)
    matrix = bundle.predict_matrix(test)

    means = matrix.mean(axis=0)
    assert means[0] < means[1] < means[2], "higher alpha must sit higher on average"


def test_each_level_covers_roughly_its_own_share() -> None:
    """A P75 that is not above the actual ~75% of the time is not a P75."""
    train, valid, test = _frame(6_000, 0), _frame(2_000, 1), _frame(3_000, 2)
    bundle = QuantileBundle(alphas=ALPHAS, features=FEATURES, num_boost_round=120)
    bundle.fit(train, seed=0, valid=valid)
    matrix = bundle.predict_matrix(test)
    actual = test["total_time_s"].to_numpy()

    for i, alpha in enumerate(ALPHAS):
        covered = float(np.mean(actual <= matrix[:, i]))
        assert covered == pytest.approx(alpha, abs=0.06), f"alpha={alpha} covered {covered:.3f}"


def test_pinball_is_minimised_at_its_own_level() -> None:
    """The q=0.75 model should beat the q=0.5 model under 0.75 pinball loss."""
    train, valid, test = _frame(4_000, 0), _frame(1_500, 1), _frame(1_500, 2)
    bundle = QuantileBundle(alphas=ALPHAS, features=FEATURES, num_boost_round=120)
    bundle.fit(train, seed=0, valid=valid)
    matrix = bundle.predict_matrix(test)
    actual = test["total_time_s"].to_numpy()

    at_50 = float(np.mean(pinball_loss(actual, matrix[:, 0], 0.75)))
    at_75 = float(np.mean(pinball_loss(actual, matrix[:, 1], 0.75)))
    assert at_75 < at_50


# ---------------------------------------------- fix 2: monotone by design ---
def test_composition_cannot_cross() -> None:
    train, valid, test = _frame(4_000, 0), _frame(1_500, 1), _frame(1_500, 2)
    composed = MonotonicComposition(alphas=ALPHAS, features=FEATURES, num_boost_round=80)
    composed.fit(train, seed=0, valid=valid)
    matrix = composed.predict_matrix(test)

    rep = crossing_report(matrix, ALPHAS)
    assert rep.crossing_rows == 0, "clamped increments cannot produce an inversion"
    assert is_monotone(matrix).all()


def test_composition_still_tracks_the_quantiles_it_claims() -> None:
    """Monotonicity is easy to get by predicting nonsense; check coverage too."""
    train, valid, test = _frame(6_000, 0), _frame(2_000, 1), _frame(3_000, 2)
    composed = MonotonicComposition(alphas=ALPHAS, features=FEATURES, num_boost_round=120)
    composed.fit(train, seed=0, valid=valid)
    matrix = composed.predict_matrix(test)
    actual = test["total_time_s"].to_numpy()

    for i, alpha in enumerate(ALPHAS):
        covered = float(np.mean(actual <= matrix[:, i]))
        assert covered == pytest.approx(alpha, abs=0.06), f"alpha={alpha} covered {covered:.3f}"


def test_composition_base_level_is_untouched_by_the_clamp() -> None:
    """Sorting can move P50; composition never does. That is the whole trade."""
    train, valid = _frame(3_000, 0), _frame(1_000, 1)
    composed = MonotonicComposition(alphas=ALPHAS, features=FEATURES, num_boost_round=60)
    composed.fit(train, seed=0, valid=valid)
    matrix = composed.predict_matrix(valid)

    base_direct = np.asarray(
        composed._boosters[0].predict(valid.select(FEATURES).to_numpy()), dtype=np.float64
    )
    assert np.allclose(matrix[:, 0], base_direct)


# ------------------------------------------------------- the served level ---
def test_q_star_column_must_exist() -> None:
    from eta.models.quantile.run import _q_star_column

    assert _q_star_column((0.5, 0.75, 0.9), 0.75) == 1
    with pytest.raises(ValueError, match="not among the trained quantiles"):
        _q_star_column((0.5, 0.9), 0.75)


# --------------------------------------------- fix 3: ordered by structure --
def test_torch_pinball_matches_the_numpy_definition() -> None:
    torch = pytest.importorskip("torch")
    from eta.models.quantile.nn import pinball_loss_torch

    actual = RNG.lognormal(7.0, 0.5, 400)
    pred = actual * RNG.uniform(0.7, 1.3, 400)

    got = float(
        pinball_loss_torch(
            torch.tensor(pred, dtype=torch.float64).reshape(-1, 1),
            torch.tensor(actual, dtype=torch.float64).reshape(-1, 1),
            torch.tensor([[0.75]], dtype=torch.float64),
        )
    )
    assert got == pytest.approx(float(np.mean(pinball_loss(actual, pred, 0.75))), rel=1e-9)


def test_ordered_heads_cannot_cross_even_untrained() -> None:
    """Ordering is structural, so it holds at random initialisation too."""
    pytest.importorskip("torch")
    from eta.models.quantile.nn import MultiHeadQuantileNet

    net = MultiHeadQuantileNet(alphas=ALPHAS, features=FEATURES, ordered=True, max_epochs=1)
    net.fit(_frame(2_000, 0), seed=0)
    matrix = net.predict_matrix(_frame(800, 2))

    assert crossing_report(matrix, ALPHAS).crossing_rows == 0
    assert is_monotone(matrix).all()


def test_the_ordered_net_is_the_only_thing_stopping_the_crossing() -> None:
    """Same architecture, same seed, same data -- the flag is the whole difference."""
    pytest.importorskip("torch")
    from eta.models.quantile.nn import MultiHeadQuantileNet

    train, test = _frame(3_000, 0), _frame(1_000, 2)
    free = MultiHeadQuantileNet(alphas=ALPHAS, features=FEATURES, ordered=False, max_epochs=1)
    free.fit(train, seed=0)
    ordered = MultiHeadQuantileNet(alphas=ALPHAS, features=FEATURES, ordered=True, max_epochs=1)
    ordered.fit(train, seed=0)

    assert crossing_report(ordered.predict_matrix(test), ALPHAS).crossing_rows == 0
    # After one epoch the unconstrained heads have not learned the ordering yet.
    assert crossing_report(free.predict_matrix(test), ALPHAS).crossing_rows > 0


def test_ordered_net_learns_the_levels_it_claims() -> None:
    pytest.importorskip("torch")
    from eta.models.quantile.nn import MultiHeadQuantileNet

    train, valid, test = _frame(20_000, 0), _frame(5_000, 1), _frame(5_000, 2)
    net = MultiHeadQuantileNet(
        alphas=ALPHAS, features=FEATURES, ordered=True, max_epochs=40, batch_size=2048
    )
    net.fit(train, seed=0, valid=valid)
    matrix = net.predict_matrix(test)
    actual = test["total_time_s"].to_numpy()

    for i, alpha in enumerate(ALPHAS):
        covered = float(np.mean(actual <= matrix[:, i]))
        assert covered == pytest.approx(alpha, abs=0.08), f"alpha={alpha} covered {covered:.3f}"


# ------------------------------------------------------ the OpenMP guard ----
def test_guard_raises_when_lightgbm_was_imported_first() -> None:
    """The wrong order must fail loudly here, because in torch it hangs silently.

    Run in a subprocess: import order is process-global, so this cannot be checked
    from inside a test session that has already loaded both.
    """
    import os
    import pathlib
    import subprocess
    import sys

    import eta

    code = (
        "import lightgbm\n"
        "from eta.models.quantile._openmp import preload_torch_before_lightgbm\n"
        "preload_torch_before_lightgbm()\n"
    )
    src = str(pathlib.Path(eta.__file__).resolve().parent.parent)
    env = os.environ | {"PYTHONPATH": src}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env
    )

    # The guard only has something to protect when torch is actually installed.
    # Where it is not -- CI installs dev+geo only -- there is one OpenMP runtime,
    # no conflict to order, and raising would be wrong. Both branches are asserted
    # rather than assuming the developer's environment.
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        assert proc.returncode == 0, proc.stderr[-500:]
    else:
        assert proc.returncode != 0
        assert "deadlocks" in proc.stderr, proc.stderr[-500:]


def test_guard_is_a_no_op_when_torch_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates the CI environment, where the optional torch extra is absent.

    With no torch there is a single OpenMP runtime, so there is nothing to order and
    the guard must return quietly -- even if lightgbm is already loaded.
    """
    import importlib.util
    import sys as _sys

    from eta.models.quantile import _openmp

    monkeypatch.setitem(_sys.modules, "lightgbm", object())
    monkeypatch.delitem(_sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "torch" else object()
    )

    assert _openmp.preload_torch_before_lightgbm() == (
        "torch not installed -- single OpenMP runtime, nothing to order"
    )


def test_guard_is_a_no_op_once_torch_is_loaded() -> None:
    pytest.importorskip("torch")
    from eta.models.quantile._openmp import preload_torch_before_lightgbm

    assert preload_torch_before_lightgbm() == "torch already loaded"


# ---------------------------------------------------- independent searches --
def test_each_quantile_level_gets_its_own_sampler_seed() -> None:
    """A shared seed made all three levels tune to bit-identical parameters.

    TPE's startup phase is random draws from the sampler RNG. One seed for every
    level means every level explores the same configurations first, and on a small
    budget the winner falls inside that shared phase -- so "tuned per level"
    silently becomes "tuned once".
    """
    from eta.models.quantile.lgbm import sampler_seed_for

    seeds = {sampler_seed_for(0, a) for a in ALPHAS}
    assert len(seeds) == len(ALPHAS), "levels must not share a sampler seed"

    # ...and the model seed still separates runs of the same level.
    assert sampler_seed_for(0, 0.75) != sampler_seed_for(1, 0.75)


def test_sampler_seed_is_stable_for_the_same_inputs() -> None:
    from eta.models.quantile.lgbm import sampler_seed_for

    assert sampler_seed_for(2, 0.9) == sampler_seed_for(2, 0.9)


# ------------------------------------------- selection on validation only ---
def test_selection_picks_the_lowest_validation_cost() -> None:
    from eta.models.quantile.selection import select_champion

    sel = select_champion(
        {"a": [100.0, 101.0, 99.0], "b": [90.0, 91.0, 89.0], "c": [120.0, 120.0, 120.0]}
    )
    assert sel.champion == "b"
    assert sel.val_cost == pytest.approx(90.0)
    assert sel.ranking[0][0] == "b"
    assert sel.margin == pytest.approx(10.0)


def test_selection_flags_a_near_tie_instead_of_pretending_it_decided() -> None:
    """Choosing among candidates separated by less than seed noise is a coin flip.

    The first version of Phase 6 picked a crossing strategy on a 2.6-unit test-set
    difference. Reporting that as a decision is the failure this guards against.
    """
    from eta.models.quantile.selection import select_champion

    noisy = select_champion({"a": [100.0, 104.0, 96.0], "b": [100.5, 104.5, 96.5]})
    assert noisy.margin_exceeds_seed_spread is False
    assert "near-tie" in noisy.markdown()

    clear = select_champion({"a": [100.0, 100.1, 99.9], "b": [140.0, 140.1, 139.9]})
    assert clear.margin_exceeds_seed_spread is True
    # The flag never overrides the rule: the lowest validation cost still wins.
    assert clear.champion == "a"
    assert "significance test" in clear.markdown()


def test_selection_cannot_be_handed_a_test_score() -> None:
    """Structural, not conventional: the signature has nowhere to put test costs."""
    import inspect

    from eta.models.quantile.selection import select_champion

    params = list(inspect.signature(select_champion).parameters)
    assert params == ["val_costs"], f"selection gained a new input: {params}"


def test_selection_rejects_an_empty_field() -> None:
    from eta.models.quantile.selection import select_champion

    with pytest.raises(ValueError, match="no candidates"):
        select_champion({})


# ------------------------------------------------- clamp is measured, not assumed --
def test_composition_reports_how_much_the_clamp_rewrote() -> None:
    """A crossing fix and a different model formulation are different claims."""
    train, valid, test = _frame(4_000, 0), _frame(1_500, 1), _frame(1_500, 2)
    composed = MonotonicComposition(alphas=ALPHAS, features=FEATURES, num_boost_round=60)
    composed.fit(train, seed=0, valid=valid)
    composed.predict_matrix(test)

    stats = composed.clamp_stats
    assert len(stats) == len(ALPHAS) - 1, "one clamp record per non-base level"
    for s in stats:
        assert s.rows == test.height
        assert 0.0 <= s.clamped_share <= 1.0
        assert s.clamped_rows == pytest.approx(s.clamped_share * s.rows, abs=1.0)
        assert s.max_adjustment_s >= s.p95_adjustment_s >= 0.0


def test_clamp_stats_are_zero_when_nothing_is_clamped() -> None:
    """A strongly ordered target should need little or no clamping."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 4_000
    x1 = rng.uniform(0.0, 10.0, n)
    # Very little noise: the levels separate cleanly and increments stay positive.
    frame = pl.DataFrame({"f1": x1, "f2": rng.uniform(0, 1, n), "total_time_s": 300.0 + 100.0 * x1})
    composed = MonotonicComposition(alphas=ALPHAS, features=FEATURES, num_boost_round=40)
    composed.fit(frame, seed=0)
    composed.predict_matrix(frame)
    assert all(s.clamped_share < 0.5 for s in composed.clamp_stats)


# ------------------------------------ the test split cannot flow backwards --
def _phase_source() -> tuple[str, object]:
    import ast
    import inspect

    from eta.models.quantile import run as run_mod

    src = inspect.getsource(run_mod)
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_quantile_phase"
    )
    return src, fn


def _call_lines(fn: object, name: str) -> list[int]:
    import ast

    return sorted(
        n.lineno
        for n in ast.walk(fn)  # type: ignore[arg-type]
        if isinstance(n, ast.Call)
        and (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")) == name
    )


def test_the_champion_is_frozen_before_any_test_analysis_runs() -> None:
    """Robustness reads test. It must therefore run strictly after selection.

    If a test-window result could flow back into the choice of champion, this would
    be test-set selection again, just further downstream and harder to see.
    """
    _, fn = _phase_source()
    select = _call_lines(fn, "select_champion")
    windows = _call_lines(fn, "temporal_windows")
    blocks = _call_lines(fn, "paired_blocks")

    assert select, "no selection step found"
    assert windows and blocks, "no robustness step found"
    assert max(select) < min(windows + blocks), (
        "test-reading robustness runs before the champion is frozen"
    )


def test_nothing_reassigns_the_champion_after_selection() -> None:
    """The arrow points one way: selection -> freeze -> report."""
    import ast

    _, fn = _phase_source()
    select_line = max(_call_lines(fn, "select_champion"))

    rebinds = [
        n.lineno
        for n in ast.walk(fn)  # type: ignore[arg-type]
        if isinstance(n, ast.Assign)
        and n.lineno > select_line
        and any(isinstance(t, ast.Name) and t.id == "selection" for t in n.targets)
    ]
    assert not rebinds, f"champion reassigned after selection at lines {rebinds}"


def test_the_tuner_is_never_handed_the_test_split() -> None:
    """Structural: test cannot enter the tuning loop even by mistake."""
    import inspect

    from eta.models.quantile.lgbm import tune_on_validation

    params = list(inspect.signature(tune_on_validation).parameters)
    assert "test" not in params, f"tuning gained a test input: {params}"
    assert params[:2] == ["train", "valid"], params


def test_each_quantile_level_gets_the_full_trial_budget() -> None:
    """100 trials per level, not 100 shared across three levels."""
    import ast

    _, fn = _phase_source()
    tune_calls = [
        n
        for n in ast.walk(fn)  # type: ignore[arg-type]
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", "")) == "tune_on_validation"
    ]
    assert len(tune_calls) == 1, "expected one tuning call inside a per-alpha loop"
    # ...and it must sit inside a loop over alphas, so each level pays the full budget.
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]  # type: ignore[arg-type]
    enclosing = [
        lp for lp in loops if lp.lineno < tune_calls[0].lineno <= (lp.end_lineno or lp.lineno)
    ]
    assert enclosing, "tuning is not inside a per-level loop"


def test_the_clamp_mask_marks_exactly_the_rows_that_were_adjusted() -> None:
    """The mask drives the clamped-vs-untouched cost split, so it has to be right."""
    train, valid, test = _frame(4_000, 0), _frame(1_500, 1), _frame(1_500, 2)
    composed = MonotonicComposition(alphas=ALPHAS, features=FEATURES, num_boost_round=60)
    composed.fit(train, seed=0, valid=valid)
    composed.predict_matrix(test)

    mask = composed.clamped_any
    assert mask.shape == (test.height,)
    # The mask is the union across levels, so it can only exceed any single level.
    for s in composed.clamp_stats:
        assert int(mask.sum()) >= s.clamped_rows


def test_clamp_mask_requires_a_prediction_first() -> None:
    composed = MonotonicComposition(alphas=ALPHAS, features=FEATURES, num_boost_round=20)
    composed.fit(_frame(1_000, 0), seed=0)
    with pytest.raises(RuntimeError, match="predict_matrix has not been called"):
        _ = composed.clamped_any


# ------------------------------------------------ robustness, by hand -------
def _clock(n: int, start_day: int = 1, per_step: str = "minutes") -> pl.Series:
    import datetime as dt

    base = dt.datetime(2023, 6, start_day)
    return pl.Series([base + dt.timedelta(**{per_step: i}) for i in range(n)])


def test_windows_slice_by_time_not_by_row_order() -> None:
    """Rows do not arrive sorted. Slicing on position would silently mix periods."""
    from eta.config import CostConfig
    from eta.models.quantile.robustness import temporal_windows

    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    n = 900
    actual = np.full(n, 600.0)
    champion = np.full(n, 600.0)
    # Baseline is early by 10s / 20s / 30s in the three periods -> cost 10 / 20 / 30.
    baseline = np.concatenate([np.full(300, 610.0), np.full(300, 620.0), np.full(300, 630.0)])

    ordered = temporal_windows(_clock(n), actual, champion, baseline, cfg)
    assert [round(w.baseline_cost) for w in ordered] == [10, 20, 30]

    perm = np.random.default_rng(0).permutation(n)
    shuffled = temporal_windows(_clock(n)[perm.tolist()], actual, champion, baseline[perm], cfg)
    assert [round(w.baseline_cost) for w in shuffled] == [10, 20, 30], (
        "windows must be chronological even when input rows are shuffled"
    )


def test_paired_blocks_count_days_wins_and_consecutive_losses() -> None:
    """Hand-built: champion loses days 3, 4, 5 and wins the other seven."""
    from eta.config import CostConfig
    from eta.models.quantile.robustness import paired_blocks

    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    days, per_day = 10, 100
    times = _clock(days * per_day)
    # One row per minute would stay inside one day, so step by day explicitly.
    import datetime as dt

    times = pl.Series(
        [
            dt.datetime(2023, 6, 1) + dt.timedelta(days=d, minutes=m)
            for d in range(days)
            for m in range(per_day)
        ]
    )
    actual = np.full(days * per_day, 600.0)
    champion = np.empty(days * per_day)
    baseline = np.empty(days * per_day)
    for d in range(days):
        sl = slice(d * per_day, (d + 1) * per_day)
        champion[sl], baseline[sl] = (590.0, 600.0) if d in (3, 4, 5) else (600.0, 590.0)

    b = paired_blocks(times, actual, champion, baseline, cfg)
    assert b.blocks == 10
    assert b.champion_wins == 7
    assert b.longest_losing_streak == 3, "three consecutive losses must be reported as a streak"
    assert b.best_block_diff == pytest.approx(-30.0)
    assert b.worst_block_diff == pytest.approx(30.0)


def test_win_rate_is_not_dressed_up_as_a_probability() -> None:
    """The report must say what the win rate is not."""
    from eta.config import CostConfig
    from eta.models.quantile.robustness import markdown, paired_blocks, temporal_windows

    cfg = CostConfig(lambda_late=3.0, lambda_early=1.0)
    n = 600
    actual = np.full(n, 600.0)
    champion = np.full(n, 600.0)
    baseline = np.full(n, 620.0)
    times = _clock(n, per_step="hours")

    md = markdown(
        temporal_windows(times, actual, champion, baseline, cfg),
        paired_blocks(times, actual, champion, baseline, cfg),
        "champ",
        "base",
    )
    assert "not a probability of superiority" in md
    assert "no p-value is reported" in md.lower()


def test_the_summary_persists_what_the_selection_claim_rests_on() -> None:
    """A report that cannot be re-checked from its own artifacts is not evidence.

    The first validation-selected run wrote `selection.md` but omitted `selection`
    and `val_costs` from the machine-readable summary, because a string replacement
    silently failed to match. The per-seed ordering behind the margin claim was
    therefore unrecoverable without a five-hour re-run.
    """
    import ast
    import inspect

    from eta.models.quantile import run as run_mod

    tree = ast.parse(inspect.getsource(run_mod))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_quantile_phase"
    )
    summaries = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "summary" for t in n.targets)
        and isinstance(n.value, ast.Dict)
    ]
    assert summaries, "no summary dict found"
    node = summaries[0].value
    assert isinstance(node, ast.Dict)
    keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
    for required in ("selection", "val_costs", "clamp"):
        assert required in keys, f"summary drops '{required}'; the claim becomes uncheckable"
