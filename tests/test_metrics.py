import numpy as np
import pytest

from tlv.metrics import (
    bootstrap_ci,
    bootstrap_reduction_ci,
    censored_values,
    relative_reduction,
    time_to_target,
)


def test_time_to_target_first_crossing():
    times = [10.0, 20.0, 30.0, 40.0]
    accs = [0.2, 0.5, 0.8, 0.9]
    r = time_to_target(times, accs, 0.75)
    assert r.reached and r.seconds == 30.0 and r.epochs == 3


def test_time_to_target_uses_first_not_best():
    # accuracy touches the target at epoch 2, then drops: first crossing counts.
    r = time_to_target([1, 2, 3, 4], [0.1, 0.8, 0.5, 0.9], 0.8)
    assert r.reached and r.epochs == 2 and r.seconds == 2


def test_time_to_target_exact_equality_counts():
    assert time_to_target([5, 10], [0.5, 0.7], 0.7).epochs == 2


def test_time_to_target_never_reached():
    r = time_to_target([1, 2, 3], [0.1, 0.2, 0.3], 0.9)
    assert not r.reached and r.seconds is None and r.epochs is None


def test_time_to_target_validates_inputs():
    with pytest.raises(ValueError):
        time_to_target([1, 2], [0.1], 0.5)
    with pytest.raises(ValueError):
        time_to_target([2, 1], [0.1, 0.2], 0.5)  # clock must not go backwards
    with pytest.raises(ValueError):
        time_to_target([], [], 0.5)


def test_time_to_target_monotone_in_target():
    times = np.arange(1, 11, dtype=float)
    accs = np.linspace(0.1, 1.0, 10)
    ep = [time_to_target(times, accs, t).epochs for t in (0.2, 0.5, 0.8)]
    assert ep == sorted(ep)


def test_censored_values_use_budget_for_misses():
    runs = [
        time_to_target([1, 2, 3], [0.1, 0.6, 0.9], 0.5),  # reached at 2
        time_to_target([1, 2, 3], [0.1, 0.2, 0.3], 0.5),  # miss, total 3
    ]
    secs, eps, n_reached = censored_values(runs, budget_seconds=[3.0, 3.0], budget_epochs=3)
    assert list(secs) == [2.0, 3.0] and list(eps) == [2, 3] and n_reached == 1


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    x = np.array([10.0, 12.0, 11.0, 13.0, 9.0])
    a = bootstrap_ci(x, n_boot=2000, seed=1)
    b = bootstrap_ci(x, n_boot=2000, seed=1)
    assert a == b
    lo, hi = a
    assert lo <= x.mean() <= hi
    assert lo >= x.min() and hi <= x.max()


def test_bootstrap_ci_degenerate_constant():
    assert bootstrap_ci([4.0, 4.0, 4.0], n_boot=500, seed=0) == (4.0, 4.0)


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = rng.normal(10, 2, 5)
    large = rng.normal(10, 2, 200)

    def width(v):
        lo, hi = bootstrap_ci(v, n_boot=2000, seed=3)
        return hi - lo

    assert width(large) < width(small)


def test_relative_reduction():
    assert relative_reduction([10, 10], [7, 7]) == pytest.approx(0.3)
    assert relative_reduction([10], [12]) == pytest.approx(-0.2)  # slower = negative
    with pytest.raises(ValueError):
        relative_reduction([0, 0], [1, 1])


def test_bootstrap_reduction_ci():
    base = [100.0, 102.0, 98.0, 101.0, 99.0]
    fast = [50.0, 51.0, 49.0, 50.5, 49.5]
    est, lo, hi = bootstrap_reduction_ci(base, fast, n_boot=3000, seed=2)
    assert est == pytest.approx(0.5, abs=0.01)
    assert lo < est < hi and 0.45 < lo and hi < 0.55


def test_bootstrap_reduction_ci_flags_no_difference():
    base = [10.0, 14.0, 9.0, 13.0, 11.0]
    same = [11.0, 9.0, 14.0, 10.0, 13.0]
    _, lo, hi = bootstrap_reduction_ci(base, same, n_boot=3000, seed=2)
    assert lo < 0 < hi  # interval spans zero: not significant


def test_bootstrap_diff_ci_brackets_known_difference():
    from tlv.metrics import bootstrap_diff_ci

    rng = np.random.default_rng(0)
    a, b = rng.normal(0.8, 0.01, 10), rng.normal(0.6, 0.01, 10)
    est, lo, hi = bootstrap_diff_ci(a, b, n_boot=2000)
    assert lo < est < hi and 0.18 < lo and hi < 0.22
    assert bootstrap_diff_ci(a, a, n_boot=2000)[1] < 0 < bootstrap_diff_ci(a, a, n_boot=2000)[2]
    with pytest.raises(ValueError):
        bootstrap_diff_ci([], b)
