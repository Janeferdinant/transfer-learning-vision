"""Time-to-target metric and bootstrap statistics (no TensorFlow dependency)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TimeToTarget:
    """Result of asking: when did validation accuracy FIRST reach the target?

    seconds: cumulative wall-clock training time at the end of that epoch
    epochs:  1-based number of epochs completed at that point
    Both are None when the target was never reached within the run (censored).
    """

    reached: bool
    seconds: float | None
    epochs: int | None


def time_to_target(times: ArrayLike, accs: ArrayLike, target: float) -> TimeToTarget:
    """First epoch whose validation accuracy is >= target.

    ``times[i]`` is the cumulative wall-clock time at the end of epoch i+1 and
    ``accs[i]`` the validation accuracy measured at that point. Accuracy is only
    observed once per epoch, so the result has epoch granularity (no interpolation).
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(accs, dtype=float)
    if t.shape != a.shape or t.ndim != 1:
        raise ValueError("times and accs must be 1-D arrays of equal length")
    if t.size == 0:
        raise ValueError("empty curve")
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be non-decreasing")
    hits = np.flatnonzero(a >= target)
    if hits.size == 0:
        return TimeToTarget(False, None, None)
    i = int(hits[0])
    return TimeToTarget(True, float(t[i]), i + 1)


def censored_values(runs: Sequence[TimeToTarget], budget_seconds: Sequence[float], budget_epochs: int):
    """Turn per-run results into numbers, charging runs that never reached the
    target the full budget. This makes the time a LOWER BOUND for those runs, so
    savings computed against them are conservative when the baseline is censored
    and optimistic-free when the candidate is censored.

    Returns (seconds array, epochs array, number of runs that reached the target).
    """
    secs, eps, n_reached = [], [], 0
    for r, b in zip(runs, budget_seconds):
        if r.reached:
            secs.append(r.seconds)
            eps.append(r.epochs)
            n_reached += 1
        else:
            secs.append(float(b))
            eps.append(int(budget_epochs))
    return np.asarray(secs, dtype=float), np.asarray(eps, dtype=float), n_reached


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap interval for the mean (resampling runs/seeds)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def relative_reduction(baseline, other) -> float:
    """1 - mean(other)/mean(baseline). Positive means ``other`` is faster."""
    b = float(np.mean(baseline))
    if b <= 0:
        raise ValueError("baseline mean must be positive")
    return 1.0 - float(np.mean(other)) / b


def bootstrap_reduction_ci(baseline, other, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """(estimate, lo, hi) for the relative reduction, resampling each group's runs
    independently (seeds are not paired across approaches)."""
    b = np.asarray(baseline, dtype=float)
    o = np.asarray(other, dtype=float)
    rng = np.random.default_rng(seed)
    bm = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    om = o[rng.integers(0, o.size, size=(n_boot, o.size))].mean(axis=1)
    ok = bm > 0
    red = 1.0 - om[ok] / bm[ok]
    lo, hi = np.percentile(red, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return relative_reduction(b, o), float(lo), float(hi)


def bootstrap_diff_ci(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """(estimate, lo, hi) for mean(a) - mean(b), resampling each group independently."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    xm = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    ym = y[rng.integers(0, y.size, size=(n_boot, y.size))].mean(axis=1)
    lo, hi = np.percentile(xm - ym, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean() - y.mean()), float(lo), float(hi)
