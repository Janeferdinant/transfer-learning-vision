import pandas as pd
import pytest

from tlv.aggregate import summarise


def make_epochs():
    rows = []
    # scratch: slow, reaches 0.7 at epoch 4; frozen: reaches 0.7 at epoch 1
    for seed in range(5):
        for epoch in range(1, 5):
            rows.append(dict(approach="scratch", seed=seed, epoch=epoch,
                             seconds=epoch * 10.0 + seed * 0.1,
                             val_accuracy=[0.3, 0.5, 0.6, 0.7][epoch - 1]))
            rows.append(dict(approach="frozen", seed=seed, epoch=epoch,
                             seconds=epoch * 4.0 + seed * 0.1,
                             val_accuracy=[0.7, 0.8, 0.85, 0.9][epoch - 1]))
    return pd.DataFrame(rows)


def test_summarise_reduction_and_counts():
    s = summarise(make_epochs(), target=0.7, n_boot=500, seed=0)
    by = {r["approach"]: r for r in s}
    assert by["scratch"]["n_reached"] == 5 and by["frozen"]["n_reached"] == 5
    assert by["scratch"]["mean_seconds"] == pytest.approx(40.2)
    assert by["frozen"]["mean_seconds"] == pytest.approx(4.2)
    assert by["frozen"]["reduction_seconds"] == pytest.approx(1 - 4.2 / 40.2)
    assert by["frozen"]["reduction_epochs"] == pytest.approx(0.75)
    assert by["scratch"]["reduction_seconds"] == 0.0
    lo, hi = by["frozen"]["reduction_seconds_ci"]
    assert lo <= by["frozen"]["reduction_seconds"] <= hi


def test_summarise_censors_misses_at_budget():
    s = summarise(make_epochs(), target=0.95, n_boot=200, seed=0)
    by = {r["approach"]: r for r in s}
    assert by["scratch"]["n_reached"] == 0
    # censored at the full budget (epoch 4), so the figure is a lower bound
    assert by["scratch"]["mean_epochs"] == 4.0
    assert by["scratch"]["censored"] is True


def test_summarise_final_accuracy():
    s = summarise(make_epochs(), target=0.7, n_boot=200, seed=0)
    by = {r["approach"]: r for r in s}
    assert by["frozen"]["final_val_accuracy_mean"] == pytest.approx(0.9)
    assert by["scratch"]["final_val_accuracy_mean"] == pytest.approx(0.7)
