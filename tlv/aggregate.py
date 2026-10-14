"""Aggregate per-epoch logs into time-to-target summaries with bootstrap intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    bootstrap_ci,
    bootstrap_reduction_ci,
    censored_values,
    time_to_target,
)

BASELINE = "scratch"
ORDER = ["scratch", "frozen", "finetune"]


def _per_run(epochs: pd.DataFrame, approach: str):
    for seed, g in epochs[epochs.approach == approach].groupby("seed"):
        g = g.sort_values("epoch")
        yield int(seed), g["seconds"].to_numpy(), g["val_accuracy"].to_numpy()


def _ttt_arrays(epochs: pd.DataFrame, approach: str, target: float):
    runs, budgets, n_epochs = [], [], 0
    for _, t, a in _per_run(epochs, approach):
        runs.append(time_to_target(t, a, target))
        budgets.append(float(t[-1]))
        n_epochs = max(n_epochs, len(t))
    return censored_values(runs, budgets, n_epochs)


def summarise(epochs: pd.DataFrame, target: float, n_boot: int = 10000, seed: int = 0):
    """One summary row per approach for a given target validation accuracy.

    Runs that never reach the target are charged the full budget (see
    ``censored_values``), so their time is a lower bound and ``censored`` is True.
    """
    approaches = [a for a in ORDER if a in set(epochs.approach)] + sorted(
        set(epochs.approach) - set(ORDER)
    )
    base_s, base_e, _ = _ttt_arrays(epochs, BASELINE, target)
    rows = []
    for ap in approaches:
        secs, eps, n_reached = _ttt_arrays(epochs, ap, target)
        finals = np.array([a[-1] for _, _, a in _per_run(epochs, ap)])
        row = dict(
            approach=ap,
            target=target,
            n_runs=int(secs.size),
            n_reached=int(n_reached),
            censored=bool(n_reached < secs.size),
            mean_seconds=float(secs.mean()),
            seconds_ci=bootstrap_ci(secs, n_boot=n_boot, seed=seed),
            mean_epochs=float(eps.mean()),
            epochs_ci=bootstrap_ci(eps, n_boot=n_boot, seed=seed),
            final_val_accuracy_mean=float(finals.mean()),
            final_val_accuracy_ci=bootstrap_ci(finals, n_boot=n_boot, seed=seed),
        )
        if ap == BASELINE:
            row.update(reduction_seconds=0.0, reduction_seconds_ci=(0.0, 0.0),
                       reduction_epochs=0.0, reduction_epochs_ci=(0.0, 0.0))
        else:
            r, lo, hi = bootstrap_reduction_ci(base_s, secs, n_boot=n_boot, seed=seed)
            re, elo, ehi = bootstrap_reduction_ci(base_e, eps, n_boot=n_boot, seed=seed)
            row.update(reduction_seconds=r, reduction_seconds_ci=(lo, hi),
                       reduction_epochs=re, reduction_epochs_ci=(elo, ehi))
        rows.append(row)
    return rows


def final_accuracy_table(epochs: pd.DataFrame, per_run: pd.DataFrame, n_boot=10000, seed=0):
    out = []
    for ap in [a for a in ORDER if a in set(per_run.approach)]:
        g = per_run[per_run.approach == ap]
        row = dict(approach=ap, n_runs=len(g))
        for col in ("final_val_accuracy", "test_accuracy", "total_seconds"):
            v = g[col].to_numpy(dtype=float)
            row[col + "_mean"] = float(v.mean())
            row[col + "_ci"] = bootstrap_ci(v, n_boot=n_boot, seed=seed)
        out.append(row)
    return out


def primary_target(epochs: pd.DataFrame, fraction: float = 0.95) -> float:
    """Pre-registered primary target: ``fraction`` x the mean FINAL validation
    accuracy of the from-scratch runs, i.e. "match (almost all of) what training
    from scratch achieves in the full budget"."""
    finals = [a[-1] for _, _, a in _per_run(epochs, BASELINE)]
    return fraction * float(np.mean(finals))


def secondary_target(epochs: pd.DataFrame, fraction: float = 0.90) -> float:
    """``fraction`` x the best approach's mean final validation accuracy."""
    best = max(
        np.mean([a[-1] for _, _, a in _per_run(epochs, ap)]) for ap in set(epochs.approach)
    )
    return fraction * float(best)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-csv", default="results/per_epoch.csv")
    p.add_argument("--runs-csv", default="results/per_run.csv")
    p.add_argument("--out", default="results/summary.json")
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args(argv)
    epochs = pd.read_csv(a.epochs_csv)
    runs = pd.read_csv(a.runs_csv)
    targets = {
        "primary_0.95_of_scratch_final": primary_target(epochs),
        "secondary_0.90_of_best_final": secondary_target(epochs),
    }
    for fixed in (0.6, 0.7, 0.8, 0.9):
        targets[f"fixed_{fixed:.2f}"] = fixed
    out = {
        "targets": targets,
        "time_to_target": {k: summarise(epochs, v, n_boot=a.n_boot) for k, v in targets.items()},
        "final_accuracy": final_accuracy_table(epochs, runs, n_boot=a.n_boot),
        "n_boot": a.n_boot,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
