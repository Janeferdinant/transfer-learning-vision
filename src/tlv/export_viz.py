"""Precompute the data for docs/index.html from the recorded per-epoch log.

Writes docs/curves.js (``window.CURVES = {...}``) as a script rather than JSON so
the page also works when opened straight from disk (file://, no server).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .aggregate import ORDER


def build(epochs: pd.DataFrame, hardware: dict, run_date: str) -> dict:
    approaches = [a for a in ORDER if a in set(epochs.approach)]
    curves = {}
    for ap in approaches:
        runs = []
        for seed, g in epochs[epochs.approach == ap].groupby("seed"):
            g = g.sort_values("epoch")
            runs.append(dict(seed=int(seed),
                             seconds=[round(float(v), 3) for v in g.seconds],
                             val_accuracy=[round(float(v), 4) for v in g.val_accuracy]))
        curves[ap] = runs
    n_seeds = len({r["seed"] for r in curves[approaches[0]]})
    return dict(meta=dict(date=run_date, n_seeds=n_seeds, hardware=hardware,
                          epochs=int(epochs.epoch.max())),
                approaches=approaches, curves=curves)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-csv", default="results/per_epoch.csv")
    p.add_argument("--hardware", default="results/hardware.json")
    p.add_argument("--date", required=True)
    p.add_argument("--out", default="docs/curves.js")
    a = p.parse_args(argv)
    data = build(pd.read_csv(a.epochs_csv), json.loads(Path(a.hardware).read_text()), a.date)
    Path(a.out).write_text("window.CURVES = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
