"""Precompute the data for docs/index.html from the recorded per-epoch logs.

Writes two scripts rather than JSON so the page also works when opened straight from disk
(file://, no server):

* ``docs/curves.js``          ``window.CURVES``: the 2026-10-14 TF Flowers run (MobileNetV2, 10 seeds)
* ``docs/explorer-data.js``   ``window.EXPLORER``: both datasets, all backbones and the
                              data-efficiency runs, built from the CSVs in ``results/``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregate import ORDER

DATASET_INFO = {
    "flowers": dict(label="TF Flowers (5 classes)", classes=5),
    "eurosat": dict(label="EuroSAT RGB (10 classes)", classes=10),
}


def build(epochs: pd.DataFrame, hardware: dict, run_date: str) -> dict:
    approaches = [a for a in ORDER if a in set(epochs.approach)]
    curves = {}
    for ap in approaches:
        runs = []
        for seed, g in epochs[epochs.approach == ap].groupby("seed"):
            g = g.sort_values("epoch")
            runs.append(dict(seed=int(seed),  # type: ignore[arg-type]
                             seconds=[round(float(v), 3) for v in g.seconds],
                             val_accuracy=[round(float(v), 4) for v in g.val_accuracy]))
        curves[ap] = runs
    n_seeds = len({r["seed"] for r in curves[approaches[0]]})
    return dict(meta=dict(date=run_date, n_seeds=n_seeds, hardware=hardware,
                          epochs=int(epochs.epoch.max())),
                approaches=approaches, curves=curves)


def _runs(runs: pd.DataFrame, epochs: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    by = {rid: g.sort_values("epoch") for rid, g in epochs.groupby("run_id")}
    for r in runs.sort_values("seed").itertuples():
        g = by[r.run_id]
        out.append(dict(seed=int(r.seed), seconds=[round(float(v), 3) for v in g.seconds],  # type: ignore[arg-type]
                        val_accuracy=[round(float(v), 4) for v in g.val_accuracy]))
    return out


def build_explorer(results: Path, run_date: str, hardware: dict) -> dict[str, Any]:
    """Curves of every committed suite run, grouped for the explorer."""
    data: dict[str, Any] = {"meta": dict(date=run_date, hardware=hardware), "datasets": {}}
    for ds, info in DATASET_INFO.items():
        entry: dict[str, Any] = dict(info, backbones={}, scratch=None, dataEff={})
        bb = results / f"backbones_{ds}"
        if (bb / "per_run.csv").exists():
            runs, ep = pd.read_csv(bb / "per_run.csv"), pd.read_csv(bb / "per_epoch.csv")
            s = runs[runs.approach == "scratch"]
            entry["scratch"] = dict(runs=_runs(s, ep), trainable=int(s.trainable_params.iloc[0]),
                                    total=int(s.total_params.iloc[0]))
            for b in sorted(set(runs.backbone) - {"-"}):
                entry["backbones"][b] = {}
                for ap in ("frozen", "finetune"):
                    g = runs[(runs.backbone == b) & (runs.approach == ap)]
                    if len(g):
                        entry["backbones"][b][ap] = dict(runs=_runs(g, ep), trainable=int(g.trainable_params.iloc[0]),
                                                         total=int(g.total_params.iloc[0]))
        de = results / f"data_efficiency_{ds}"
        if (de / "per_run.csv").exists():
            runs, ep = pd.read_csv(de / "per_run.csv"), pd.read_csv(de / "per_epoch.csv")
            runs["train_per_class"] = runs["train_per_class"].fillna(-1).astype(int)
            for ap in ORDER:
                entry["dataEff"][ap] = {}
                for _, g in runs[runs.approach == ap].groupby("train_per_class"):
                    per_class = int(g.n_train.iloc[0] // info["classes"])
                    entry["dataEff"][ap][str(per_class)] = _runs(g, ep)
        data["datasets"][ds] = entry
    return data


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-csv", default="results/per_epoch.csv")
    p.add_argument("--hardware", default="results/hardware.json")
    p.add_argument("--date", required=True)
    p.add_argument("--out", default="docs/curves.js")
    p.add_argument("--explorer-out", default="docs/explorer-data.js")
    p.add_argument("--results", default="results")
    p.add_argument("--legacy-only", action="store_true", help="only rewrite docs/curves.js")
    a = p.parse_args(argv)
    hw = json.loads(Path(a.hardware).read_text())
    if a.legacy_only:
        data = build(pd.read_csv(a.epochs_csv), hw, a.date)
        Path(a.out).write_text("window.CURVES = " + json.dumps(data, separators=(",", ":")) + ";\n")
        print("wrote", a.out)
        return
    ex = build_explorer(Path(a.results), a.date, hw)
    Path(a.explorer_out).write_text("window.EXPLORER = " + json.dumps(ex, separators=(",", ":")) + ";\n")
    print("wrote", a.explorer_out)


if __name__ == "__main__":
    main()
