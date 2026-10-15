"""Run the experiment: train each approach for each seed, log per-epoch curves.

    python -m tlv.experiment --seeds 0 1 2 3 4 --epochs 20

Resumable: (approach, seed) pairs already present in results/per_run.csv are
skipped. The timer covers model.fit only (training plus per-epoch validation):
dataset download/caching, model construction and ImageNet weight download are
excluded, as is the final test-set evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import pandas as pd
import tensorflow as tf

from . import hardware
from .data import build_cache, load_splits
from .models import APPROACHES, LEARNING_RATES, build_model, count_params
from .runner import TimingCallback, make_ds, seed_everything


def run_one(approach, seed, epochs, splits, image_size, lr=None):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    seed_everything(seed)
    model = build_model(approach, image_size, learning_rate=lr)
    trainable, total = count_params(model)
    cb = TimingCallback()
    model.fit(make_ds(Xtr, ytr, seed, shuffle=True), validation_data=make_ds(Xva, yva),
              epochs=epochs, callbacks=[cb], verbose=0)
    test_loss, test_acc = model.evaluate(make_ds(Xte, yte), verbose=0)  # outside the timer
    epoch_rows = [dict(approach=approach, seed=seed, **r) for r in cb.rows]
    run_row = dict(approach=approach, seed=seed, epochs=epochs,
                   learning_rate=lr or LEARNING_RATES[approach],
                   total_seconds=cb.rows[-1]["seconds"],
                   final_val_accuracy=cb.rows[-1]["val_accuracy"],
                   best_val_accuracy=max(r["val_accuracy"] for r in cb.rows),
                   test_accuracy=float(test_acc), trainable_params=trainable, total_params=total,
                   finished_utc=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return epoch_rows, run_row


def warm_up(splits, image_size):
    """Untimed one-epoch runs so one-off framework start-up cost is not charged to
    whichever approach happens to run first. Also pre-fetches ImageNet weights."""
    (Xtr, ytr), (Xva, yva), _ = splits
    for ap in APPROACHES:
        m = build_model(ap, image_size)
        m.fit(make_ds(Xtr[:64], ytr[:64]), validation_data=make_ds(Xva[:32], yva[:32]),
              epochs=1, verbose=0)


def append_csv(path: Path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--approaches", nargs="+", default=list(APPROACHES), choices=APPROACHES)
    p.add_argument("--lr", type=float, default=None, help="override the learning rate for every approach (pilot sweeps)")
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--n-per-class", type=int, default=240)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = build_cache(Path(a.data_dir), a.image_size, a.n_per_class)
    splits = load_splits(cache)
    print("split sizes (train/val/test):", [len(s[0]) for s in splits])
    hw = hardware.describe(tf.__version__)
    (out / "hardware.json").write_text(json.dumps(hw, indent=2))
    print("hardware:", hardware.short(hw))

    runs_csv, epochs_csv = out / "per_run.csv", out / "per_epoch.csv"
    done = set()
    if runs_csv.exists():
        prev = pd.read_csv(runs_csv)
        done = set(zip(prev.approach, prev.seed))

    warm_up(splits, a.image_size)
    for i, seed in enumerate(a.seeds):
        # rotate the order each seed so no approach is always first or last
        order = a.approaches[i % len(a.approaches):] + a.approaches[:i % len(a.approaches)]
        for ap in order:
            if (ap, seed) in done:
                continue
            t0 = time.perf_counter()
            ep_rows, run_row = run_one(ap, seed, a.epochs, splits, a.image_size, a.lr)
            append_csv(epochs_csv, ep_rows)
            append_csv(runs_csv, [run_row])
            print(f"{ap:9s} seed={seed} train={run_row['total_seconds']:.1f}s "
                  f"val={run_row['final_val_accuracy']:.3f} test={run_row['test_accuracy']:.3f} "
                  f"(wall incl. eval {time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
