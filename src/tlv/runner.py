"""Train one configuration and return per-epoch and per-run rows.

The timer covers ``model.fit`` only (training plus the per-epoch validation pass): dataset
download and caching, model construction, ImageNet weight download and the final test-set
evaluation are excluded.
"""
from __future__ import annotations

import datetime as dt
import os
import random
from dataclasses import asdict
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from .config import BATCH_SIZE, RunConfig, steps_per_epoch  # noqa: E402
from .data import Splits, select_per_class  # noqa: E402
from .models import build_model, count_params  # noqa: E402
from .timing import EpochTimer  # noqa: E402


class TimingCallback(keras.callbacks.Callback):
    """Thin Keras adapter around EpochTimer; lap happens after validation."""

    def __init__(self) -> None:
        super().__init__()
        self.timer = EpochTimer()
        self.rows: list[dict[str, float]] = []

    def on_train_begin(self, logs: Any = None) -> None:
        self.timer.start()

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        secs = self.timer.lap()
        self.rows.append(dict(epoch=epoch + 1, seconds=secs,
                              train_loss=float(logs["loss"]), train_accuracy=float(logs["accuracy"]),
                              val_loss=float(logs["val_loss"]), val_accuracy=float(logs["val_accuracy"])))


def make_ds(X: np.ndarray, y: np.ndarray, seed: int | None = None, shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def run_config(cfg: RunConfig, splits: Splits, n_classes: int) -> tuple[list[dict], dict]:
    """Train ``cfg`` on its (sub)sampled training set; returns (epoch rows, run row)."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    c = cfg.normalised()
    sel = select_per_class(ytr, c.train_per_class, c.seed)
    Xtr, ytr = Xtr[sel], ytr[sel]
    epochs = c.effective_epochs(len(Xtr))
    seed_everything(c.seed)
    model = build_model(c.approach, c.image_size, n_classes, c.learning_rate,
                        backbone=c.backbone if c.approach != "scratch" else "mobilenetv2",
                        unfreeze_blocks=c.unfreeze_blocks if c.approach == "finetune" else 4)
    trainable, total = count_params(model)
    cb = TimingCallback()
    model.fit(make_ds(Xtr, ytr, c.seed, shuffle=True), validation_data=make_ds(Xva, yva),
              epochs=epochs, callbacks=[cb], verbose=0)
    _, test_acc = model.evaluate(make_ds(Xte, yte), verbose=0)  # outside the timer
    rid = c.run_id()
    epoch_rows = [dict(run_id=rid, **r) for r in cb.rows]
    secs = [r["seconds"] for r in cb.rows]
    steady = np.diff(secs) if len(secs) > 1 else np.array(secs)
    run_row = dict(run_id=rid, **asdict(c), epochs_run=epochs, n_train=len(Xtr),
                   steps_per_epoch=steps_per_epoch(len(Xtr)),
                   total_seconds=secs[-1], first_epoch_seconds=secs[0],
                   steady_epoch_seconds=float(np.median(steady)),
                   final_val_accuracy=cb.rows[-1]["val_accuracy"],
                   best_val_accuracy=max(r["val_accuracy"] for r in cb.rows),
                   test_accuracy=float(test_acc), trainable_params=trainable, total_params=total,
                   finished_utc=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    keras.backend.clear_session()
    return epoch_rows, run_row


def warm_up(splits: Splits, n_classes: int, image_size: int, combos: list[tuple[str, str]]) -> None:
    """Untimed tiny runs so one-off framework start-up cost (and the ImageNet weight download)
    is not charged to whichever configuration runs first."""
    (Xtr, ytr), (Xva, yva), _ = splits
    for approach, backbone in combos:
        m = build_model(approach, image_size, n_classes, backbone=backbone)
        m.fit(make_ds(Xtr[:64], ytr[:64]), validation_data=make_ds(Xva[:32], yva[:32]), epochs=1, verbose=0)
        keras.backend.clear_session()
