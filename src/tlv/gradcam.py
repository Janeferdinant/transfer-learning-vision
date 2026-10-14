"""Grad-CAM comparison of a from-scratch CNN and a fine-tuned MobileNetV2 on the flowers test set.

    python -m tlv.gradcam --out results/gradcam

Trains one model of each kind (seed 0, the same 30-epoch protocol as the main runs), then
  1. draws Grad-CAM heat maps for one test image per class (saved as a small PNG), and
  2. runs a masking check on all test images: grey out the 20% of pixels Grad-CAM rates highest
     and measure how much the model's confidence in its own prediction drops, compared with
     greying out 20% chosen by a random control map of the same resolution.
This is an explanation aid for two models, not a proof of what either one "looks at".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .cam_utils import apply_mask, normalise_map, random_maps, top_fraction_mask  # noqa: E402
from .data import CLASSES, build_cache, load_splits  # noqa: E402
from .grids import LEGACY_LRS  # noqa: E402
from .metrics import bootstrap_ci, bootstrap_diff_ci  # noqa: E402
from .models import build_model  # noqa: E402
from .runner import make_ds, seed_everything  # noqa: E402

MASK_FRACTION = 0.2


def train(approach: str, splits, seed: int, epochs: int) -> keras.Model:
    (Xtr, ytr), (Xva, yva), _ = splits
    seed_everything(seed)
    model = build_model(approach, 96, len(CLASSES), LEGACY_LRS[approach])
    model.fit(make_ds(Xtr, ytr, seed, shuffle=True), validation_data=make_ds(Xva, yva), epochs=epochs, verbose=0)
    return model


def gradcam(model: keras.Model, X: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM for each image's predicted class, on the ``feature_map`` tensor (last conv features).

    Returns (maps at the native feature-map resolution, N x h x w in [0, 1]; predicted class).
    The class score is the pre-softmax logit, so saturated probabilities do not kill the gradient.
    """
    head = model.layers[-1]
    old_activation = head.activation
    head.activation = keras.activations.linear
    grad_model = keras.Model(model.inputs, [model.get_layer("feature_map").output, model.output])
    maps, classes = [], []
    try:
        for i in range(0, len(X), batch):
            xb = tf.convert_to_tensor(X[i:i + batch].astype("float32"))
            with tf.GradientTape() as tape:
                fmap, logits = grad_model(xb, training=False)
                cls = tf.argmax(logits, axis=1)
                score = tf.gather(logits, cls, batch_dims=1)
            grads = tape.gradient(score, fmap)
            weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
            cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))
            maps.append(cam.numpy())
            classes.append(cls.numpy())
    finally:
        head.activation = old_activation
    return normalise_map(np.concatenate(maps)), np.concatenate(classes)


def upsample(maps: np.ndarray, size: int) -> np.ndarray:
    return tf.image.resize(maps[..., None], (size, size), method="bilinear").numpy()[..., 0]


def predict(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X.astype("float32"), batch_size=64, verbose=0)


def masking_check(model: keras.Model, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Confidence drop when the top 20% Grad-CAM pixels vs a random-control 20% are greyed out."""
    maps, pred = gradcam(model, X)
    probs = predict(model, X)
    correct = pred == y
    grid = maps.shape[1]
    cam_up = upsample(maps, X.shape[1])
    rnd_up = upsample(random_maps(len(X), grid, seed), X.shape[1])
    p0 = probs[np.arange(len(X)), pred]
    drops = {}
    for name, m in (("gradcam", cam_up), ("random", rnd_up)):
        masked = apply_mask(X, top_fraction_mask(m, MASK_FRACTION))
        p1 = predict(model, masked)[np.arange(len(X)), pred]
        drops[name] = (p0 - p1)[correct]
    est, lo, hi = bootstrap_diff_ci(drops["gradcam"], drops["random"], seed=0)
    gl, gh = bootstrap_ci(drops["gradcam"], seed=0)
    rl, rh = bootstrap_ci(drops["random"], seed=0)
    return dict(feature_map_grid=int(grid), n_correct=int(correct.sum()), n_images=int(len(X)),
                test_accuracy=float(correct.mean()),
                gradcam_drop_mean=float(drops["gradcam"].mean()), gradcam_drop_ci=[gl, gh],
                random_drop_mean=float(drops["random"].mean()), random_drop_ci=[rl, rh],
                gradcam_minus_random=est, gradcam_minus_random_ci=[lo, hi])


def pick_examples(y: np.ndarray, seed: int) -> list[int]:
    """One test image per class, chosen with a seeded RNG (not cherry-picked)."""
    rng = np.random.default_rng(seed)
    return [int(rng.choice(np.flatnonzero(y == c))) for c in range(len(CLASSES))]


def figure(X, y, idx, maps: dict[str, np.ndarray], preds: dict[str, np.ndarray],
           probs: dict[str, np.ndarray], out: Path) -> None:
    names = list(maps)
    fig, axes = plt.subplots(len(idx), 1 + len(names), figsize=(2.0 * (1 + len(names)), 2.0 * len(idx)), dpi=90)
    for r, i in enumerate(idx):
        axes[r, 0].imshow(X[i].astype("uint8"))
        axes[r, 0].set_title(f"true: {CLASSES[y[i]]}", fontsize=7)
        for c, n in enumerate(names, start=1):
            ax = axes[r, c]
            ax.imshow(X[i].astype("uint8"))
            ax.imshow(upsample(maps[n][i:i + 1], X.shape[1])[0], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ok = "ok" if preds[n][i] == y[i] else "wrong"
            ax.set_title(f"{n}: {CLASSES[preds[n][i]]} {probs[n][i, preds[n][i]]:.2f} ({ok})", fontsize=6)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/gradcam")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)
    out = Path(a.out)
    splits = load_splits(build_cache(Path(a.data_dir), 96, 240))
    Xte, yte = splits[2]
    models = {"scratch": train("scratch", splits, a.seed, a.epochs),
              "fine-tuned": train("finetune", splits, a.seed, a.epochs)}
    maps, preds, probs, summary = {}, {}, {}, {}
    for n, m in models.items():
        maps[n], preds[n] = gradcam(m, Xte)
        probs[n] = predict(m, Xte)
        summary[n] = masking_check(m, Xte, yte, a.seed)
    idx = pick_examples(yte, a.seed)
    figure(Xte, yte, idx, maps, preds, probs, out / "gradcam_scratch_vs_finetuned.png")
    summary["examples"] = dict(test_indices=idx, seed=a.seed, mask_fraction=MASK_FRACTION, epochs=a.epochs)
    (out / "gradcam_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
