"""The three approaches. Everything except the trainable weights is identical:
input scaling, augmentation, classification head, optimiser, loss and batch size."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

APPROACHES = ("scratch", "frozen", "finetune")
FINE_TUNE_FROM = "block_13_expand"  # last four inverted-residual blocks + final 1x1 conv
# Adam everywhere; the learning rate is the one per-approach hyper-parameter and was
# chosen on a pilot seed that is NOT among the reported seeds (see docs/runs/).
LEARNING_RATES = {"scratch": 3e-4, "frozen": 1e-3, "finetune": 3e-4}
BATCH_SIZE = 32


def _front(image_size: int):
    """Shared input pipeline: scale to [-1, 1] (what MobileNetV2 expects) + augmentation."""
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    return inputs, x


def _head(x, n_classes: int):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def build_model(approach: str, image_size: int = 96, n_classes: int = 5,
                learning_rate: float | None = None) -> keras.Model:
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach {approach!r}")
    inputs, x = _front(image_size)
    if approach == "scratch":
        for filters in (32, 64, 128, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D(2)(x)
    else:
        backbone = keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3), include_top=False,
            weights="imagenet", alpha=1.0)
        backbone.trainable = False
        if approach == "finetune":
            backbone.trainable = True
            unfreeze = False
            for layer in backbone.layers:
                if layer.name == FINE_TUNE_FROM:
                    unfreeze = True
                # BatchNorm statistics stay frozen; only conv/depthwise weights adapt.
                layer.trainable = unfreeze and not isinstance(layer, layers.BatchNormalization)
        x = backbone(x, training=False)  # BN always in inference mode
    outputs = _head(x, n_classes)
    model = keras.Model(inputs, outputs, name=approach)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate or LEARNING_RATES[approach]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model: keras.Model):
    trainable = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    total = int(sum(int(tf.size(w)) for w in model.weights))
    return trainable, total
