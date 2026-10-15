"""The three approaches, for several pretrained backbones.

Everything except the trainable weights is identical across approaches: input scaling,
augmentation, classification head, optimiser, loss and batch size. Backbones differ only in
the input scaling their ImageNet weights expect.
"""
from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from .blocks import ALL_BLOCKS, first_trainable_index
from .config import BATCH_SIZE

APPROACHES = ("scratch", "frozen", "finetune")
BACKBONES = ("mobilenetv2", "mobilenetv3small", "efficientnetb0", "resnet50")
DEFAULT_BLOCKS = 4  # MobileNetV2: block_13_expand onwards, the choice made in the 2026-10-14 run
FINE_TUNE_FROM = "block_13_expand"  # kept for reference: first unfrozen layer of the legacy setting
# Adam everywhere; the learning rate is the one per-approach hyper-parameter and was
# chosen on a pilot seed that is NOT among the reported seeds (see docs/runs/).
LEARNING_RATES = {"scratch": 3e-4, "frozen": 1e-3, "finetune": 3e-4}

_CAFFE_MEAN = np.array([103.939, 116.779, 123.68], dtype="float32")


def _caffe(t):
    """ResNet50 (Keras) expects BGR channel order with the ImageNet mean subtracted."""
    return keras.ops.flip(t, axis=-1) - _CAFFE_MEAN


def _prep(backbone: str):
    """Input scaling layers for a backbone (images arrive as floats in 0..255)."""
    if backbone in ("mobilenetv2", "mobilenetv3small"):
        return layers.Rescaling(1 / 127.5, offset=-1, name="prep")  # -> [-1, 1]
    if backbone == "resnet50":
        return layers.Lambda(_caffe, name="prep")
    if backbone == "efficientnetb0":
        return layers.Identity(name="prep")  # Keras EfficientNetB0 rescales and normalises inside
    raise ValueError(f"unknown backbone {backbone!r}")


def _make_backbone(backbone: str, image_size: int, pretrained: bool) -> keras.Model:
    kw = dict(input_shape=(image_size, image_size, 3), include_top=False,
              weights="imagenet" if pretrained else None)
    if backbone == "mobilenetv2":
        model = keras.applications.MobileNetV2(alpha=1.0, **kw)
    elif backbone == "mobilenetv3small":
        model = keras.applications.MobileNetV3Small(include_preprocessing=False, **kw)
    elif backbone == "efficientnetb0":
        model = keras.applications.EfficientNetB0(**kw)
    elif backbone == "resnet50":
        model = keras.applications.ResNet50(**kw)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    model.name = "backbone"  # set after construction: Keras derives weight file names from `name`
    return model


def _augment(x):
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    return layers.RandomZoom(0.1)(x)


def _head(x, n_classes: int):
    x = layers.Identity(name="feature_map")(x)  # the tensor Grad-CAM explains
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def set_trainable(backbone_model: keras.Model, name: str, k: int) -> None:
    """Unfreeze the last ``k`` blocks (``ALL_BLOCKS`` = everything); batch-norm stays frozen."""
    start = first_trainable_index([layer.name for layer in backbone_model.layers], name, k)
    backbone_model.trainable = True
    for i, layer in enumerate(backbone_model.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)


def build_model(approach: str, image_size: int = 96, n_classes: int = 5,
                learning_rate: float | None = None, backbone: str = "mobilenetv2",
                unfreeze_blocks: int = DEFAULT_BLOCKS, pretrained: bool = True) -> keras.Model:
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach {approach!r}")
    inputs = keras.Input((image_size, image_size, 3))
    if approach == "scratch":
        x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)
        x = _augment(x)
        for filters in (32, 64, 128, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D(2)(x)
    else:
        base = _make_backbone(backbone, image_size, pretrained)
        base.trainable = False
        if approach == "finetune":
            set_trainable(base, backbone, unfreeze_blocks)
        x = _augment(_prep(backbone)(inputs))
        x = base(x, training=False)  # BN always in inference mode
    outputs = _head(x, n_classes)
    model = keras.Model(inputs, outputs, name=approach)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate or LEARNING_RATES[approach]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model: keras.Model) -> tuple[int, int]:
    trainable = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    total = int(sum(int(tf.size(w)) for w in model.weights))
    return trainable, total


__all__ = ["ALL_BLOCKS", "APPROACHES", "BACKBONES", "BATCH_SIZE", "DEFAULT_BLOCKS", "LEARNING_RATES",
           "build_model", "count_params", "set_trainable"]
