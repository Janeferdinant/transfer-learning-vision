"""Pure NumPy helpers for the Grad-CAM comparison (unit-tested without TensorFlow)."""
from __future__ import annotations

import numpy as np


def normalise_map(cam: np.ndarray) -> np.ndarray:
    """Scale each map in a batch (N, H, W) to [0, 1]; an all-zero map stays all zero."""
    cam = np.maximum(np.asarray(cam, dtype=float), 0.0)
    peak = cam.reshape(len(cam), -1).max(axis=1)
    peak = np.where(peak > 0, peak, 1.0)
    return cam / peak[:, None, None]


def top_fraction_mask(maps: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask (N, H, W) marking, per image, the ``fraction`` of pixels with the highest value.

    Ties are broken by pixel order, so exactly ``round(fraction * H * W)`` pixels are selected.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    n, h, w = maps.shape
    k = max(1, int(round(fraction * h * w)))
    flat = maps.reshape(n, -1)
    order = np.argsort(-flat, axis=1, kind="stable")[:, :k]
    mask = np.zeros_like(flat, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask.reshape(n, h, w)


def apply_mask(images: np.ndarray, mask: np.ndarray, fill: float = 127.5) -> np.ndarray:
    """Replace masked pixels of (N, H, W, 3) images with a constant grey."""
    out = np.asarray(images, dtype="float32").copy()
    out[mask] = fill
    return out


def random_maps(n: int, grid: int, seed: int) -> np.ndarray:
    """Random control maps at the same native resolution as the CAM grid."""
    return np.random.default_rng(seed).random((n, grid, grid))
