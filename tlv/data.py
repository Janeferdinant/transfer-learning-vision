"""Dataset download, subsampling and a seeded stratified train/val/test split.

The TF Flowers images are downloaded at run time into ``data/`` (git-ignored) and
are never committed. Splitting logic is pure NumPy so it can be unit-tested
without TensorFlow or image files.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_SEED = 2026  # fixed: every run and every approach sees the same split


# ---------------------------------------------------------------- pure logic
def stratified_split(labels, fractions: Sequence[float] = SPLIT_FRACTIONS, seed: int = SPLIT_SEED):
    """Seeded per-class split into (train, val, test) index arrays.

    Every index appears in exactly one split, so there is no leakage by
    construction. Class proportions are preserved in each split.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be three numbers summing to 1")
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n = idx.size
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        if n >= 3:  # keep every class present in every split
            n_tr = min(max(n_tr, 1), n - 2)
            n_va = min(max(n_va, 1), n - n_tr - 1)
        parts[0].append(idx[:n_tr])
        parts[1].append(idx[n_tr:n_tr + n_va])
        parts[2].append(idx[n_tr + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)


def subsample_per_class(labels, n_per_class: int, seed: int = SPLIT_SEED):
    """Seeded, class-balanced subsample; returns sorted indices."""
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if idx.size < n_per_class:
            raise ValueError(f"class {c} has {idx.size} images, need {n_per_class}")
        keep.append(rng.choice(idx, size=n_per_class, replace=False))
    return np.sort(np.concatenate(keep))


def _as_bytes(x) -> bytes:
    return x if isinstance(x, (bytes, bytearray)) else np.ascontiguousarray(x).tobytes()


def dedupe_by_hash(items: Iterable) -> np.ndarray:
    """Indices of the first occurrence of each distinct item (bytes or arrays)."""
    seen, keep = set(), []
    for i, x in enumerate(items):
        h = hashlib.sha256(_as_bytes(x)).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return np.asarray(keep, dtype=int)


# ------------------------------------------------------------- I/O (run time)
def download_flowers(data_dir: Path) -> Path:
    """Download and extract TF Flowers once; returns the extracted folder."""
    data_dir.mkdir(parents=True, exist_ok=True)
    root = data_dir / "flower_photos"
    if root.exists():
        return root
    archive = data_dir / "flower_photos.tgz"
    if not archive.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, archive)
    print("sha256 of archive:", hashlib.sha256(archive.read_bytes()).hexdigest())
    with tarfile.open(archive) as tf:
        tf.extractall(data_dir, filter="data")
    return root


def build_cache(data_dir: Path, image_size: int = 96, n_per_class: int = 240,
                seed: int = SPLIT_SEED) -> Path:
    """Decode, centre-crop, resize and cache the subsample as uint8 arrays.

    One-off cost that is excluded from every timing. Writes a split manifest
    (file names only) next to the cache for reproducibility.
    """
    from PIL import Image  # local import: keeps the pure logic importable without Pillow

    cache = data_dir / f"flowers_{image_size}px_{n_per_class}pc_seed{seed}.npz"
    if cache.exists():
        return cache
    root = download_flowers(data_dir)
    paths, labels = [], []
    for ci, name in enumerate(CLASSES):
        for p in sorted((root / name).glob("*.jpg")):
            paths.append(p)
            labels.append(ci)
    labels = np.asarray(labels)
    raw = [p.read_bytes() for p in paths]
    uniq = dedupe_by_hash(raw)
    n_dupes = len(paths) - len(uniq)
    paths = [paths[i] for i in uniq]
    labels = labels[uniq]
    sel = subsample_per_class(labels, n_per_class, seed)
    paths = [paths[i] for i in sel]
    labels = labels[sel]

    def load(p: Path):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.asarray(im.resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8)

    X = np.stack([load(p) for p in paths])
    # second, pixel-level duplicate check after resizing (near-identical images)
    keep = dedupe_by_hash(X)
    n_pixel_dupes = len(X) - len(keep)
    X, labels, paths = X[keep], labels[keep], [paths[i] for i in keep]
    tr, va, te = stratified_split(labels, SPLIT_FRACTIONS, seed)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(a) & set(b)), "index leakage between splits"
    hashes = [hashlib.sha256(X[i].tobytes()).hexdigest() for i in range(len(X))]
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not ({hashes[i] for i in a} & {hashes[i] for i in b}), "pixel leakage between splits"
    np.savez_compressed(cache, X=X, y=labels, tr=tr, va=va, te=te)
    manifest = {
        "url": DATA_URL, "image_size": image_size, "n_per_class_requested": n_per_class,
        "exact_file_duplicates_removed": int(n_dupes),
        "pixel_duplicates_removed_after_resize": int(n_pixel_dupes),
        "split_fractions": SPLIT_FRACTIONS, "split_seed": seed,
        "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "files": {s: [f"{CLASSES[labels[i]]}/{paths[i].name}" for i in idx]
                  for s, idx in (("train", tr), ("val", va), ("test", te))},
    }
    (data_dir / (cache.stem + "_manifest.json")).write_text(json.dumps(manifest))
    return cache


def load_splits(cache: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    z = np.load(cache)
    return tuple((z["X"][z[k]], z["y"][z[k]]) for k in ("tr", "va", "te"))
