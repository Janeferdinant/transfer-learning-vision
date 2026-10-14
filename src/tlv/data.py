"""Dataset download, subsampling and a seeded stratified train/val/test split.

Two public image datasets are supported: TF Flowers (CC BY 2.0 photos) and EuroSAT RGB
(MIT licence). Both are downloaded at run time into ``data/`` (git-ignored), checked against a
SHA-256 digest and never committed. Splitting logic is pure NumPy so it can be unit-tested
without TensorFlow or image files.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
EUROSAT_SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
EUROSAT_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"  # published on the Zenodo record
EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
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
    parts: tuple[list, list, list] = ([], [], [])
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


def select_per_class(labels, n_per_class: int | None, seed: int) -> np.ndarray:
    """Seeded class-balanced training subset (sorted indices); ``None`` keeps everything.

    For a fixed seed the subsets are nested (the 10-per-class subset is contained in the
    25-per-class one) because each class is permuted once and its first ``n`` items are kept.
    """
    y = np.asarray(labels)
    if n_per_class is None:
        return np.arange(y.size)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if idx.size < n_per_class:
            raise ValueError(f"class {c} has {idx.size} images, need {n_per_class}")
        keep.append(rng.permutation(idx)[:n_per_class])
    return np.sort(np.concatenate(keep))


def _as_bytes(x) -> bytes:
    return bytes(x) if isinstance(x, (bytes, bytearray)) else np.ascontiguousarray(x).tobytes()


def dedupe_by_hash(items: Iterable) -> np.ndarray:
    """Indices of the first occurrence of each distinct item (bytes or arrays)."""
    seen, keep = set(), []
    for i, x in enumerate(items):
        h = hashlib.sha256(_as_bytes(x)).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return np.asarray(keep, dtype=int)


def verify_file(path: Path, sha256: str | None = None, md5: str | None = None) -> str:
    """Check a downloaded file against published digests; returns its SHA-256."""
    h5, hm = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h5.update(chunk)
            hm.update(chunk)
    if sha256 is not None and h5.hexdigest() != sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}: got {h5.hexdigest()}")
    if md5 is not None and hm.hexdigest() != md5:
        raise ValueError(f"MD5 mismatch for {path.name}: got {hm.hexdigest()}")
    return h5.hexdigest()


# ------------------------------------------------------------- I/O (run time)
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    sha256: str
    classes: list[str]
    licence: str
    md5: str | None = None
    n_per_class: int = 240  # images kept per class (60/20/20 split of these)


DATASETS: dict[str, DatasetSpec] = {
    "flowers": DatasetSpec("flowers", DATA_URL, FLOWERS_SHA256, CLASSES, "CC BY 2.0 (per photo)"),
    "eurosat": DatasetSpec("eurosat", EUROSAT_URL, EUROSAT_SHA256, EUROSAT_CLASSES, "MIT", EUROSAT_MD5, 120),
}


def _fetch(spec: DatasetSpec, data_dir: Path) -> Path:
    """Download the archive once and verify its digest; returns the archive path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / spec.url.rsplit("/", 1)[-1]
    if not archive.exists():
        print(f"downloading {spec.url}")
        urllib.request.urlretrieve(spec.url, archive)
    digest = verify_file(archive, sha256=spec.sha256, md5=spec.md5)
    print(f"sha256 of {archive.name} verified: {digest}")
    return archive


def download_flowers(data_dir: Path) -> Path:
    """Download and extract TF Flowers once; returns the extracted folder."""
    root = data_dir / "flower_photos"
    if root.exists():
        return root
    archive = _fetch(DATASETS["flowers"], data_dir)
    with tarfile.open(archive) as tf:
        tf.extractall(data_dir, filter="data")
    return root


# A source is (relative names, integer labels, function returning the file bytes of item i).
Source = tuple[list[str], np.ndarray, Callable[[int], bytes]]


def _flowers_source(data_dir: Path) -> Source:
    root = download_flowers(data_dir)
    paths: list[Path] = []
    labels: list[int] = []
    for ci, name in enumerate(CLASSES):
        for p in sorted((root / name).glob("*.jpg")):
            paths.append(p)
            labels.append(ci)
    return [f"{p.parent.name}/{p.name}" for p in paths], np.asarray(labels), lambda i: paths[i].read_bytes()


def _eurosat_source(data_dir: Path) -> Source:
    archive = _fetch(DATASETS["eurosat"], data_dir)
    zf = zipfile.ZipFile(archive)
    members = sorted(n for n in zf.namelist() if n.endswith(".jpg"))
    names = [f"{n.split('/')[1]}/{n.rsplit('/', 1)[-1]}" for n in members]
    labels = np.asarray([EUROSAT_CLASSES.index(n.split("/")[1]) for n in members])
    return names, labels, lambda i: zf.read(members[i])


def build_cache(data_dir: Path, image_size: int = 96, n_per_class: int = 240,
                seed: int = SPLIT_SEED, dataset: str = "flowers") -> Path:
    """Decode, centre-crop, resize and cache the subsample as uint8 arrays.

    One-off cost that is excluded from every timing. Writes a split manifest
    (file names only) next to the cache for reproducibility.
    """
    import io

    from PIL import Image  # local import: keeps the pure logic importable without Pillow

    spec = DATASETS[dataset]
    cache = data_dir / f"{dataset}_{image_size}px_{n_per_class}pc_seed{seed}.npz"
    if cache.exists():
        return cache
    names, labels, read = _flowers_source(data_dir) if dataset == "flowers" else _eurosat_source(data_dir)
    raw = [read(i) for i in range(len(names))]
    uniq = dedupe_by_hash(raw)
    n_dupes = len(names) - len(uniq)
    names = [names[i] for i in uniq]
    raw = [raw[i] for i in uniq]
    labels = labels[uniq]
    sel = subsample_per_class(labels, n_per_class, seed)
    names = [names[i] for i in sel]
    raw = [raw[i] for i in sel]
    labels = labels[sel]

    def load(b: bytes) -> np.ndarray:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.asarray(im.resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.uint8)

    X = np.stack([load(b) for b in raw])
    # second, pixel-level duplicate check after resizing (near-identical images)
    keep = dedupe_by_hash(X)
    n_pixel_dupes = len(X) - len(keep)
    X, labels, names = X[keep], labels[keep], [names[i] for i in keep]
    tr, va, te = stratified_split(labels, SPLIT_FRACTIONS, seed)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(a) & set(b)), "index leakage between splits"
    hashes = [hashlib.sha256(X[i].tobytes()).hexdigest() for i in range(len(X))]
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not ({hashes[i] for i in a} & {hashes[i] for i in b}), "pixel leakage between splits"
    np.savez_compressed(cache, X=X, y=labels, tr=tr, va=va, te=te)
    manifest = {
        "dataset": dataset, "url": spec.url, "archive_sha256": spec.sha256, "licence": spec.licence,
        "image_size": image_size, "n_per_class_requested": n_per_class,
        "exact_file_duplicates_removed": int(n_dupes),
        "pixel_duplicates_removed_after_resize": int(n_pixel_dupes),
        "split_fractions": SPLIT_FRACTIONS, "split_seed": seed,
        "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "files": {s: [names[i] for i in idx] for s, idx in (("train", tr), ("val", va), ("test", te))},
    }
    (data_dir / (cache.stem + "_manifest.json")).write_text(json.dumps(manifest))
    return cache


Pair = tuple[np.ndarray, np.ndarray]
Splits = tuple[Pair, Pair, Pair]


def load_splits(cache: Path) -> Splits:
    z = np.load(cache)

    def part(k: str) -> Pair:
        return z["X"][z[k]], z["y"][z[k]]

    return part("tr"), part("va"), part("te")
