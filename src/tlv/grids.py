"""The documented experiment grids (pure Python: no TensorFlow, unit-tested)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .config import RunConfig

APPROACHES = ("scratch", "frozen", "finetune")
BACKBONES = ("mobilenetv2", "mobilenetv3small", "efficientnetb0", "resnet50")
LEGACY_LRS = {"scratch": 3e-4, "frozen": 1e-3, "finetune": 3e-4}  # chosen in the 2026-10-14 run
N_GRID: tuple[int | None, ...] = (10, 25, 50, 100, None)  # training images per class; None = all (144)
DEPTH_GRID: tuple[int, ...] = (1, 2, 4, 8, -1)  # last k blocks; -1 = whole backbone
LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
PILOT_LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
SIZE_GRID: tuple[int, ...] = (64, 96, 128)
SUITES = ("backbones", "data-efficiency", "depth", "lr", "image-size", "pilot")
MIN_STEPS = 200  # data-efficiency suite: epoch budget raised so tiny sets still get >= 200 steps
PILOT_EPOCHS = 8


def lr_key(dataset: str, backbone: str, approach: str) -> str:
    return f"{dataset}|{'-' if approach == 'scratch' else backbone}|{approach}"


def lookup_lr(table: Mapping[str, float], dataset: str, backbone: str, approach: str) -> float:
    """Learning rate for a (dataset, backbone, approach): pilot choice if present, else legacy."""
    return float(table.get(lr_key(dataset, backbone, approach), LEGACY_LRS[approach]))


def _base(approach: str, **kw) -> dict:
    return dict(approach=approach, **kw)


def make_grid(suite: str, dataset: str, seeds: Sequence[int], table: Mapping[str, float] | None = None,
              backbones: Sequence[str] = BACKBONES, epochs: int = 30,
              train_cap: int | None = None) -> list[RunConfig]:
    """Every run of a suite, ordered seed by seed with the configuration order rotated per seed
    so that no configuration is always first or last."""
    table = table or {}
    lr = lambda b, a: lookup_lr(table, dataset, b, a)  # noqa: E731
    cfgs: list[dict] = []
    if suite == "backbones":
        cfgs.append(_base("scratch", backbone="-"))
        for b in backbones:
            cfgs += [_base("frozen", backbone=b), _base("finetune", backbone=b)]
        cfgs = [{**c, "learning_rate": lr(c["backbone"], c["approach"])} for c in cfgs]
    elif suite == "data-efficiency":
        # sizes at or above the number of available training images per class collapse to "all"
        sizes = [n for n in N_GRID if n is None or train_cap is None or n < train_cap]
        for n in sizes:
            for a in APPROACHES:
                cfgs.append(_base(a, backbone="mobilenetv2", train_per_class=n, min_steps=MIN_STEPS,
                                  learning_rate=lr("mobilenetv2", a)))
    elif suite == "depth":
        for k in DEPTH_GRID:
            cfgs.append(_base("finetune", backbone="mobilenetv2", unfreeze_blocks=k,
                              learning_rate=lr("mobilenetv2", "finetune")))
    elif suite == "lr":
        for rate in LR_GRID:
            for a in APPROACHES:
                cfgs.append(_base(a, backbone="mobilenetv2", learning_rate=rate))
    elif suite == "image-size":
        for px in SIZE_GRID:
            for a in APPROACHES:
                cfgs.append(_base(a, backbone="mobilenetv2", image_size=px, learning_rate=lr("mobilenetv2", a)))
    elif suite == "pilot":
        epochs = PILOT_EPOCHS
        legacy = dataset == "flowers"  # flowers + scratch/MobileNetV2 keep the 2026-10-14 choices
        for rate in PILOT_LR_GRID:
            if not legacy:
                cfgs.append(_base("scratch", backbone="-", learning_rate=rate))
            for b in backbones:
                if legacy and b == "mobilenetv2":
                    continue
                cfgs += [_base("frozen", backbone=b, learning_rate=rate),
                         _base("finetune", backbone=b, learning_rate=rate)]
    else:
        raise ValueError(f"unknown suite {suite!r}; choose from {SUITES}")
    out: list[RunConfig] = []
    for i, seed in enumerate(seeds):
        r = i % len(cfgs)
        for c in cfgs[r:] + cfgs[:r]:
            out.append(RunConfig(dataset=dataset, seed=seed, epochs=epochs, **c).normalised())
    return out


def pick_learning_rates(rows: Sequence[Mapping]) -> dict[str, float]:
    """Pilot rule: per (dataset, backbone, approach) choose the learning rate with the highest
    final validation accuracy (ties go to the smaller rate). Rows need the RunConfig fields plus
    ``final_val_accuracy``; with several pilot seeds the accuracies are averaged first."""
    acc: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        key = lr_key(r["dataset"], r["backbone"], r["approach"])
        acc.setdefault((key, float(r["learning_rate"])), []).append(float(r["final_val_accuracy"]))
    best: dict[str, tuple[float, float]] = {}
    for (key, rate), v in acc.items():
        m = sum(v) / len(v)
        if key not in best or m > best[key][0] + 1e-12 or (abs(m - best[key][0]) <= 1e-12 and rate < best[key][1]):
            best[key] = (m, rate)
    return {k: v[1] for k, v in sorted(best.items())}
