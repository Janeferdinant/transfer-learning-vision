"""Run configuration shared by the runner, the suites and the analysis (no TensorFlow)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

BATCH_SIZE = 32
MAX_EPOCHS = 150  # hard cap when the epoch budget is raised for tiny training sets


@dataclass(frozen=True)
class RunConfig:
    dataset: str = "flowers"
    backbone: str = "mobilenetv2"
    approach: str = "scratch"
    seed: int = 0
    epochs: int = 30
    learning_rate: float = 3e-4
    image_size: int = 96
    train_per_class: int | None = None  # None = every training image of the split
    unfreeze_blocks: int = 4  # fine-tune only; -1 = whole backbone
    min_steps: int | None = None  # raise the epoch budget so tiny sets still get this many steps

    def normalised(self) -> RunConfig:
        """Drop fields that do not apply (scratch has no backbone, frozen no depth)."""
        d = asdict(self)
        if self.approach == "scratch":
            d["backbone"], d["unfreeze_blocks"] = "-", 0
        elif self.approach == "frozen":
            d["unfreeze_blocks"] = 0
        return RunConfig(**d)

    def run_id(self) -> str:
        c = self.normalised()
        n = "all" if c.train_per_class is None else str(c.train_per_class)
        return (f"{c.dataset}|{c.backbone}|{c.approach}|k{c.unfreeze_blocks}|n{n}"
                f"|lr{c.learning_rate:g}|px{c.image_size}|seed{c.seed}")

    def effective_epochs(self, n_train: int) -> int:
        """Epoch budget: ``epochs``, raised when ``min_steps`` asks for more optimiser steps."""
        if self.min_steps is None:
            return self.epochs
        steps = steps_per_epoch(n_train)
        return min(max(self.epochs, math.ceil(self.min_steps / steps)), MAX_EPOCHS)


def steps_per_epoch(n_train: int, batch_size: int = BATCH_SIZE) -> int:
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    return math.ceil(n_train / batch_size)
