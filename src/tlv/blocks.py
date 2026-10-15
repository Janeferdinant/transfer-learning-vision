"""Which layers of a pretrained backbone are unfrozen when fine-tuning "the last k blocks".

Pure Python (layer names only), so it is unit-tested without TensorFlow. A *block* is a
group of layers sharing a name prefix (for example ``block_13_`` in MobileNetV2 or
``conv5_block2_`` in ResNet50). Fine-tuning the last ``k`` blocks unfreezes every layer from
the first layer of the k-th block from the end to the end of the network, which also covers
the final 1x1 convolution that follows the last block in some architectures.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

# One regular expression per backbone; group 1 is the block identifier.
BLOCK_PATTERNS: dict[str, str] = {
    "mobilenetv2": r"^(block_\d+)_",
    "mobilenetv3small": r"^(expanded_conv_\d+)_",
    "efficientnetb0": r"^(block\d[a-z])_",
    "resnet50": r"^(conv\d_block\d)_",
}
ALL_BLOCKS = -1  # sentinel: unfreeze the whole backbone (batch-norm layers still stay frozen)


def block_starts(layer_names: Sequence[str], pattern: str) -> list[int]:
    """Index of the first layer of every block, in network order."""
    rx = re.compile(pattern)
    seen: set[str] = set()
    starts: list[int] = []
    for i, name in enumerate(layer_names):
        m = rx.match(name)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            starts.append(i)
    return starts


def first_trainable_index(layer_names: Sequence[str], backbone: str, k: int) -> int:
    """Index of the first unfrozen layer when fine-tuning the last ``k`` blocks.

    ``k == ALL_BLOCKS`` returns 0; ``k`` larger than the number of blocks is capped at the
    first block (it does not reach the stem).
    """
    if backbone not in BLOCK_PATTERNS:
        raise ValueError(f"unknown backbone {backbone!r}")
    if k == ALL_BLOCKS:
        return 0
    if k < 1:
        raise ValueError("k must be >= 1 (or ALL_BLOCKS)")
    starts = block_starts(layer_names, BLOCK_PATTERNS[backbone])
    if not starts:
        raise ValueError(f"no blocks found for {backbone!r}")
    return starts[max(len(starts) - k, 0)]


def n_blocks(layer_names: Sequence[str], backbone: str) -> int:
    return len(block_starts(layer_names, BLOCK_PATTERNS[backbone]))
