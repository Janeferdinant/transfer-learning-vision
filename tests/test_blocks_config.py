import pytest

from tlv.blocks import ALL_BLOCKS, block_starts, first_trainable_index, n_blocks
from tlv.config import MAX_EPOCHS, RunConfig, steps_per_epoch

MNV2 = (["input", "Conv1", "bn_Conv1", "expanded_conv_depthwise"]
        + [f"block_{b}_{part}" for b in range(1, 5) for part in ("expand", "expand_BN", "project")]
        + ["Conv_1", "out_relu"])
RESNET = ["conv1_conv"] + [f"conv{s}_block{b}_{p}" for s, nb in ((2, 2), (3, 2)) for b in range(1, nb + 1)
                           for p in ("1_conv", "add", "out")]


def test_block_starts_in_order_and_unique():
    s = block_starts(MNV2, r"^(block_\d+)_")
    assert [MNV2[i] for i in s] == ["block_1_expand", "block_2_expand", "block_3_expand", "block_4_expand"]


def test_last_k_blocks_include_trailing_layers():
    i = first_trainable_index(MNV2, "mobilenetv2", 2)
    assert MNV2[i] == "block_3_expand"
    assert MNV2[i:][-2:] == ["Conv_1", "out_relu"]  # final conv after the last block is unfrozen too


def test_k_larger_than_blocks_is_capped_and_all_starts_at_zero():
    assert MNV2[first_trainable_index(MNV2, "mobilenetv2", 99)] == "block_1_expand"
    assert first_trainable_index(MNV2, "mobilenetv2", ALL_BLOCKS) == 0


def test_resnet_blocks_group_by_stage_and_block():
    assert n_blocks(RESNET, "resnet50") == 4
    assert RESNET[first_trainable_index(RESNET, "resnet50", 1)] == "conv3_block2_1_conv"


def test_bad_inputs():
    with pytest.raises(ValueError):
        first_trainable_index(MNV2, "vgg", 1)
    with pytest.raises(ValueError):
        first_trainable_index(MNV2, "mobilenetv2", 0)
    with pytest.raises(ValueError):
        first_trainable_index(["a", "b"], "mobilenetv2", 1)


def test_run_id_is_stable_and_ignores_irrelevant_fields():
    a = RunConfig(approach="scratch", backbone="resnet50", unfreeze_blocks=8)
    b = RunConfig(approach="scratch", backbone="mobilenetv2", unfreeze_blocks=2)
    assert a.run_id() == b.run_id()
    f1 = RunConfig(approach="frozen", unfreeze_blocks=1)
    f2 = RunConfig(approach="frozen", unfreeze_blocks=9)
    assert f1.run_id() == f2.run_id()
    assert RunConfig(approach="finetune", unfreeze_blocks=1).run_id() != RunConfig(approach="finetune", unfreeze_blocks=2).run_id()
    assert "seed3" in RunConfig(seed=3).run_id()


def test_effective_epochs_raises_budget_for_tiny_sets_only():
    cfg = RunConfig(epochs=30, min_steps=300)
    assert steps_per_epoch(50) == 2
    assert cfg.effective_epochs(50) == 150
    assert cfg.effective_epochs(720) == 30          # 23 steps x 30 epochs already >= 300
    assert RunConfig(epochs=30).effective_epochs(50) == 30
    assert RunConfig(epochs=30, min_steps=10**6).effective_epochs(50) == MAX_EPOCHS


def test_steps_per_epoch_rejects_empty():
    with pytest.raises(ValueError):
        steps_per_epoch(0)
