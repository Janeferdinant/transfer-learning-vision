"""Model construction tests (need TensorFlow; weights are not downloaded)."""
import pytest

pytest.importorskip("tensorflow")

from tlv.models import ALL_BLOCKS, BACKBONES, build_model, count_params  # noqa: E402


def test_legacy_parameter_counts_are_unchanged():
    """The 2026-10-14 run reported these counts; the generalised builder must reproduce them."""
    assert count_params(build_model("scratch", pretrained=False))[0] == 241_829
    assert count_params(build_model("frozen", pretrained=False))[0] == 6_405
    assert count_params(build_model("finetune", pretrained=False, unfreeze_blocks=4))[0] == 1_669_765


def test_trainable_parameters_grow_with_depth():
    counts = [count_params(build_model("finetune", pretrained=False, unfreeze_blocks=k))[0]
              for k in (1, 2, 4, 8, ALL_BLOCKS)]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    frozen = count_params(build_model("frozen", pretrained=False))[0]
    assert frozen < counts[0]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_backbone_builds_and_predicts(backbone):
    import numpy as np

    m = build_model("finetune", 64, 3, backbone=backbone, unfreeze_blocks=1, pretrained=False)
    p = m.predict(np.zeros((2, 64, 64, 3), "float32"), verbose=0)
    assert p.shape == (2, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-4)
    assert 0 < count_params(m)[0] < count_params(m)[1]


def test_batch_norm_stays_frozen_in_finetune():
    import keras

    m = build_model("finetune", 64, 3, pretrained=False, unfreeze_blocks=ALL_BLOCKS)
    bb = m.get_layer("backbone")
    assert not any(layer.trainable for layer in bb.layers if isinstance(layer, keras.layers.BatchNormalization))


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        build_model("nope")
    with pytest.raises(ValueError):
        build_model("frozen", backbone="vgg", pretrained=False)
