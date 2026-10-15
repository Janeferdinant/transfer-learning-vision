import numpy as np
import pytest

from tlv.cam_utils import apply_mask, normalise_map, random_maps, top_fraction_mask


def test_normalise_scales_to_unit_peak_and_clips_negatives():
    cam = np.array([[[-1.0, 2.0], [4.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    out = normalise_map(cam)
    assert out[0].max() == 1.0 and out[0].min() == 0.0 and out[0, 0, 1] == 0.5
    assert np.all(out[1] == 0)


def test_top_fraction_selects_exact_count_of_highest():
    rng = np.random.default_rng(0)
    maps = rng.random((3, 10, 10))
    mask = top_fraction_mask(maps, 0.2)
    assert mask.sum(axis=(1, 2)).tolist() == [20, 20, 20]
    for m, mk in zip(maps, mask, strict=True):
        assert m[mk].min() >= m[~mk].max()


def test_top_fraction_validates():
    with pytest.raises(ValueError):
        top_fraction_mask(np.zeros((1, 2, 2)), 1.0)


def test_apply_mask_does_not_modify_input():
    img = np.ones((1, 2, 2, 3), "float32")
    mask = np.array([[[True, False], [False, False]]])
    out = apply_mask(img, mask)
    assert out[0, 0, 0, 0] == 127.5 and out[0, 0, 1, 0] == 1 and img.max() == 1


def test_random_maps_are_seeded():
    assert np.array_equal(random_maps(2, 3, 1), random_maps(2, 3, 1))
    assert not np.array_equal(random_maps(2, 3, 1), random_maps(2, 3, 2))
