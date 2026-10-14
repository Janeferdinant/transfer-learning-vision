import numpy as np
import pytest

from tlv.data import dedupe_by_hash, stratified_split, subsample_per_class


def make_labels(per_class=100, classes=5):
    return np.repeat(np.arange(classes), per_class)


def test_split_is_deterministic():
    y = make_labels()
    a = stratified_split(y, (0.6, 0.2, 0.2), seed=7)
    b = stratified_split(y, (0.6, 0.2, 0.2), seed=7)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)


def test_split_changes_with_seed():
    y = make_labels()
    a = stratified_split(y, (0.6, 0.2, 0.2), seed=1)
    b = stratified_split(y, (0.6, 0.2, 0.2), seed=2)
    assert not np.array_equal(a[0], b[0])


def test_no_leakage_and_full_coverage():
    y = make_labels(per_class=37)
    tr, va, te = stratified_split(y, (0.6, 0.2, 0.2), seed=0)
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    assert s_tr | s_va | s_te == set(range(len(y)))
    assert len(tr) + len(va) + len(te) == len(y)


def test_split_is_stratified():
    y = make_labels(per_class=100)
    tr, va, te = stratified_split(y, (0.6, 0.2, 0.2), seed=0)
    for idx, n in ((tr, 60), (va, 20), (te, 20)):
        counts = np.bincount(y[idx], minlength=5)
        assert set(counts) == {n}


def test_split_handles_uneven_classes():
    y = np.array([0] * 10 + [1] * 7 + [2] * 3)
    tr, va, te = stratified_split(y, (0.6, 0.2, 0.2), seed=0)
    assert len(set(tr) | set(va) | set(te)) == len(y)
    assert all(len(np.unique(y[i])) == 3 for i in (tr, va, te))


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        stratified_split(make_labels(), (0.5, 0.2, 0.2), seed=0)


def test_subsample_per_class_deterministic_and_balanced():
    y = np.array([0] * 50 + [1] * 30 + [2] * 80)
    a = subsample_per_class(y, 20, seed=3)
    b = subsample_per_class(y, 20, seed=3)
    assert np.array_equal(a, b)
    assert list(np.bincount(y[a])) == [20, 20, 20]
    assert len(set(a)) == len(a)


def test_subsample_requires_enough_images():
    with pytest.raises(ValueError):
        subsample_per_class(np.array([0, 0, 1]), 5, seed=0)


def test_dedupe_by_hash_removes_identical_images_keeps_first():
    imgs = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    imgs[2] += 9
    imgs[3] = imgs[2]
    keep = dedupe_by_hash(imgs)
    assert list(keep) == [0, 2]  # 1 duplicates 0, 3 duplicates 2


def test_select_per_class_is_balanced_seeded_and_nested():
    from tlv.data import select_per_class

    y = np.repeat(np.arange(4), 30)
    a = select_per_class(y, 10, seed=1)
    assert np.array_equal(a, select_per_class(y, 10, seed=1))
    assert not np.array_equal(a, select_per_class(y, 10, seed=2))
    assert np.bincount(y[a]).tolist() == [10, 10, 10, 10]
    assert set(a) <= set(select_per_class(y, 25, seed=1))  # nested for a fixed seed
    assert np.array_equal(select_per_class(y, None, seed=1), np.arange(len(y)))


def test_select_per_class_rejects_too_many():
    from tlv.data import select_per_class

    with pytest.raises(ValueError):
        select_per_class(np.repeat(np.arange(2), 5), 6, seed=0)


def test_sha256_verification(tmp_path):
    from tlv.data import verify_file

    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    verify_file(f, sha256=good)
    with pytest.raises(ValueError):
        verify_file(f, sha256="0" * 64)
