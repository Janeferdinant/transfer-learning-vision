import pytest

from tlv.grids import (
    BACKBONES,
    LEGACY_LRS,
    lookup_lr,
    lr_key,
    make_grid,
    pick_learning_rates,
)


def test_lookup_falls_back_to_legacy_and_uses_table():
    assert lookup_lr({}, "flowers", "resnet50", "frozen") == LEGACY_LRS["frozen"]
    t = {lr_key("eurosat", "resnet50", "frozen"): 1e-4}
    assert lookup_lr(t, "eurosat", "resnet50", "frozen") == 1e-4
    assert lookup_lr(t, "eurosat", "mobilenetv2", "frozen") == LEGACY_LRS["frozen"]
    assert lr_key("eurosat", "anything", "scratch") == "eurosat|-|scratch"


def test_backbones_grid_size_and_unique_ids():
    g = make_grid("backbones", "flowers", [0, 1])
    assert len(g) == 2 * (1 + 2 * len(BACKBONES))
    assert len({c.run_id() for c in g}) == len(g)
    assert {c.seed for c in g} == {0, 1}


def test_order_rotates_between_seeds():
    g = make_grid("backbones", "flowers", [0, 1])
    n = len(g) // 2
    assert g[0].run_id().replace("seed0", "") != g[n].run_id().replace("seed1", "")


def test_data_efficiency_grid_has_all_sizes_and_budget_rule():
    g = make_grid("data-efficiency", "eurosat", [0])
    assert {c.train_per_class for c in g} == {10, 25, 50, 100, None}
    assert all(c.min_steps == 200 for c in g)
    assert len(g) == 15


def test_depth_lr_size_pilot_grids():
    assert [c.unfreeze_blocks for c in make_grid("depth", "flowers", [0])] == [1, 2, 4, 8, -1]
    assert len(make_grid("lr", "flowers", [0])) == 12
    assert {c.image_size for c in make_grid("image-size", "flowers", [0])} == {64, 96, 128}
    p = make_grid("pilot", "eurosat", [100])
    assert all(c.epochs == 8 and c.seed == 100 for c in p)
    assert len(p) == 3 * (1 + 2 * len(BACKBONES))
    pf = make_grid("pilot", "flowers", [100])
    assert not any(c.approach == "scratch" or c.backbone == "mobilenetv2" for c in pf)


def test_unknown_suite_raises():
    with pytest.raises(ValueError):
        make_grid("nope", "flowers", [0])


def test_pick_learning_rates_takes_best_and_breaks_ties_low():
    rows = [
        dict(dataset="d", backbone="b", approach="frozen", learning_rate=1e-3, final_val_accuracy=0.7),
        dict(dataset="d", backbone="b", approach="frozen", learning_rate=1e-4, final_val_accuracy=0.8),
        dict(dataset="d", backbone="b", approach="finetune", learning_rate=1e-3, final_val_accuracy=0.8),
        dict(dataset="d", backbone="b", approach="finetune", learning_rate=3e-4, final_val_accuracy=0.8),
        dict(dataset="d", backbone="b", approach="scratch", learning_rate=3e-4, final_val_accuracy=0.5),
    ]
    got = pick_learning_rates(rows)
    assert got["d|b|frozen"] == 1e-4
    assert got["d|b|finetune"] == 3e-4
    assert got["d|-|scratch"] == 3e-4


def test_data_efficiency_grid_respects_train_cap():
    g = make_grid("data-efficiency", "eurosat", [0], train_cap=72)
    assert {c.train_per_class for c in g} == {10, 25, 50, None}
