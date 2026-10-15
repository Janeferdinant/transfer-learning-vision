import numpy as np
import pandas as pd

from tlv.analysis import curves_by_run, primary_target, summarise, ttt_arrays


def make(approach, seed, accs, secs_per_epoch, n=-1, backbone="mobilenetv2", dataset="d"):
    rid = f"{dataset}|{backbone}|{approach}|n{n}|seed{seed}"
    ep = pd.DataFrame(dict(run_id=rid, epoch=np.arange(1, len(accs) + 1),
                           seconds=np.cumsum([secs_per_epoch] * len(accs)), val_accuracy=accs))
    run = dict(run_id=rid, dataset=dataset, backbone=backbone, approach=approach, seed=seed, train_per_class=n,
               final_val_accuracy=accs[-1], test_accuracy=accs[-1] - 0.01, total_seconds=ep.seconds.iloc[-1],
               steady_epoch_seconds=secs_per_epoch, first_epoch_seconds=secs_per_epoch, trainable_params=10,
               total_params=20, epochs_run=len(accs), n_train=100)
    return run, ep


def build():
    runs, eps = [], []
    for seed in range(4):
        for ap, accs, sec in (("scratch", [0.2, 0.4, 0.6, 0.8], 2.0), ("frozen", [0.7, 0.8, 0.8, 0.8], 1.0)):
            r, e = make(ap, seed, accs, sec)
            runs.append(r)
            eps.append(e)
    return pd.DataFrame(runs), pd.concat(eps)


def test_primary_target_is_fraction_of_scratch_final():
    runs, _ = build()
    assert abs(primary_target(runs) - 0.95 * 0.8) < 1e-12


def test_ttt_arrays_uses_censoring_budget():
    runs, eps = build()
    c = curves_by_run(eps)
    secs, epochs, reached = ttt_arrays(c, runs[runs.approach == "scratch"].run_id, 0.9)
    assert reached == 0 and np.all(secs == 8.0) and np.all(epochs == 4)
    secs, epochs, reached = ttt_arrays(c, runs[runs.approach == "scratch"].run_id, 0.6)
    assert reached == 4 and np.all(secs == 6.0) and np.all(epochs == 3)


def test_summarise_reduction_vs_scratch():
    runs, eps = build()
    rows = summarise(runs, curves_by_run(eps), ["dataset", "backbone", "approach"], {"t": 0.7}, baseline_on=["dataset"])
    by = {r["approach"]: r for r in rows}
    assert "t_red_sec" not in by["scratch"]
    f = by["frozen"]
    # scratch reaches 0.7 at epoch 4 (8 s); frozen at epoch 1 (1 s): reduction 1 - 1/8
    assert abs(f["t_red_sec"] - 0.875) < 1e-9 and abs(f["t_red_ep"] - 0.75) < 1e-9
    assert f["t_reached"] == 4 and f["t_base_reached"] == 4
    assert abs(f["test_acc_diff_vs_scratch"]) < 1e-9
