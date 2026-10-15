"""Summaries of the suite runs: accuracy, time-to-target, cost and bootstrap intervals.

Everything here is pandas/NumPy (no TensorFlow). Time-to-target keeps the definition used since
the first run: cumulative wall-clock time at the end of the first epoch whose validation accuracy
reaches the target; a run that never reaches it is charged its full budget (a lower bound).

    python -m tlv.analysis        # writes results/summary_*.json and .md from the committed CSVs
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .metrics import (
    TimeToTarget,
    bootstrap_ci,
    bootstrap_diff_ci,
    bootstrap_reduction_ci,
    censored_values,
    time_to_target,
)

PRIMARY_FRACTION = 0.95  # primary target = 95% of the from-scratch final validation accuracy
DISTURBED = 0.5  # flag a run when other programs used more than half of all CPU time during its fit
FIXED_TARGETS = (0.5,)  # extra fixed target used for the data-efficiency curves
CONFIG_COLS = ["dataset", "backbone", "approach", "unfreeze_blocks", "train_per_class", "learning_rate", "image_size"]

Curves = dict[str, tuple[np.ndarray, np.ndarray]]


def load_suite(path: Path, max_bg: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs and per-epoch logs of a suite; ``max_bg`` drops runs disturbed by other programs."""
    runs = pd.read_csv(path / "per_run.csv")
    runs["train_per_class"] = runs["train_per_class"].fillna(-1).astype(int)  # -1 = all images
    epochs = pd.read_csv(path / "per_epoch.csv")
    if max_bg is not None:
        runs = runs[runs.bg_cpu_fraction <= max_bg]
        epochs = epochs[epochs.run_id.isin(runs.run_id)]
    return runs, epochs


def curves_by_run(epochs: pd.DataFrame) -> Curves:
    out: Curves = {}
    for rid, g in epochs.groupby("run_id"):
        g = g.sort_values("epoch")
        out[str(rid)] = (g["seconds"].to_numpy(), g["val_accuracy"].to_numpy())
    return out


def ttt_arrays(curves: Curves, run_ids: Iterable[str], target: float) -> tuple[np.ndarray, np.ndarray, int]:
    res: list[TimeToTarget] = []
    budget_s: list[float] = []
    budget_e = 0
    for rid in run_ids:
        t, a = curves[rid]
        res.append(time_to_target(t, a, target))
        budget_s.append(float(t[-1]))
        budget_e = max(budget_e, len(t))
    return censored_values(res, budget_s, budget_e)


def primary_target(runs: pd.DataFrame, fraction: float = PRIMARY_FRACTION) -> float:
    """``fraction`` x mean final validation accuracy of the from-scratch, all-data runs."""
    s = runs[(runs.approach == "scratch") & (runs.train_per_class == -1)]
    if s.empty:
        raise ValueError("no from-scratch all-data runs to define the target")
    return fraction * float(s.final_val_accuracy.mean())


def _ci(v: ArrayLike) -> tuple[float, float]:
    return bootstrap_ci(v, n_boot=10000, seed=0)


def summarise(runs: pd.DataFrame, curves: Curves, by: Sequence[str], targets: dict[str, float],
              baseline_on: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """One flat row per group of runs.

    ``baseline_on``: if given, rows also report the reduction in time and epochs and the
    accuracy difference relative to the ``scratch`` runs that agree with the group on those
    columns (for example ``["dataset"]`` or ``["dataset", "train_per_class"]``).
    """
    rows: list[dict[str, Any]] = []
    for key, g in runs.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(by, key, strict=True))
        row["n_runs"] = len(g)
        for col, name in (("final_val_accuracy", "val_acc"), ("test_accuracy", "test_acc"),
                          ("total_seconds", "total_s"), ("steady_epoch_seconds", "steady_epoch_s"),
                          ("first_epoch_seconds", "first_epoch_s")):
            v = g[col].to_numpy(dtype=float)
            lo, hi = _ci(v)
            row.update({f"{name}_mean": float(v.mean()), f"{name}_lo": lo, f"{name}_hi": hi})
        if "bg_cpu_fraction" in g:
            row["bg_cpu_mean"] = float(g.bg_cpu_fraction.mean())  # CPU share used by other programs during the fits
            row["n_disturbed"] = int((g.bg_cpu_fraction > DISTURBED).sum())
        row["trainable_params"] = int(g.trainable_params.iloc[0])
        row["total_params"] = int(g.total_params.iloc[0])
        row["epochs_run"] = int(g.epochs_run.iloc[0])
        row["n_train"] = int(g.n_train.iloc[0])
        base = None
        if baseline_on is not None and row.get("approach") != "scratch":
            m = runs.approach == "scratch"
            for c in baseline_on:
                m &= runs[c] == row[c]
            base = runs[m]
        if base is not None and not base.empty:
            est, lo, hi = bootstrap_diff_ci(g.test_accuracy, base.test_accuracy)
            row.update(test_acc_diff_vs_scratch=est, test_acc_diff_lo=lo, test_acc_diff_hi=hi)
        for tname, target in targets.items():
            secs, eps, n_reached = ttt_arrays(curves, g.run_id, target)
            slo, shi = _ci(secs)
            elo, ehi = _ci(eps)
            row.update({f"{tname}_reached": int(n_reached), f"{tname}_sec": float(secs.mean()),
                        f"{tname}_sec_lo": slo, f"{tname}_sec_hi": shi, f"{tname}_ep": float(eps.mean()),
                        f"{tname}_ep_lo": elo, f"{tname}_ep_hi": ehi})
            if base is not None and not base.empty:
                bs, be, b_reached = ttt_arrays(curves, base.run_id, target)
                r, rlo, rhi = bootstrap_reduction_ci(bs, secs, seed=0)
                re_, elo2, ehi2 = bootstrap_reduction_ci(be, eps, seed=0)
                row.update({f"{tname}_red_sec": r, f"{tname}_red_sec_lo": rlo, f"{tname}_red_sec_hi": rhi,
                            f"{tname}_red_ep": re_, f"{tname}_red_ep_lo": elo2, f"{tname}_red_ep_hi": ehi2,
                            f"{tname}_base_reached": int(b_reached), f"{tname}_base_n": int(len(base))})
        rows.append(row)
    return rows


# ------------------------------------------------------------------ formatting
def pct(v: float, nd: int = 1) -> str:
    return f"{v * 100:.{nd}f}"


def with_ci(row: dict, stem: str, scale: float = 1.0, nd: int = 1) -> str:
    return f"{row[stem + '_mean'] * scale:.{nd}f} ({row[stem + '_lo'] * scale:.{nd}f} to {row[stem + '_hi'] * scale:.{nd}f})"


def diff_str(row: dict) -> str:
    if "test_acc_diff_vs_scratch" not in row:
        return ""
    e, lo, hi = (row[k] * 100 for k in ("test_acc_diff_vs_scratch", "test_acc_diff_lo", "test_acc_diff_hi"))
    return f"{e:+.1f} ({lo:+.1f} to {hi:+.1f})"


def reach_str(row: dict, t: str) -> str:
    return f"{row[t + '_reached']}/{row['n_runs']}"


def sec_str(row: dict, t: str) -> str:
    ge = ">= " if row[t + "_reached"] < row["n_runs"] else ""
    return f"{ge}{row[t + '_sec']:.1f} ({row[t + '_sec_lo']:.1f} to {row[t + '_sec_hi']:.1f})"


def red_str(row: dict, t: str, key: str = "red_sec") -> str:
    k = f"{t}_{key}"
    if k not in row:
        return "baseline"
    base_cens = row[f"{t}_base_reached"] < row[f"{t}_base_n"]
    cand_cens = row[t + "_reached"] < row["n_runs"]
    tag = ">= " if base_cens and not cand_cens else ("<= " if cand_cens and not base_cens else "")
    s = f"{tag}{row[k] * 100:.0f}% ({row[k + '_lo'] * 100:.0f} to {row[k + '_hi'] * 100:.0f}%)"
    return s + (" (both censored)" if base_cens and cand_cens else "")


def _md_table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out) + "\n"


NAMES = {"scratch": "From scratch", "frozen": "Frozen", "finetune": "Fine-tune"}
BB_NAMES = {"mobilenetv2": "MobileNetV2", "mobilenetv3small": "MobileNetV3Small", "efficientnetb0": "EfficientNetB0",
            "resnet50": "ResNet50", "-": "small CNN"}


def backbones_markdown(rows: list[dict], dataset: str, target: float) -> str:
    rs = [r for r in rows if r["dataset"] == dataset]
    order = {"scratch": 0, "frozen": 1, "finetune": 2}
    bb_order = ["-", "mobilenetv2", "mobilenetv3small", "efficientnetb0", "resnet50"]
    rs.sort(key=lambda r: (bb_order.index(r["backbone"]), order[r["approach"]]))
    body = []
    for r in rs:
        body.append([BB_NAMES[r["backbone"]], NAMES[r["approach"]], f"{r['trainable_params']:,}", f"{r['total_params']:,}",
                     with_ci(r, "test_acc", 100), reach_str(r, "primary"), sec_str(r, "primary"),
                     f"{r['primary_ep']:.1f}", red_str(r, "primary"),
                     f"{r['first_epoch_s_mean']:.1f}", f"{r['steady_epoch_s_mean']:.2f}"])
    head = ["Backbone", "Approach", "Trainable params", "Total params", "Test acc, % (95% CI)", "Reached",
            f"Time to {target * 100:.1f}% val, s (95% CI)", "Epochs", "Time reduction vs scratch",
            "Epoch 1, s", "Steady epoch, s"]
    return f"#### {dataset} (primary target {target * 100:.1f}% validation accuracy)\n\n" + _md_table(head, body)


def extra_target_markdown(rows: list[dict], dataset: str, tname: str, target: float) -> str:
    """Compact table for an additional fixed target (shows how the saving depends on the target)."""
    rs = [r for r in rows if r["dataset"] == dataset]
    order = {"scratch": 0, "frozen": 1, "finetune": 2}
    bb_order = ["-", "mobilenetv2", "mobilenetv3small", "efficientnetb0", "resnet50"]
    rs.sort(key=lambda r: (bb_order.index(r["backbone"]), order[r["approach"]]))
    body = [[BB_NAMES[r["backbone"]], NAMES[r["approach"]], reach_str(r, tname), sec_str(r, tname),
             f"{r[tname + '_ep']:.1f}", red_str(r, tname), red_str(r, tname, "red_ep")] for r in rs]
    head = ["Backbone", "Approach", "Reached", f"Time to {target * 100:.1f}% val, s (95% CI)", "Epochs",
            "Time reduction vs scratch", "Epoch reduction vs scratch"]
    return f"#### {dataset}, fixed target {target * 100:.0f}% validation accuracy\n\n" + _md_table(head, body)


def data_efficiency_markdown(rows: list[dict], dataset: str, targets: dict[str, float]) -> str:
    rs = [r for r in rows if r["dataset"] == dataset]
    order = {"scratch": 0, "frozen": 1, "finetune": 2}
    rs.sort(key=lambda r: (r["n_train"], order[r["approach"]]))
    out = [f"#### {dataset}\n"]
    for t in targets:
        body = []
        for r in rs:
            body.append(["all" if r["train_per_class"] == -1 else str(r["train_per_class"]),
                         NAMES[r["approach"]], str(r["epochs_run"]), with_ci(r, "test_acc", 100),
                         diff_str(r),
                         reach_str(r, t), sec_str(r, t), f"{r[t + '_ep']:.1f}", red_str(r, t)])
        head = ["Train images/class", "Approach", "Epoch budget", "Test acc, % (95% CI)", "Diff vs scratch, pts (95% CI)", "Reached",
                f"Time to {targets[t] * 100:.1f}% val, s (95% CI)", "Epochs", "Time reduction vs scratch"]
        out.append(f"Target `{t}` = {targets[t] * 100:.1f}% validation accuracy\n\n" + _md_table(head, body))
    return "\n".join(out)


def ablation_markdown(rows: list[dict], factor: str, label: str) -> str:
    order = {"scratch": 0, "frozen": 1, "finetune": 2}
    def sort_key(r: dict) -> tuple:
        v = r[factor] if r[factor] != -1 else 10**6
        return (r["dataset"], order[r["approach"]], v)

    rs = sorted(rows, key=sort_key)
    body = []
    for r in rs:
        v = r[factor]
        shown = ("all" if v == -1 else str(v)) if factor == "unfreeze_blocks" else f"{v:g}"
        body.append([r["dataset"], NAMES[r["approach"]], shown, f"{r['trainable_params']:,}", with_ci(r, "val_acc", 100),
                     with_ci(r, "test_acc", 100), reach_str(r, "primary"), sec_str(r, "primary"),
                     f"{r['primary_ep']:.1f}", f"{r['total_s_mean']:.0f}"])
    head = ["Dataset", "Approach", label, "Trainable params", "Final val acc, % (95% CI)", "Test acc, % (95% CI)", "Reached",
            "Time to primary target, s (95% CI)", "Epochs", "Total train, s"]
    return _md_table(head, body)


# ------------------------------------------------------------------ driver
def _write(path: Path, obj: Any, md: str) -> None:
    path.with_suffix(".json").write_text(json.dumps(obj, indent=1))
    path.with_suffix(".md").write_text(md)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-bg", type=float, default=None,
                    help="sensitivity analysis: drop runs whose background CPU share exceeded this value")
    ap.add_argument("--suffix", default="", help="appended to the summary file names, e.g. _lowbg")
    a = ap.parse_args(argv)
    res = Path(a.results)
    targets_by_ds: dict[str, float] = {}
    curves: Curves = {}
    summaries: dict[str, Any] = {}

    md_bb = ["### Backbone comparison (all-data training, 30 epochs)\n"]
    bb_rows: list[dict] = []
    for ds in ("flowers", "eurosat"):
        d = res / f"backbones_{ds}"
        if not (d / "per_run.csv").exists():
            continue
        runs, epochs = load_suite(d, a.max_bg)
        c = curves_by_run(epochs)
        curves.update(c)
        targets_by_ds[ds] = primary_target(runs)
        tg = {"primary": targets_by_ds[ds], "fixed_0.60": 0.6, "fixed_0.80": 0.8}
        rows = summarise(runs, c, ["dataset", "backbone", "approach"], tg, baseline_on=["dataset"])
        bb_rows += rows
        md_bb.append(backbones_markdown(rows, ds, targets_by_ds[ds]))
        md_bb += [extra_target_markdown(rows, ds, t, tg[t]) for t in ("fixed_0.60", "fixed_0.80")]
    if bb_rows:
        _write(res / f"summary_backbones{a.suffix}", dict(targets=targets_by_ds, rows=bb_rows), "\n".join(md_bb))
        summaries["backbones"] = len(bb_rows)

    de_rows: list[dict] = []
    md_de = ["### Data efficiency (MobileNetV2; training images per class vary, validation and test sets fixed)\n"]
    for ds in ("flowers", "eurosat"):
        d = res / f"data_efficiency_{ds}"
        if not (d / "per_run.csv").exists() or ds not in targets_by_ds:
            continue
        runs, epochs = load_suite(d, a.max_bg)
        c = curves_by_run(epochs)
        curves.update(c)
        tg = {"primary": targets_by_ds[ds], "fixed_0.50": 0.5}
        rows = summarise(runs, c, ["dataset", "approach", "train_per_class"], tg, baseline_on=["dataset", "train_per_class"])
        de_rows += rows
        md_de.append(data_efficiency_markdown(rows, ds, tg))
    if de_rows:
        _write(res / f"summary_data_efficiency{a.suffix}", dict(targets=targets_by_ds, rows=de_rows), "\n".join(md_de))
        summaries["data_efficiency"] = len(de_rows)

    abl = {"depth": ("unfreeze_blocks", "Blocks fine-tuned"), "lr": ("learning_rate", "Learning rate"),
           "size": ("image_size", "Image size, px")}
    md_ab = ["### Ablations (MobileNetV2; 3 seeds each)\n"]
    ab_rows: dict[str, list[dict]] = {}
    for name, (factor, label) in abl.items():
        rows_all: list[dict] = []
        for ds in ("flowers", "eurosat"):
            d = res / f"ablation_{name}_{ds}"
            if not (d / "per_run.csv").exists() or ds not in targets_by_ds:
                continue
            runs, epochs = load_suite(d, a.max_bg)
            c = curves_by_run(epochs)
            curves.update(c)
            rows_all += summarise(runs, c, ["dataset", "approach", factor], {"primary": targets_by_ds[ds]})
        if rows_all:
            ab_rows[name] = rows_all
            tg_txt = ", ".join(f"{k} {v * 100:.1f}%" for k, v in targets_by_ds.items())
            md_ab.append(f"#### {label}\n\nPrimary targets used: {tg_txt}.\n\n" + ablation_markdown(rows_all, factor, label))
    if ab_rows:
        _write(res / f"summary_ablations{a.suffix}", dict(targets=targets_by_ds, rows=ab_rows), "\n".join(md_ab))
        summaries["ablations"] = {k: len(v) for k, v in ab_rows.items()}
    print("wrote summaries:", json.dumps(summaries))


if __name__ == "__main__":
    main()
