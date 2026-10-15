"""Static figures (PNG) from the summary JSON files written by ``tlv.analysis``.

    python -m tlv.plots        # writes docs/figures/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"scratch": "#2a78d6", "frozen": "#eb6834", "finetune": "#1baf7a"}
NAMES = {"scratch": "From scratch", "frozen": "Frozen backbone", "finetune": "Fine-tuned"}
MARK = {"scratch": "o", "frozen": "s", "finetune": "D"}
DS_NAMES = {"flowers": "TF Flowers (5 classes)", "eurosat": "EuroSAT RGB (10 classes)"}
BB_NAMES = {"mobilenetv2": "MobileNetV2", "mobilenetv3small": "MobileNetV3Small", "efficientnetb0": "EfficientNetB0",
            "resnet50": "ResNet50", "-": "small CNN"}


def _style(ax: Any) -> None:
    ax.grid(True, color="#e7e6e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def data_efficiency_figure(summary: dict, out: Path) -> None:
    rows = summary["rows"]
    datasets = [d for d in ("flowers", "eurosat") if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.6 * len(datasets)), squeeze=False, dpi=110)
    for i, ds in enumerate(datasets):
        for ap in ("scratch", "frozen", "finetune"):
            rs = sorted((r for r in rows if r["dataset"] == ds and r["approach"] == ap), key=lambda r: r["n_train"])
            if not rs:
                continue
            x = [r["n_train"] / (10 if ds == "eurosat" else 5) for r in rs]  # images per class
            ax = axes[i][0]
            ax.plot(x, [r["test_acc_mean"] * 100 for r in rs], color=COLORS[ap], marker=MARK[ap], label=NAMES[ap])
            ax.fill_between(x, [r["test_acc_lo"] * 100 for r in rs], [r["test_acc_hi"] * 100 for r in rs],
                            color=COLORS[ap], alpha=0.18, linewidth=0)
            ax = axes[i][1]
            y = [r["primary_sec"] for r in rs]
            ax.plot(x, y, color=COLORS[ap], marker=MARK[ap], label=NAMES[ap])
            ax.fill_between(x, [r["primary_sec_lo"] for r in rs], [r["primary_sec_hi"] for r in rs],
                            color=COLORS[ap], alpha=0.18, linewidth=0)
            for xi, yi, r in zip(x, y, rs, strict=True):  # hollow marker: some runs never reached the target
                if r["primary_reached"] < r["n_runs"]:
                    ax.plot([xi], [yi], marker=MARK[ap], mfc="white", mec=COLORS[ap], ms=8, ls="none")
        a0, a1 = axes[i]
        a0.set_title(f"{DS_NAMES[ds]}: test accuracy", fontsize=10)
        a0.set_ylabel("Test accuracy (%)")
        a1.set_title(f"{DS_NAMES[ds]}: time to primary target", fontsize=10)
        a1.set_ylabel("Seconds (hollow = not all runs reached it)")
        for ax in (a0, a1):
            ax.set_xscale("log")
            ax.set_xlabel("Training images per class (log scale)")
            ticks = sorted({r["n_train"] / (10 if ds == "eurosat" else 5) for r in rows if r["dataset"] == ds})
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.minorticks_off()
            _style(ax)
        a0.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def backbones_figure(summary: dict, out: Path) -> None:
    rows = summary["rows"]
    datasets = [d for d in ("flowers", "eurosat") if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.6 * len(datasets), 4.4), squeeze=False, dpi=110)
    for ax, ds in zip(axes[0], datasets, strict=True):
        for r in (r for r in rows if r["dataset"] == ds):
            ap = r["approach"]
            acc = r["test_acc_mean"] * 100
            ax.errorbar(r["steady_epoch_s_mean"], acc,
                        yerr=[[acc - r["test_acc_lo"] * 100], [r["test_acc_hi"] * 100 - acc]],
                        color=COLORS[ap], marker=MARK[ap], ms=7, capsize=2, linestyle="none")
            ax.annotate(BB_NAMES[r["backbone"]] if ap != "scratch" else "small CNN",
                        (r["steady_epoch_s_mean"], acc),
                        textcoords="offset points", xytext=(5, -9), fontsize=6.5, color="#52514e")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([1, 2, 3, 4, 6])
        ax.set_xticklabels(["1", "2", "3", "4", "6"])
        ax.set_xlabel("Steady-state epoch time, s (log scale; 720 training images)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(DS_NAMES[ds], fontsize=10)
        _style(ax)
    handles = [plt.Line2D([], [], color=COLORS[a], marker=MARK[a], linestyle="none", label=NAMES[a]) for a in COLORS]
    axes[0][0].legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def ablation_figure(summary: dict, out: Path) -> None:
    abl = summary["rows"]
    spec = [("depth", "unfreeze_blocks", "Blocks fine-tuned (last k; all = whole backbone)"),
            ("lr", "learning_rate", "Learning rate"), ("size", "image_size", "Image size (px)")]
    spec = [s for s in spec if s[0] in abl]
    datasets = sorted({r["dataset"] for s in spec for r in abl[s[0]]})
    fig, axes = plt.subplots(len(datasets), len(spec), figsize=(4.4 * len(spec), 3.4 * len(datasets)),
                             squeeze=False, dpi=110)
    legend_done = False
    for i, ds in enumerate(datasets):
        for j, (name, factor, label) in enumerate(spec):
            ax = axes[i][j]
            rs = [r for r in abl[name] if r["dataset"] == ds]
            if not rs:
                ax.set_visible(False)  # this ablation was run on the other dataset only
                continue
            for ap in ("scratch", "frozen", "finetune"):
                pts = sorted(((r[factor] if r[factor] != -1 else 99, r) for r in rs if r["approach"] == ap), key=lambda t: t[0])
                if not pts:
                    continue
                xs = list(range(len(pts))) if name == "depth" else [p[0] for p in pts]
                ax.plot(xs, [p[1]["test_acc_mean"] * 100 for p in pts], color=COLORS[ap], marker=MARK[ap], label=NAMES[ap])
                ax.fill_between(xs, [p[1]["test_acc_lo"] * 100 for p in pts], [p[1]["test_acc_hi"] * 100 for p in pts],
                                color=COLORS[ap], alpha=0.18, linewidth=0)
            if name == "depth" and rs:
                ticks = sorted({r[factor] if r[factor] != -1 else 99 for r in rs})
                ax.set_xticks(range(len(ticks)))
                ax.set_xticklabels(["all" if t == 99 else str(t) for t in ticks])
            if name == "lr":
                ax.set_xscale("log")
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Test accuracy (%)", fontsize=8)
            ax.set_title(f"{DS_NAMES.get(ds, ds)}", fontsize=9)
            _style(ax)
            if len({r["approach"] for r in rs}) == 3 and not legend_done:
                ax.legend(frameon=False, fontsize=7)
                legend_done = True
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args(argv)
    res, out = Path(a.results), Path(a.out)
    done = []
    for name, fn, fig in (("summary_data_efficiency", data_efficiency_figure, "data_efficiency.png"),
                          ("summary_backbones", backbones_figure, "backbones.png"),
                          ("summary_ablations", ablation_figure, "ablations.png")):
        f = res / f"{name}.json"
        if f.exists():
            fn(_load(f), out / fig)
            done.append(fig)
    print("wrote", ", ".join(done) or "nothing")


if __name__ == "__main__":
    main()
