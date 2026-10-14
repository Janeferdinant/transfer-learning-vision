"""Run one experiment suite (backbones, data-efficiency, depth, lr, image-size, pilot).

    python -m tlv.suite --suite data-efficiency --dataset flowers --seeds 0 1 2 3 4 --out results/data_efficiency_flowers

Resumable: run ids already present in ``<out>/per_run.csv`` are skipped. Each run is timed as
described in ``tlv.runner``. The pilot suite also writes the chosen learning rates to
``configs/lr_choices.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import hardware
from .data import DATASETS, build_cache, load_splits
from .grids import BACKBONES, SUITES, make_grid, pick_learning_rates

LR_FILE = Path("configs/lr_choices.json")


def load_lr_table(path: Path = LR_FILE) -> dict[str, float]:
    return json.loads(path.read_text()) if path.exists() else {}


def _append(path: Path, rows: list[dict]) -> None:
    """Append rows; if the new rows carry columns the file does not have yet, rewrite it with the union."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        if set(new.columns) - set(old.columns):
            pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
            return
        new = new.reindex(columns=old.columns)
    new.to_csv(path, mode="a", header=not path.exists(), index=False)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--dataset", default="flowers", choices=sorted(DATASETS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    p.add_argument("--n-per-class", type=int, default=None, help="images per class (default: dataset setting)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--lr-table", default=str(LR_FILE))
    p.add_argument("--no-quiet-wait", action="store_true",
                   help="do not wait for a quiet machine before each run (always set for the accuracy-only pilot)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    import tensorflow as tf

    from .runner import run_config, warm_up

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = load_lr_table(Path(a.lr_table))
    npc = a.n_per_class or DATASETS[a.dataset].n_per_class
    grid = make_grid(a.suite, a.dataset, a.seeds, table, a.backbones, a.epochs, train_cap=int(0.6 * npc))
    runs_csv, epochs_csv = out / "per_run.csv", out / "per_epoch.csv"
    done = set(pd.read_csv(runs_csv).run_id) if runs_csv.exists() else set()
    todo = [c for c in grid if c.run_id() not in done]
    hw = hardware.describe(tf.__version__)
    (out / "hardware.json").write_text(json.dumps(hw, indent=2))
    print(f"suite={a.suite} dataset={a.dataset}: {len(todo)} runs to do ({len(done)} already done)")
    print("hardware:", hardware.short(hw), flush=True)
    if todo:
        needed: dict[int, set[tuple[str, str]]] = {}
        for c in todo:
            needed.setdefault(c.image_size, set()).add((c.approach, c.backbone if c.backbone != "-" else "mobilenetv2"))
        for size, combos in sorted(needed.items()):
            splits = load_splits(build_cache(Path(a.data_dir), size, npc, dataset=a.dataset))
            n_classes = len(DATASETS[a.dataset].classes)
            warm_up(splits, n_classes, size, sorted(combos))
    cache: dict[int, tuple] = {}
    t_start = time.perf_counter()
    for i, c in enumerate(todo, 1):
        if c.image_size not in cache:
            cache[c.image_size] = load_splits(build_cache(Path(a.data_dir), c.image_size, npc,
                                                          dataset=a.dataset))
        t0 = time.perf_counter()
        for attempt in range(1, 6):  # the desktop is shared: retry a run that hits a memory shortage
            try:
                ep_rows, run_row = run_config(c, cache[c.image_size], len(DATASETS[a.dataset].classes),
                                              wait_quiet=not (a.no_quiet_wait or a.suite == "pilot"))
                break
            except (MemoryError, tf.errors.ResourceExhaustedError) as exc:
                print(f"  memory shortage on attempt {attempt} ({type(exc).__name__}); retrying in 60 s", flush=True)
                if attempt == 5:
                    raise
                time.sleep(60)
        _append(epochs_csv, ep_rows)
        _append(runs_csv, [run_row])
        print(f"[{i}/{len(todo)}] {c.run_id()} train={run_row['total_seconds']:.1f}s "
              f"epochs={run_row['epochs_run']} val={run_row['final_val_accuracy']:.3f} "
              f"test={run_row['test_accuracy']:.3f} (wall {time.perf_counter() - t0:.0f}s, "
              f"elapsed {(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    if a.suite == "pilot":
        picks = pick_learning_rates(pd.read_csv(runs_csv).to_dict("records"))
        merged = {**table, **picks}
        Path(a.lr_table).parent.mkdir(parents=True, exist_ok=True)
        Path(a.lr_table).write_text(json.dumps(merged, indent=2, sort_keys=True))
        print("learning rates chosen on the pilot seed:", json.dumps(picks))


if __name__ == "__main__":
    main()
