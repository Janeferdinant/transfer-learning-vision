# ruff: noqa: E501
"""Old (shared-CPU) versus new (idle-machine rerun) timing summaries, as Markdown tables.

    python scripts/compare_idle_rerun.py --old results --new results/idle_2026-10-17 --out docs/runs/idle-2026-10-17/comparison.md

Both sides come from ``python -m tlv.analysis`` (summary_*.json); every number is copied from those
files. Reductions are ``1 - mean(time-to-target of approach) / mean(time-to-target of scratch)`` with
95 % percentile bootstrap intervals over seeds (the repository's own analysis code).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def load(d: Path, name: str) -> dict[str, Any] | None:
    p = d / f"summary_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(row: Row, key: str) -> str:
    """Reduction with interval; a leading '>=' marks a censored (lower-bound) value."""
    v = row.get(f"{key}_red_sec")
    if v is None or v != v:
        return "n/a"
    lo, hi = row[f"{key}_red_sec_lo"], row[f"{key}_red_sec_hi"]
    cens = row.get(f"{key}_base_reached", row.get("primary_base_reached", 1)) == 0 or row.get(f"{key}_reached", 1) < row.get("n_runs", 0)
    mark = ">= " if cens else ""
    return f"{mark}{v * 100:.0f}% ({lo * 100:.0f} to {hi * 100:.0f})"


def secs(row: Row | None, key: str) -> str:
    if row is None or row.get(f"{key}_sec") is None or row[f"{key}_sec"] != row[f"{key}_sec"]:
        return "n/a"
    reached = row.get(f"{key}_reached")
    mark = ">= " if reached is not None and reached < row.get("n_runs", 0) else ""
    return f"{mark}{row[f'{key}_sec']:.1f}"


def acc(row: Row | None) -> str:
    return "n/a" if row is None else f"{row['test_acc_mean'] * 100:.1f}"


def verdict(o: Row | None, n: Row | None, key: str = "primary") -> str:
    """Do the old and new reduction intervals overlap, and is the new one still above zero?"""
    if not o or not n or o.get(f"{key}_red_sec") is None or n.get(f"{key}_red_sec") is None:
        return ""
    ol, oh = o[f"{key}_red_sec_lo"], o[f"{key}_red_sec_hi"]
    nl, nh = n[f"{key}_red_sec_lo"], n[f"{key}_red_sec_hi"]
    overlap = max(ol, nl) <= min(oh, nh)
    sign_o = "+" if ol > 0 else "-" if oh < 0 else "0"
    sign_n = "+" if nl > 0 else "-" if nh < 0 else "0"
    return ("overlap" if overlap else "DISJOINT") + ("" if sign_o == sign_n else f"; sign of interval {sign_o}->{sign_n}")


def index(rows: list[Row], keys: tuple[str, ...]) -> dict[tuple[Any, ...], Row]:
    return {tuple(r[k] for k in keys): r for r in rows}


def backbones(old: dict, new: dict) -> list[str]:
    keys = ("dataset", "backbone", "approach")
    o, n = index(old["rows"], keys), index(new["rows"], keys)
    out = ["### Backbones: time to the primary target (95% of scratch's final validation accuracy)", "",
           f"Primary targets, old / new: flowers {old['targets']['flowers'] * 100:.1f}% / {new['targets']['flowers'] * 100:.1f}%, "
           f"EuroSAT {old['targets']['eurosat'] * 100:.1f}% / {new['targets']['eurosat'] * 100:.1f}%.", "",
           "| Dataset | Backbone | Approach | Time, s: old | new | Reduction vs scratch: old (95% CI) | new (95% CI) | Intervals | Test acc %: old | new | Mean bg CPU share: old | new |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(n, key=lambda t: (t[0] != "flowers", t[1] != "-", t[1], t[2] != "scratch", t[2])):
        a, b = o.get(k), n[k]
        out.append(f"| {k[0]} | {k[1] if k[1] != '-' else 'small CNN'} | {k[2]} | {secs(a, 'primary')} | {secs(b, 'primary')} | "
                   f"{pct(a, 'primary') if a and k[2] != 'scratch' else 'baseline'} | {pct(b, 'primary') if k[2] != 'scratch' else 'baseline'} | "
                   f"{verdict(a, b) if k[2] != 'scratch' else ''} | {acc(a)} | {acc(b)} | "
                   f"{a['bg_cpu_mean'] if a else float('nan'):.2f} | {b['bg_cpu_mean']:.2f} |")
    out += ["", "### Backbones: fixed 60% target (where the saving disappears)", "",
            "| Dataset | Backbone | Approach | Reduction: old (95% CI) | new (95% CI) | Intervals |", "|---|---|---|---|---|---|"]
    for k in sorted(n, key=lambda t: (t[0] != "flowers", t[1], t[2])):
        if k[2] == "scratch":
            continue
        a, b = o.get(k), n[k]
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {pct(a, 'fixed_0.60') if a else 'n/a'} | {pct(b, 'fixed_0.60')} | {verdict(a, b, 'fixed_0.60')} |")
    return out


def data_eff(old: dict, new: dict) -> list[str]:
    keys = ("dataset", "approach", "train_per_class")
    o, n = index(old["rows"], keys), index(new["rows"], keys)
    out = ["### Data efficiency: time to the primary target and to 50%", "",
           "| Dataset | Images/class | Approach | Primary: reached (old / new) | Time, s: old | new | Reduction: old (95% CI) | new (95% CI) | 50% reduction: old | new | Test acc %: old | new |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(n, key=lambda t: (t[0] != "flowers", t[2] == -1, t[2], t[1] != "scratch", t[1])):
        a, b = o.get(k), n[k]
        base = k[1] == "scratch"
        reach = f"{a['primary_reached'] if a else '?'}/{a['n_runs'] if a else '?'} / {b['primary_reached']}/{b['n_runs']}"
        out.append(f"| {k[0]} | {'all' if k[2] == -1 else k[2]} | {k[1]} | {reach} | {secs(a, 'primary')} | {secs(b, 'primary')} | "
                   f"{'baseline' if base else (pct(a, 'primary') if a else 'n/a')} | {'baseline' if base else pct(b, 'primary')} | "
                   f"{'' if base else (pct(a, 'fixed_0.50') if a else 'n/a')} | {'' if base else pct(b, 'fixed_0.50')} | {acc(a)} | {acc(b)} |")
    return out


def ablations(old: dict, new: dict) -> list[str]:
    out = ["### Ablations: time to the primary target (3 seeds, wide intervals)", ""]
    for name, factor in (("depth", "unfreeze_blocks"), ("lr", "learning_rate"), ("size", "image_size")):
        if name not in new["rows"]:
            continue
        keys = ("dataset", "approach", factor)
        o = index(old["rows"].get(name, []), keys)
        n = index(new["rows"][name], keys)
        out += [f"#### {name}", "", f"| Dataset | Approach | {factor} | Reached: old / new | Time, s: old | new | Test acc %: old | new |", "|---|---|---|---|---|---|---|---|"]
        for k in sorted(n, key=lambda t: (t[0] != "flowers", t[1], t[2] == -1, t[2])):
            a, b = o.get(k), n[k]
            reach = f"{a['primary_reached'] if a else '?'}/{a['n_runs'] if a else '?'} / {b['primary_reached']}/{b['n_runs']}"
            out.append(f"| {k[0]} | {k[1]} | {'all' if k[2] == -1 else k[2]} | {reach} | {secs(a, 'primary')} | {secs(b, 'primary')} | {acc(a)} | {acc(b)} |")
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="results")
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old_dir, new_dir = Path(a.old), Path(a.new)
    lines = ["# Old (shared CPU) versus new (idle-machine rerun) timings", "",
             "Old: `results/` (run 2026-10-14/15, other jobs on the machine). New: `" + a.new + "` (rerun 2026-10-17). "
             "Same seeds, settings and code; new times are cumulative wall-clock seconds on this machine. "
             "'Intervals' says whether the old and new 95% intervals of the reduction overlap, and flags a change in the sign of the interval.", ""]
    for name, fn in (("backbones", backbones), ("data_efficiency", data_eff), ("ablations", ablations)):
        o, n = load(old_dir, name), load(new_dir, name)
        if o and n:
            lines += fn(o, n) + [""]
    Path(a.out).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
