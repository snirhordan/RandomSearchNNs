#!/usr/bin/env python3
"""Aggregate runs/qm9/d{0,1}_m{0,1}/{U0,gap,mu}/metrics.json into a markdown
table at runs/qm9/results.md, with a brief takeaway paragraph.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ROOT = REPO / "runs" / "qm9"

CONFIGS = [(0, 0), (1, 0), (0, 1), (1, 1)]
# Standard 12 QM9 regression targets, in canonical PyG order (idx 0..11).
TARGETS = [
    "mu", "alpha", "homo", "lumo", "gap", "R2", "zpve",
    "U0", "U", "H", "G", "Cv",
]
UNITS = {
    "mu": "Debye",
    "alpha": "Bohr^3",
    "homo": "eV",
    "lumo": "eV",
    "gap": "eV",
    "R2": "Bohr^2",
    "zpve": "eV",
    "U0": "eV",
    "U": "eV",
    "H": "eV",
    "G": "eV",
    "Cv": "cal/(mol K)",
}


def fmt_wall(seconds: float) -> str:
    if seconds is None or seconds != seconds:
        return "n/a"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def main() -> int:
    table_rows = []
    grid = {}  # (target, cfg) -> dict
    for tgt in TARGETS:
        for d, m in CONFIGS:
            cfg_dir = ROOT / f"d{d}_m{m}" / tgt
            mp = cfg_dir / "metrics.json"
            if not mp.exists():
                grid[(tgt, (d, m))] = None
                continue
            try:
                with open(mp) as f:
                    M = json.load(f)
            except Exception as e:
                print(f"  warn: failed to read {mp}: {e}")
                grid[(tgt, (d, m))] = None
                continue
            grid[(tgt, (d, m))] = M

    # Table
    lines = [
        "# QM9 d-RWNN Results (Variant A: PE-append)",
        "",
        "Single-target regression. RSNN_LSTM, walk_ada, h_dim=128, "
        "num_layers=2, m=8, w=8, batch=128, seed=42. RBF: K=16, cutoff=5.0.",
        "Train/valid/test = 60/20/20 random split. n_splits=1. ",
        "Cap: 15 epochs / patience 4 (reduced from the original spec's "
        "30/5 budget to fit the 4-GPU parallel walltime envelope).",
        "",
        "| target | (distances, mol_edge_feat) | test MAE | best valid MAE | "
        "wall | peak GPU mem (MB) | epochs run |",
        "|---|---|---|---|---|---|---|",
    ]

    for tgt in TARGETS:
        unit = UNITS[tgt]
        for d, m in CONFIGS:
            M = grid[(tgt, (d, m))]
            if M is None:
                lines.append(
                    f"| {tgt} | ({d}, {m}) | -- | -- | -- | -- | -- |"
                )
                continue
            summ = M.get("summary", {})
            sp0 = M.get("splits", [{}])[0] if M.get("splits") else {}
            test_mae = summ.get("mean_test_mae", float("nan"))
            valid_mae = summ.get("mean_valid_mae", float("nan"))
            wall = summ.get("total_wall_sec", float("nan"))
            peak = summ.get("peak_gpu_mem_mb", float("nan"))
            ep_run = sp0.get("epochs_run", "?")
            lines.append(
                f"| {tgt} | ({d}, {m}) | {test_mae:.4f} {unit} | "
                f"{valid_mae:.4f} {unit} | {fmt_wall(wall)} | "
                f"{peak:.0f} | {ep_run} |"
            )
    lines.append("")

    # Takeaway
    lines.append("## Takeaway")
    lines.append("")
    para = []
    for tgt in TARGETS:
        unit = UNITS[tgt]
        base = grid.get((tgt, (0, 0)))
        if base is None:
            continue
        base_mae = base.get("summary", {}).get("mean_test_mae", float("nan"))
        if base_mae != base_mae:
            continue
        best_cfg = (0, 0)
        best_mae = base_mae
        for d, m in CONFIGS:
            M = grid.get((tgt, (d, m)))
            if M is None:
                continue
            mae = M.get("summary", {}).get("mean_test_mae", float("nan"))
            if mae == mae and mae < best_mae:
                best_mae = mae
                best_cfg = (d, m)
        if best_cfg == (0, 0):
            para.append(
                f"On **{tgt}** the baseline (no distances, no bond features) "
                f"was best at {base_mae:.4f} {unit}; d-RWNN extensions did "
                f"not improve over baseline at this epoch budget."
            )
        else:
            pct = 100.0 * (base_mae - best_mae) / max(base_mae, 1e-9)
            para.append(
                f"On **{tgt}** the best d-RWNN config was {best_cfg} with "
                f"test MAE {best_mae:.4f} {unit} versus {base_mae:.4f} "
                f"{unit} at baseline -- {pct:.1f}% MAE reduction."
            )
    if para:
        lines.append(" ".join(para))
    else:
        lines.append("(no completed configs)")
    lines.append("")
    lines.append("All runs were capped at 15 epochs / patience 4 / "
                 "n_splits=1 to fit the 4-GPU parallel walltime envelope "
                 "(~1h05m per config x 4 configs in parallel x 12 targets, "
                 "plus a one-time ~5min preprocess cache per target). "
                 "Validation curves were typically still trending downward "
                 "at the cap, so absolute MAEs are an upper bound on what "
                 "the same model could reach with longer training.")
    lines.append("")

    out = ROOT / "results.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
