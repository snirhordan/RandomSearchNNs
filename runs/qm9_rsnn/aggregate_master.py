#!/usr/bin/env python3
"""Aggregate master QM9 results: d-RWNN 4 configs + RSNN 4 m values.

Reads:
  runs/qm9/d{0,1}_m{0,1}/{target}/metrics.json     (d-RWNN, single split)
  runs/qm9_rsnn/m{1,4,8,16}/{target}/split{0,1,2}/metrics.json   (RSNN, 3 splits)

Writes:
  runs/qm9_rsnn/results_master.md
"""
import json
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QM9_RWNN = REPO / "runs" / "qm9"
QM9_RSNN = REPO / "runs" / "qm9_rsnn"

TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "R2", "zpve", "U0", "U", "H", "G", "Cv"]
UNITS = {
    "mu": "D", "alpha": "Bohr^3", "homo": "eV", "lumo": "eV", "gap": "eV",
    "R2": "Bohr^2", "zpve": "eV", "U0": "eV", "U": "eV", "H": "eV", "G": "eV",
    "Cv": "cal/(mol K)",
}
RWNN_CONFIGS = [
    ("d-RWNN (d=0, b=0)", "d0_m0"),
    ("d-RWNN (d=1, b=0)", "d1_m0"),
    ("d-RWNN (d=0, b=1)", "d0_m1"),
    ("d-RWNN (d=1, b=1)", "d1_m1"),
]
RSNN_MS = [1, 4, 8, 16]


def read_test_mae(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    # d-RWNN format
    if "summary" in data and "mean_test_mae" in data["summary"]:
        return float(data["summary"]["mean_test_mae"])
    # RSNN per-split format
    if "test_mae" in data:
        return float(data["test_mae"])
    if "mean_test_mae" in data:
        return float(data["mean_test_mae"])
    if "best_test_mae" in data:
        return float(data["best_test_mae"])
    # split fallback
    if "splits" in data and data["splits"]:
        s0 = data["splits"][0]
        for k in ("final_test_mae", "test_mae", "best_test_mae"):
            if k in s0:
                return float(s0[k])
    return None


def fmt(x):
    if x is None:
        return "--"
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def main():
    rows = []

    # d-RWNN rows
    for name, cfg in RWNN_CONFIGS:
        row = [name]
        for t in TARGETS:
            v = read_test_mae(QM9_RWNN / cfg / t / "metrics.json")
            row.append(fmt(v) if v is not None else "--")
        rows.append(row)

    # RSNN rows (median (min, max) across 3 splits)
    for m in RSNN_MS:
        row = [f"RSNN (m={m})"]
        for t in TARGETS:
            vals = []
            for s in (0, 1, 2):
                v = read_test_mae(QM9_RSNN / f"m{m}" / t / f"split{s}" / "metrics.json")
                if v is not None:
                    vals.append(v)
            if not vals:
                row.append("--")
            elif len(vals) == 1:
                row.append(fmt(vals[0]))
            else:
                med = statistics.median(vals)
                lo = min(vals)
                hi = max(vals)
                row.append(f"{fmt(med)} ({fmt(lo)}, {fmt(hi)})")
        rows.append(row)

    # build markdown
    header = "| Model | " + " | ".join(f"{t}<br>({UNITS[t]})" for t in TARGETS) + " |"
    sep = "|---|" + "|".join(["---"] * len(TARGETS)) + "|"
    lines = [
        "# QM9 master results: all models × all 12 standard properties",
        "",
        "Test MAE (lower is better). RSNN cells are `median (min, max)` over 3 random splits "
        "(seeds 42/43/44, walk_type=search). d-RWNN cells are single-split test MAE "
        "(walk_type=walk_ada, seed=42).",
        "",
        "Hyperparams (all rows): RSNN_LSTM, h_dim=128, num_layers=2, w=8, batch=128, lr=1e-3, reduce=mean. "
        "d-RWNN: 15 epochs / patience 4. RSNN: 10 epochs / patience 3. Full QM9 (130k molecules).",
        "",
        "Best per column in **bold** (best RSNN m vs best d-RWNN config).",
        "",
        header,
        sep,
    ]

    # find best per column for bolding (numeric only)
    def parse_first_num(cell):
        if cell == "--":
            return None
        # strip "(...)" part
        tok = cell.split(" ")[0]
        try:
            return float(tok)
        except Exception:
            return None

    bests = []
    for c_idx in range(len(TARGETS)):
        col_vals = []
        for r in rows:
            v = parse_first_num(r[c_idx + 1])
            col_vals.append(v)
        valid = [v for v in col_vals if v is not None]
        bests.append(min(valid) if valid else None)

    for r in rows:
        new_cells = [r[0]]
        for c_idx in range(len(TARGETS)):
            cell = r[c_idx + 1]
            v = parse_first_num(cell)
            if v is not None and bests[c_idx] is not None and abs(v - bests[c_idx]) < 1e-9:
                new_cells.append(f"**{cell}**")
            else:
                new_cells.append(cell)
        lines.append("| " + " | ".join(new_cells) + " |")

    out = QM9_RSNN / "results_master.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
