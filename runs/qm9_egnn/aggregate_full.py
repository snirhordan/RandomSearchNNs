#!/usr/bin/env python3
"""Aggregate the FULL QM9 master table: d-RWNN + RSNN + d-RWNN m-sweep + EGNN.

Reads:
  runs/qm9/d{0,1}_m{0,1}/{target}/metrics.json              (d-RWNN, single split)
  runs/qm9_rsnn/m{1,4,8,16}/{target}/split{0,1,2}/metrics.json   (RSNN, 3 splits)
  runs/qm9_rwnn/m{4,8,16}/{target}/split{0,1,2}/metrics.json     (d-RWNN m-sweep)
  runs/qm9_egnn/{target}/seed{42,43,44}/metrics.json             (EGNN, 3 seeds)

Writes:
  runs/qm9_egnn/results_full.md
"""
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QM9_RWNN_LEGACY = REPO / "runs" / "qm9"
QM9_RSNN = REPO / "runs" / "qm9_rsnn"
QM9_RWNN_SWEEP = REPO / "runs" / "qm9_rwnn"
QM9_EGNN = REPO / "runs" / "qm9_egnn"

TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "R2", "zpve",
           "U0", "U", "H", "G", "Cv"]
UNITS = {
    "mu": "D", "alpha": "Bohr^3", "homo": "eV", "lumo": "eV", "gap": "eV",
    "R2": "Bohr^2", "zpve": "eV", "U0": "eV", "U": "eV", "H": "eV", "G": "eV",
    "Cv": "cal/(mol K)",
}
RWNN_LEGACY_CONFIGS = [
    ("d-RWNN (d=0, b=0)", "d0_m0"),
    ("d-RWNN (d=1, b=0)", "d1_m0"),
    ("d-RWNN (d=0, b=1)", "d0_m1"),
    ("d-RWNN (d=1, b=1)", "d1_m1"),
]
RSNN_MS = [1, 4, 8, 16]
RWNN_SWEEP_MS = [16, 8, 4]
EGNN_SEEDS = [42, 43, 44]


def read_test_mae(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    if "summary" in data and "mean_test_mae" in data["summary"]:
        return float(data["summary"]["mean_test_mae"])
    if "test_mae" in data:
        return float(data["test_mae"])
    if "splits" in data and data["splits"]:
        s0 = data["splits"][0]
        for k in ("test_mae", "final_test_mae", "best_test_mae"):
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


def cell_from_replicates(vals):
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt(vals[0])
    return f"{fmt(statistics.median(vals))} ({fmt(min(vals))}, {fmt(max(vals))})"


def parse_first_num(cell):
    if cell == "--":
        return None
    tok = cell.split(" ")[0].replace("**", "")
    try:
        return float(tok)
    except Exception:
        return None


def main():
    rows = []

    for name, cfg in RWNN_LEGACY_CONFIGS:
        row = [name]
        for t in TARGETS:
            v = read_test_mae(QM9_RWNN_LEGACY / cfg / t / "metrics.json")
            row.append(fmt(v) if v is not None else "--")
        rows.append(row)

    for m in RSNN_MS:
        row = [f"RSNN (m={m})"]
        for t in TARGETS:
            vals = []
            for s in (0, 1, 2):
                v = read_test_mae(QM9_RSNN / f"m{m}" / t / f"split{s}" / "metrics.json")
                if v is not None:
                    vals.append(v)
            row.append(cell_from_replicates(vals))
        rows.append(row)

    for m in RWNN_SWEEP_MS:
        row = [f"d-RWNN (m={m})"]
        for t in TARGETS:
            vals = []
            for s in (0, 1, 2):
                v = read_test_mae(QM9_RWNN_SWEEP / f"m{m}" / t / f"split{s}" / "metrics.json")
                if v is not None:
                    vals.append(v)
            row.append(cell_from_replicates(vals))
        rows.append(row)

    row = ["EGNN (paper-faithful)"]
    for t in TARGETS:
        vals = []
        for s in EGNN_SEEDS:
            v = read_test_mae(QM9_EGNN / t / f"seed{s}" / "metrics.json")
            if v is not None:
                vals.append(v)
        row.append(cell_from_replicates(vals))
    rows.append(row)

    header = "| Model | " + " | ".join(f"{t}<br>({UNITS[t]})" for t in TARGETS) + " |"
    sep = "|---|" + "|".join(["---"] * len(TARGETS)) + "|"
    lines = [
        "# QM9 full master results: all models × all 12 standard properties",
        "",
        "Test MAE (lower is better).",
        "",
        "- **d-RWNN (d/b)**: single split (walk_type=walk_ada, seed=42, 15 epochs).",
        "- **RSNN (m=...)**: median (min, max) over 3 random splits (seeds 42/43/44, walk_type=search, 10 epochs/patience 3).",
        "- **d-RWNN (m=...)**: median (min, max) over 3 random splits (walk_type=walk_ada, distances=1, mol_edge_feat=1, 15 epochs/patience 4).",
        "- **EGNN (paper-faithful)**: median (min, max) over 3 seeds, n_layers=7, nf=128, attention=1, batch=96, 300 epochs/patience=50, cormorant fixed split (seed only affects init+minibatch order). zpve test MAE converted from meV to eV.",
        "",
        "Hyperparams shared (RSNN/RWNN families): RSNN_LSTM h_dim=128, num_layers=2, w=8 walks, batch=128, lr=1e-3, reduce=mean, ~743k params. EGNN ~745k params.",
        "",
        "Best per column in **bold**.",
        "",
        header,
        sep,
    ]

    bests = []
    for c_idx in range(len(TARGETS)):
        col_vals = [parse_first_num(r[c_idx + 1]) for r in rows]
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

    out = QM9_EGNN / "results_full.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
