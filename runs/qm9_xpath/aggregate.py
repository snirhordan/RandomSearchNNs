#!/usr/bin/env python3
"""Aggregate the cross-path x bonded-angle 2x2 ablation on QM9 gap.

Prints mean+/-std test MAE (eV) over seeds for each cell, plus the EGNN gap
reference (0.0504 eV local rerun) for context. Cells:
    A_full_path    = within-walk full attention, path-order angles (baseline)
    B_xpath_path   = cross-path attention,        path-order angles
    C_full_bonded  = within-walk full attention,  bonded-only angles
    D_xpath_bonded = cross-path attention,         bonded-only angles
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
CELLS = ["A_full_path", "B_xpath_path", "C_full_bonded", "D_xpath_bonded"]
EGNN_GAP = 0.0504  # local EGNN rerun, eV (paper 0.048)


def maes(cell):
    out = []
    for s in SEEDS:
        p = ROOT / cell / f"seed{s}" / "gap" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            if "summary" in m:
                out.append(float(m["splits"][0]["test_mae"]))
    return out


def main():
    print(f"{'cell':16} {'test MAE (eV)':22} {'n':3} {'/EGNN':7}")
    base = None
    for c in CELLS:
        v = maes(c)
        if not v:
            print(f"{c:16} {'(pending)':22} {len(v)}")
            continue
        mean, std = float(np.mean(v)), float(np.std(v))
        if c == "A_full_path":
            base = mean
        print(f"{c:16} {mean:.4f}+/-{std:.4f}{'':6} {len(v)}  "
              f"{mean/EGNN_GAP:.2f}x")
    print(f"\nEGNN gap reference: {EGNN_GAP:.4f} eV")
    if base is not None:
        print("(deltas vs A_full_path baseline, meV):")
        for c in CELLS[1:]:
            v = maes(c)
            if v:
                print(f"  {c:16} {(np.mean(v)-base)*1000:+.1f} meV")


if __name__ == "__main__":
    main()
