#!/usr/bin/env python3
"""Aggregate qm9_trsf sweep results into the deliverable table.

Rows = QM9 targets; columns = transformer RSNN+angles+dihedrals with full
attention vs causal attention (mean±std test MAE over seeds, natural units),
plus the locally-run EGNN baseline (745,224 params, same cormorant split).

Usage: python3 aggregate.py [--markdown]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_json_prefix(path):
    """Parse the first JSON object in a file (some EGNN metrics files have
    a second object appended after a rerun)."""
    return json.JSONDecoder().raw_decode(path.read_text())[0]

ROOT = Path(__file__).resolve().parent
EGNN_ROOT = ROOT.parent / "qm9_egnn"
TARGETS = ["gap", "homo", "lumo", "mu", "alpha", "U0",
           "U", "H", "G", "zpve", "Cv", "R2"]
SEEDS = [42, 43, 44]
UNITS = {"gap": "eV", "homo": "eV", "lumo": "eV", "mu": "D",
         "alpha": "a0^3", "U0": "eV", "U": "eV", "H": "eV", "G": "eV",
         "zpve": "eV", "Cv": "cal/mol K", "R2": "a0^2"}


def collect(pattern_dir):
    maes = []
    for s in SEEDS:
        p = pattern_dir / f"seed{s}" / "metrics.json"
        if not p.exists():
            # qm9_trsf layout: <attn>/seed<S>/<target>/metrics.json handled
            # by caller; egnn layout: <target>/seed<S>/metrics.json
            continue
        m = load_json_prefix(p)
        maes.append(float(m["splits"][0]["test_mae"]))
    return maes


def trsf_maes(attn, target):
    maes = []
    for s in SEEDS:
        p = ROOT / attn / f"seed{s}" / target / "metrics.json"
        if p.exists():
            m = load_json_prefix(p)
            maes.append(float(m["splits"][0]["test_mae"]))
    return maes


def fmt(maes, n_expected=3):
    if not maes:
        return "—"
    mean, std = float(np.mean(maes)), float(np.std(maes))
    note = "" if len(maes) == n_expected else f" ({len(maes)}/{n_expected} seeds)"
    return f"{mean:.4f}±{std:.4f}{note}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows = []
    for t in TARGETS:
        full = trsf_maes("full", t)
        causal = trsf_maes("causal", t)
        egnn = collect(EGNN_ROOT / t)
        rows.append((t, UNITS.get(t, ""), fmt(full), fmt(causal), fmt(egnn)))

    if args.markdown:
        print("| Target | Units | TRSF full attn | TRSF causal attn | EGNN (745K) |")
        print("|---|---|---|---|---|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        w = [8, 10, 24, 24, 24]
        hdr = ["target", "units", "TRSF full", "TRSF causal", "EGNN"]
        print("".join(h.ljust(x) for h, x in zip(hdr, w)))
        for r in rows:
            print("".join(c.ljust(x) for c, x in zip(r, w)))


if __name__ == "__main__":
    main()
