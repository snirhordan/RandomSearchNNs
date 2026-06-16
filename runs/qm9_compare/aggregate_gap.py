#!/usr/bin/env python3
"""Aggregate gap apples-to-apples results: EGNN + RSNN + d-RWNN.

Reads 9 metrics.json files:
  runs/qm9_compare/egnn/seed{42,43,44}/metrics.json    (symlinked to qm9_egnn)
  runs/qm9_compare/rsnn/seed{42,43,44}/metrics.json
  runs/qm9_compare/d_rwnn/seed{42,43,44}/metrics.json

Computes per-model mean ± std of test_mae across 3 seeds. Applies the
pooled-std competitive criterion vs EGNN:
    MAE_X < MAE_EGNN + sqrt((var_X + var_EGNN) / 2)

Writes:
  runs/qm9_compare/results_gap.json
  runs/qm9_compare/results_gap.md

Also reports `epochs_run` per cell for transparency about early-stop.
"""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMPARE_DIR = REPO / "runs" / "qm9_compare"
MODELS = ["egnn", "rsnn", "d_rwnn"]
SEEDS = [42, 43, 44]


def read_test_mae(metrics_path: Path) -> tuple[float | None, int | None]:
    if not metrics_path.exists():
        return None, None
    try:
        d = json.loads(metrics_path.read_text())
    except Exception:
        return None, None
    splits = d.get("splits") or [{}]
    s0 = splits[0]
    mae = s0.get("test_mae")
    if mae is None:
        mae = s0.get("final_test_mae")
    if mae is None:
        sm = d.get("summary") or {}
        mae = sm.get("mean_test_mae")
    epochs_run = s0.get("epochs_run") or s0.get("best_epoch")
    return (float(mae) if mae is not None else None,
            int(epochs_run) if epochs_run is not None else None)


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "var": None,
                "min": None, "max": None}
    mean = statistics.mean(values)
    var = (statistics.pvariance(values) if len(values) == 1
           else statistics.variance(values))
    std = math.sqrt(var)
    return {"n": len(values), "mean": mean, "std": std, "var": var,
            "min": min(values), "max": max(values)}


def fmt(x):
    if x is None:
        return "--"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def main() -> int:
    cells = {}
    for model in MODELS:
        cells[model] = {}
        for s in SEEDS:
            mp = COMPARE_DIR / model / f"seed{s}" / "metrics.json"
            mae, ep = read_test_mae(mp)
            cells[model][s] = {"test_mae": mae, "epochs_run": ep,
                               "path": str(mp.relative_to(REPO))}

    # Per-model stats.
    summary = {}
    for model in MODELS:
        maes = [c["test_mae"] for c in cells[model].values()
                if c["test_mae"] is not None]
        summary[model] = stats(maes)

    egnn_s = summary["egnn"]
    verdict = {}
    if egnn_s["mean"] is None:
        verdict_note = "no EGNN reference; cannot adjudicate competitiveness"
        for m in ("rsnn", "d_rwnn"):
            verdict[m] = None
    else:
        verdict_note = (
            "Competitive iff MAE_X < MAE_EGNN + "
            "sqrt((var_X + var_EGNN)/2)")
        for m in ("rsnn", "d_rwnn"):
            ms = summary[m]
            if ms["mean"] is None or ms["var"] is None:
                verdict[m] = None
                continue
            pooled_std = math.sqrt((ms["var"] + egnn_s["var"]) / 2.0)
            threshold = egnn_s["mean"] + pooled_std
            verdict[m] = {
                "competitive": ms["mean"] < threshold,
                "delta_mean": ms["mean"] - egnn_s["mean"],
                "pooled_std": pooled_std,
                "threshold": threshold,
            }

    out_json = {
        "schema_version": 1,
        "target": "gap",
        "units": "eV",
        "verdict_rule": verdict_note,
        "summary": summary,
        "verdict": verdict,
        "cells": cells,
    }
    (COMPARE_DIR / "results_gap.json").write_text(
        json.dumps(out_json, indent=2) + "\n")

    # Markdown
    lines = [
        "# Gap apples-to-apples comparison",
        "",
        "Test MAE (eV, lower=better) on QM9 `gap` (Cormorant fixed split, "
        "100k/17748/13083). 3 seeds {42,43,44}.",
        "",
        "**Protocol (all three models)**: Adam, batch=96, 300 epochs cap, "
        "patience=50 on val MAE, EGNN-style `meann/MAD` normalization "
        "(loaded from `preprocessing_audit.json`), L1 loss for RSNN/d-RWNN. "
        "EGNN uses upstream cosine schedule (3 step()/epoch — accepted as-is); "
        "RSNN/d-RWNN use cosine T_max=300 stepped 1×/epoch (textbook). "
        "RSNN/d-RWNN LR=7.5e-4 (1e-3 × 96/128 batch rescale). EGNN LR=1e-3.",
        "",
        "## Per-model statistics (3 seeds)",
        "",
        "| Model | Mean test MAE | Std | N | Min | Max |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        s = summary[m]
        lines.append(
            f"| {m} | {fmt(s['mean'])} | {fmt(s['std'])} | {s['n']} | "
            f"{fmt(s['min'])} | {fmt(s['max'])} |")
    lines += [
        "",
        "## Verdict (pooled-std rule vs EGNN)",
        "",
        f"_{verdict_note}_",
        "",
        "| Model | MAE - MAE_EGNN | Pooled std | Threshold | Competitive? |",
        "|---|---|---|---|---|",
    ]
    for m in ("rsnn", "d_rwnn"):
        v = verdict[m]
        if v is None:
            lines.append(f"| {m} | -- | -- | -- | -- |")
        else:
            lines.append(
                f"| {m} | {fmt(v['delta_mean'])} | "
                f"{fmt(v['pooled_std'])} | {fmt(v['threshold'])} | "
                f"{'**YES**' if v['competitive'] else 'NO'} |")

    lines += [
        "",
        "## Per-cell detail",
        "",
        "| Model | Seed | test MAE (eV) | epochs_run |",
        "|---|---|---|---|",
    ]
    for m in MODELS:
        for s in SEEDS:
            c = cells[m][s]
            lines.append(
                f"| {m} | {s} | {fmt(c['test_mae'])} | "
                f"{c['epochs_run'] if c['epochs_run'] is not None else '--'} |")

    (COMPARE_DIR / "results_gap.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {COMPARE_DIR / 'results_gap.json'}")
    print(f"wrote {COMPARE_DIR / 'results_gap.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
