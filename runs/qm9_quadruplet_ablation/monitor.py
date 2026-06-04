#!/usr/bin/env python3
"""Single-tick monitor for the Phase 2 ablation (9 cells across 4 L40S GPUs).

Reports a one-line status with running-cell progress; prints an ALL DONE
banner + the 3-config × 3-seed aggregate table when all 9 cells finish.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/snirhordan/ito/RandomSearchNNs/runs/qm9_quadruplet_ablation")
CONFIGS = ["dist_only", "plus_angle", "plus_angle_dihedral"]
SEEDS = [42, 43, 44]


def cell_state(cfg: str, seed: int):
    """Return ('done'|'running'|'pending', metric_or_progress)."""
    cell_dir = ROOT / cfg / f"seed{seed}"
    metrics_flat = cell_dir / "metrics.json"
    metrics_nested = cell_dir / "gap" / "metrics.json"
    log = cell_dir / "train.log"
    # train_qm9.py writes metrics to <run_subdir>/<target>/metrics.json; we
    # also check the flat path in case someone hoisted it.
    for metrics in (metrics_nested, metrics_flat):
        if metrics.exists():
            try:
                data = json.loads(metrics.read_text())
                test_mae = data["splits"][0]["test_mae"]
                return "done", float(test_mae)
            except Exception:
                pass
    if not log.exists():
        return "pending", None
    # Running. Get latest epoch + best valid_mae from the log.
    try:
        # grep -a tolerates NUL bytes that occasionally appear under NFS.
        out = subprocess.run(
            ["grep", "-a", "split 0 epoch", str(log)],
            capture_output=True, text=True, timeout=10,
        ).stdout
        if not out:
            return "running", None
        last = out.strip().splitlines()[-1]
        m_ep = re.search(r"epoch\s+(\d+)", last)
        ep = int(m_ep.group(1)) if m_ep else None
        # Best valid mae so far = min of "valid_mae=X" across all logged epochs.
        vmaes = [float(x) for x in re.findall(r"valid_mae=([0-9.]+)", out)]
        best = min(vmaes) if vmaes else None
        age = int(time.time() - log.stat().st_mtime)
        return "running", {"epoch": ep, "best_vmae": best, "age_s": age}
    except Exception as e:
        return "running", {"error": str(e)}


def main():
    done, running, pending = [], [], []
    for cfg in CONFIGS:
        for s in SEEDS:
            st, val = cell_state(cfg, s)
            label = f"{cfg}/seed{s}"
            if st == "done":
                done.append((label, val))
            elif st == "running":
                running.append((label, val))
            else:
                pending.append(label)

    n_done = len(done)
    n_running = len(running)
    n_pending = len(pending)
    total = n_done + n_running + n_pending

    if n_done == total:
        # ---- ALL DONE: aggregate the table ---------------------------------
        print(f"[ablation] ALL DONE: {n_done}/{total} cells complete")
        print()
        print("| Config | seed42 test_mae | seed43 | seed44 | mean ± std |")
        print("|--------|------|------|------|------|")
        for cfg in CONFIGS:
            row = [cfg]
            vals = []
            for s in SEEDS:
                for label, v in done:
                    if label == f"{cfg}/seed{s}":
                        vals.append(v)
                        row.append(f"{v:.4f}")
                        break
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                row.append(f"{mean:.4f} ± {std:.4f}")
            print("| " + " | ".join(row) + " |")
        return 0

    # ---- still running: one-line tick + per-cell progress -----------------
    print(f"[ablation] tick: done={n_done}/{total} running={n_running} pending={n_pending}")
    for label, prog in sorted(running):
        if isinstance(prog, dict):
            ep = prog.get("epoch", "?")
            best = prog.get("best_vmae")
            best_s = f"{best:.4f}" if best is not None else "—"
            age = prog.get("age_s", "?")
            print(f"  {label}: ep{ep} best_vmae={best_s} age={age}s")
        else:
            print(f"  {label}: (no epoch lines yet)")
    for label in sorted(pending):
        print(f"  {label}: pending")

    # Sanity: is the slot-scheduler PID still alive?
    try:
        with open("/proc/3876185/status") as f:
            print(f"[ablation] scheduler PID 3876185: alive")
    except FileNotFoundError:
        print(f"[ablation] WARNING: scheduler PID 3876185 not found (may have crashed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
