#!/usr/bin/env python3
"""Run EGNN for one (target, seed) pair.

Calls vgsatorras/egnn's main_qm9.py via subprocess, then parses losess.json
into our standard metrics.json schema for unified aggregation with RSNN/RWNN.

EGNN-specific notes:
- Param count at nf=128, n_layers=7 = 745,224 (matches RSNN_LSTM at 742,801, +0.3%).
- zpve test_mae is reported in meV by EGNN (Hartree x 27211.4); divide by 1000
  to express in eV (matches the RSNN/RWNN convention).
- Data split is fixed (cormorant) — seed only affects init + minibatch order.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EGNN_DIR = REPO / "external" / "egnn"
PY = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"

# Our target names (preserve case used in PyG QM9 / RSNN runs) -> EGNN's flag value.
TARGET_MAP = {
    "mu": "mu", "alpha": "alpha", "homo": "homo", "lumo": "lumo",
    "gap": "gap", "R2": "r2", "zpve": "zpve",
    "U0": "U0", "U": "U", "H": "H", "G": "G", "Cv": "Cv",
}
# zpve is reported in meV by EGNN; multiply by this to convert to eV
ZPVE_MEV_TO_EV = 1e-3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=list(TARGET_MAP.keys()))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=0,
                   help="early-stopping patience on val loss; 0 disables")
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--nf", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=7)
    p.add_argument("--attention", type=int, default=1)
    p.add_argument("--node_attr", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_root", required=True,
                   help="Output base, e.g. runs/qm9_egnn")
    p.add_argument("--device_idx", type=int, default=0,
                   help="GPU index for CUDA_VISIBLE_DEVICES")
    args = p.parse_args()

    egnn_prop = TARGET_MAP[args.target]
    out_dir = Path(args.out_root) / args.target / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and metrics_path.stat().st_size > 100:
        print(f"[run_one] skip {args.target}/seed{args.seed} (metrics exists)")
        return 0

    exp_name = f"r_{args.target}_s{args.seed}_e{args.epochs}"
    egnn_log_dir = EGNN_DIR / "qm9" / "logs" / exp_name
    if egnn_log_dir.exists():
        shutil.rmtree(egnn_log_dir)

    train_log = out_dir / "train.log"
    cmd = [
        PY, "main_qm9.py",
        "--exp_name", exp_name,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--nf", str(args.nf),
        "--n_layers", str(args.n_layers),
        "--attention", str(args.attention),
        "--node_attr", str(args.node_attr),
        "--property", egnn_prop,
        "--seed", str(args.seed),
        "--patience", str(args.patience),
    ]
    env = os.environ.copy()
    # If the dispatcher already pinned CUDA_VISIBLE_DEVICES, inherit it
    # (every job gets the physical GPU set by the dispatcher's
    # CUDA_VISIBLE_DEVICES=N env). Only fall back to --device_idx when the
    # caller didn't pin anything, e.g. when running this script standalone.
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device_idx)

    t0 = time.time()
    print(f"[run_one] launching {args.target}/seed{args.seed} on GPU {args.device_idx}")
    with open(train_log, "w") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(EGNN_DIR),
            env=env,
            stdout=f, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"[run_one] EGNN failed for {args.target}/seed{args.seed} (rc={proc.returncode})")
        return proc.returncode

    losess_path = EGNN_DIR / "qm9" / "logs" / exp_name / "losess.json"
    if not losess_path.exists():
        print(f"[run_one] no losess.json at {losess_path}")
        return 2
    with open(losess_path) as f:
        losess = json.load(f)

    test_mae_eV = float(losess["best_test"])
    if args.target == "zpve":
        test_mae_eV *= ZPVE_MEV_TO_EV

    metrics = {
        "config": {
            "model": "EGNN",
            "target": args.target,
            "egnn_property": egnn_prop,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "nf": args.nf,
            "n_layers": args.n_layers,
            "attention": args.attention,
            "node_attr": args.node_attr,
            "n_splits": 1,
            "dataset": "cormorant-qm9",
            "param_count": 745224,
            "zpve_units": "eV (converted from meV by /1000)",
        },
        "splits": [
            {
                "split": 0,
                "epochs_run": int(losess["best_epoch"]) + 1,
                "best_valid_mae": float(losess["best_val"]),
                "test_mae": test_mae_eV,
                "raw_best_test": float(losess["best_test"]),
                "split_dt_sec": dt,
                "epochs": [
                    {"epoch": e, "test_mae": l}
                    for e, l in zip(losess["epochs"], losess["losess"])
                ],
            }
        ],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[run_one] {args.target}/seed{args.seed} done: test_mae={test_mae_eV:.6g} dt={dt/60:.1f}min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
