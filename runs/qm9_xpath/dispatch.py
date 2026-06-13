#!/usr/bin/env python3
"""Dispatcher: cross-path attention x bonded-angle gating ablation on QM9 gap.

2x2 design, 3 seeds = 12 jobs, slot-scheduled over the local A40 GPUs:
    attn_mode in {full, full_xpath}  x  bonded_angles_only in {0, 1}

All four cells are run from the same commit (feat/transformer-xpath) for a
self-consistent comparison; the 'full + path-angles' cell reproduces the
prior qm9_trsf gap baseline (0.0804 eV) as a built-in sanity check.

Protocol identical to runs/qm9_trsf (cormorant split, EGNN norm + L1, cosine
LR 7.5e-4, AdamW, warmup 5, m=8 w=16 reduce=sum, angles+dihedrals, 300 ep /
ES 50). Transformer EGNN-param-matched: h128 nl3 ffn4 nhead8 rope.

Idempotent: skips jobs whose metrics.json already has a summary. Relaunch with
nohup .../envs/rwnn/bin/python3 -u runs/qm9_xpath/dispatch.py after interruption.
Layout: runs/qm9_xpath/<cell>/seed<S>/gap/{metrics.json,train.log}
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"
OUT_ROOT = REPO / "runs/qm9_xpath"

GPUS = [0, 1, 2, 3, 4, 5]
SEEDS = [42, 43, 44]
MAX_RETRIES = 1

# (cell name, attn_mode, bonded_angles_only)
CELLS = [
    ("A_full_path",    "full",       0),
    ("B_xpath_path",   "full_xpath", 0),
    ("C_full_bonded",  "full",       1),
    ("D_xpath_bonded", "full_xpath", 1),
]

COMMON = [
    "--target", "gap", "--split", "cormorant",
    "--cormorant_data_dir", str(REPO / "external/egnn/qm9/temp/qm9"),
    "--lr_scheduler", "cosine",
    "--use_egnn_normalization", "1",
    "--norm_constants_json", str(REPO / "runs/qm9_compare/preprocessing_audit.json"),
    "--walk_type", "search", "--distances", "1", "--mol_edge_feat", "1",
    "--max_search_len", "16",
    "--angles", "1", "--dihedrals", "1", "--angle_K", "8", "--dihedral_K", "4",
    "--vectorize_quadruplet", "1",
    "--epochs", "300", "--early_stopping", "50", "--n_splits", "1",
    "--batch_size", "96", "--m", "8", "--w", "16", "--reduce", "sum",
    "--lr", "0.00075", "--optimizer", "adamw",
    "--grad_clip", "1.0", "--weight_decay", "0.0001",
    "--dropout", "0.0", "--warmup_epochs", "5",
    "--num_workers", "7",
    "--base", "transformer", "--h_dim", "128", "--num_layers", "3",
    "--ffn_mult", "4", "--nhead", "8", "--pos_enc", "rope",
    "--out_root", str(OUT_ROOT), "--limit", "0",
]


def jobs():
    for cell, attn, bonded in CELLS:
        for seed in SEEDS:
            yield {"cell": cell, "attn": attn, "bonded": bonded, "seed": seed}


def job_dir(j):
    return OUT_ROOT / j["cell"] / f"seed{j['seed']}" / "gap"


def job_done(j):
    m = job_dir(j) / "metrics.json"
    if not m.exists():
        return False
    try:
        with open(m) as f:
            return "summary" in json.load(f)
    except Exception:
        return False


def launch(j, gpu):
    d = job_dir(j)
    d.mkdir(parents=True, exist_ok=True)
    logf = open(d / "train.log", "a")
    cmd = [PYTHON, "-u", str(REPO / "quickstart/train_qm9.py"),
           "--attn_mode", j["attn"],
           "--bonded_angles_only", str(j["bonded"]),
           "--seed", str(j["seed"]),
           "--device_idx", "0",
           "--run_subdir", f"{j['cell']}/seed{j['seed']}",
           ] + COMMON
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[dispatch] gpu={gpu} pid={p.pid} {j['cell']}/seed{j['seed']}",
          flush=True)
    return p


def main():
    pending = [j for j in jobs() if not job_done(j)]
    print(f"[dispatch] {len(pending)} pending, {12 - len(pending)} done",
          flush=True)
    retries, slots = {}, {}
    while pending or slots:
        for gpu in GPUS:
            if gpu in slots:
                p, j = slots[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                key = f"{j['cell']}/seed{j['seed']}"
                if rc == 0 and job_done(j):
                    print(f"[dispatch] DONE {key} (gpu {gpu})", flush=True)
                else:
                    n = retries.get(key, 0)
                    if n < MAX_RETRIES:
                        retries[key] = n + 1
                        pending.append(j)
                        print(f"[dispatch] RETRY {key} rc={rc}", flush=True)
                    else:
                        print(f"[dispatch] FAILED {key} rc={rc}", flush=True)
                del slots[gpu]
            if gpu not in slots and pending:
                j = pending.pop(0)
                slots[gpu] = (launch(j, gpu), j)
        time.sleep(60)
    print("[dispatch] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
