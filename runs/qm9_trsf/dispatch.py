#!/usr/bin/env python3
"""Dispatcher: transformer-based RSNN+angles+dihedrals on all QM9 targets.

72 jobs = 12 targets x {full, causal} attention x seeds {42, 43, 44},
slot-scheduled over the local A40 GPUs. Idempotent: jobs whose metrics.json
already exists are skipped, so the script can be re-run after interruption.

Protocol matches the Phase-2 quadruplet ablation (cormorant split, EGNN
normalization + L1, cosine LR, batch 96, m=8 w=16 reduce=sum) with the
transformer base sized to EGNN's param count: h_dim 128, num_layers 3,
ffn_mult 4, nhead 8, pos_enc rope -> 774,433 params (EGNN: 745,224, +3.9%).
warmup_epochs 5 added (transformer best practice; LSTM runs used 0).

Layout: runs/qm9_trsf/<attn>/seed<S>/<target>/{metrics.json,train.log}
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"
OUT_ROOT = REPO / "runs/qm9_trsf"

GPUS = [0, 1, 2, 3, 4, 5]
# gap first (headline target), then the rest of the 12 cormorant targets.
TARGETS = ["gap", "homo", "lumo", "mu", "alpha", "U0",
           "U", "H", "G", "zpve", "Cv", "R2"]
ATTN_MODES = ["full", "causal"]
SEEDS = [42, 43, 44]
MAX_RETRIES = 1

COMMON = [
    "--split", "cormorant",
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
    # 6 concurrent jobs x 7 workers = 42 loader procs on 48 cores.
    "--num_workers", "7",
    # transformer base, EGNN-param-matched
    "--base", "transformer", "--h_dim", "128", "--num_layers", "3",
    "--ffn_mult", "4", "--nhead", "8", "--pos_enc", "rope",
    "--out_root", str(OUT_ROOT), "--limit", "0",
]


def jobs():
    for target in TARGETS:
        for attn in ATTN_MODES:
            for seed in SEEDS:
                yield {"target": target, "attn": attn, "seed": seed}


def job_dir(j):
    return OUT_ROOT / j["attn"] / f"seed{j['seed']}" / j["target"]


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
           "--target", j["target"],
           "--attn_mode", j["attn"],
           "--seed", str(j["seed"]),
           "--device_idx", "0",
           "--run_subdir", f"{j['attn']}/seed{j['seed']}",
           ] + COMMON
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[dispatch] gpu={gpu} pid={p.pid} "
          f"{j['target']}/{j['attn']}/seed{j['seed']}", flush=True)
    return p


def main():
    pending = [j for j in jobs() if not job_done(j)]
    skipped = 72 - len(pending)
    print(f"[dispatch] {len(pending)} pending, {skipped} already done",
          flush=True)
    retries = {}
    slots = {}   # gpu -> (Popen, job)
    while pending or slots:
        for gpu in GPUS:
            if gpu in slots:
                p, j = slots[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                key = f"{j['target']}/{j['attn']}/seed{j['seed']}"
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
