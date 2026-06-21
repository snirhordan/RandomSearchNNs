#!/usr/bin/env python3
"""Dispatcher: canonical-walk + geometric-attention-bias ladder on QM9 gap.

3 cells x 3 seeds = 9 jobs, slot-scheduled over the local A40 GPUs. The ladder
isolates the two levers in sequence ("determinize the path first, then add the
bias"):

    base_random     random multi-walk (m=8), no bias   -> reproduces the
                    prior qm9_trsf gap baseline (~0.0804 eV) as a sanity anchor.
    canonical_only  single deterministic canonical DFS walk (m forced to 1),
                    no bias                              -> does determinizing
                    the path move gap?
    canonical_bias  canonical walk + E(3)-invariant geometric attention bias
                    (distance + 2 angles + dihedral)    -> does the bias help?

Canonical cells leave --max_search_len at its default (None) so the single walk
covers every atom; the random baseline keeps the prior --max_search_len 16.

Protocol otherwise identical to runs/qm9_trsf / runs/qm9_xpath (cormorant split,
EGNN norm + L1, cosine LR 7.5e-4, AdamW, warmup 5, w=16 reduce=sum,
angles+dihedrals, 300 ep / ES 50). Transformer EGNN-param-matched: h128 nl3
ffn4 nhead8 rope.

Idempotent: skips jobs whose metrics.json already has a summary. Relaunch with
  nohup .../envs/rwnn/bin/python3 -u runs/qm9_geom/dispatch.py \
      > runs/qm9_geom/dispatch.log 2>&1 &
Layout: runs/qm9_geom/<cell>/seed<S>/gap/{metrics.json,train.log}
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"
OUT_ROOT = REPO / "runs/qm9_geom"

GPUS = [0, 1, 2, 3, 4, 5]
SEEDS = [42, 43, 44]
MAX_RETRIES = 1

# (cell name, extra flags). Canonical cells omit --max_search_len (default None)
# so the single full-coverage walk is not truncated; --m 8 is forced to 1
# internally in canonical mode.
CELLS = [
    ("base_random",    ["--canonical", "0", "--max_search_len", "16"]),
    ("canonical_only", ["--canonical", "1"]),
    ("canonical_bias", ["--canonical", "1", "--geom_bias", "1",
                        "--geom_rbf_K", "16", "--geom_angle_K", "8",
                        "--geom_dihedral_K", "4", "--geom_hidden", "32"]),
    # Decoupling arm: bias on the random multi-walk (m=8) baseline -> isolates
    # the geometric bias from the path-determinization cost. Same config as
    # base_random (incl. --max_search_len 16) + --geom_bias 1.
    ("random_bias",    ["--canonical", "0", "--max_search_len", "16",
                        "--geom_bias", "1", "--geom_rbf_K", "16",
                        "--geom_angle_K", "8", "--geom_dihedral_K", "4",
                        "--geom_hidden", "32"]),
]

COMMON = [
    "--target", "gap", "--split", "cormorant",
    "--cormorant_data_dir", str(REPO / "external/egnn/qm9/temp/qm9"),
    "--lr_scheduler", "cosine",
    "--use_egnn_normalization", "1",
    "--norm_constants_json", str(REPO / "runs/qm9_compare/preprocessing_audit.json"),
    "--walk_type", "search", "--distances", "1", "--mol_edge_feat", "1",
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
    "--attn_mode", "full",
    "--out_root", str(OUT_ROOT), "--limit", "0",
]


def jobs():
    for cell, extra in CELLS:
        for seed in SEEDS:
            yield {"cell": cell, "extra": extra, "seed": seed}


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
           "--seed", str(j["seed"]),
           "--device_idx", "0",
           "--run_subdir", f"{j['cell']}/seed{j['seed']}",
           ] + j["extra"] + COMMON
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[dispatch] gpu={gpu} pid={p.pid} {j['cell']}/seed{j['seed']}",
          flush=True)
    return p


def main():
    total = len(CELLS) * len(SEEDS)
    pending = [j for j in jobs() if not job_done(j)]
    print(f"[dispatch] {len(pending)} pending, {total - len(pending)} done",
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
