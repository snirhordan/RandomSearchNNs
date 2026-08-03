#!/usr/bin/env python3
"""3-seed confirmation dispatcher for the two best transformer variants.

The main grid (dispatch.py) evaluated every variant on seed 42 only. sinusoidal
(seed42 test P=0.5941) and dropout0.1 (seed42 test P=0.5995) matched / beat the
GET baseline (0.5887), so they need seeds 43 & 44 for a real 3-seed estimate
before any "transformer matches GET" claim. seed42 for both is already done and
is NOT re-run here (disjoint job dirs from the main sweep -> no collision).

Same BASE config and same per-variant flags as dispatch.py. Adaptive GPU count,
idempotent (skips a run whose metrics.json already has test_metrics).

Layout: runs/pdbbind_trsf/<config>/seed<S>/{metrics.json, model.pt, train.log}
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"
OUT_ROOT = REPO / "runs/pdbbind_trsf"
DATA_DIR = REPO / "data/pdbbind/identity30"
MAX_RETRIES = 1


def _visible_ids():
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v not in (None, ""):
        return [x for x in v.split(",") if x != ""]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] if n > 0 else ["0"]


VISIBLE = _visible_ids()
GPUS = list(range(len(VISIBLE)))

# Identical to dispatch.py's BASE (param-matched architecture + sampling).
BASE = [
    "--data_dir", str(DATA_DIR),
    "--base", "transformer",
    "--h_dim", "120", "--num_layers", "3", "--nhead", "4", "--ffn_mult", "4",
    "--reduce", "mean",
    "--m", "6", "--w", "8", "--max_len", "448", "--ligand_max_len", "72",
    "--angles", "1", "--dihedrals", "1", "--angle_K", "8", "--dihedral_K", "4",
    "--distances", "0", "--mol_edge_feat", "0",
    "--epochs", "500", "--patience", "50", "--batch_size", "8",
    "--num_workers", "8", "--standardize", "1",
]

# The two best variants; only the NEW seeds (42 already done in the main sweep).
CONFIGS = {
    "sinusoidal": {"flags": ["--pos_enc", "sinusoidal", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.0"], "seeds": [43, 44]},
    "dropout0.1": {"flags": ["--pos_enc", "rope", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.1"], "seeds": [43, 44]},
}


def jobs():
    for name, cfg in CONFIGS.items():
        for seed in cfg["seeds"]:
            yield {"name": name, "seed": seed, "flags": cfg["flags"]}


def job_dir(j):
    return OUT_ROOT / j["name"] / f"seed{j['seed']}"


def job_done(j):
    m = job_dir(j) / "metrics.json"
    if not m.exists():
        return False
    try:
        with open(m) as f:
            return "test_metrics" in json.load(f)
    except Exception:
        return False


def launch(j, gpu):
    d = job_dir(j)
    d.mkdir(parents=True, exist_ok=True)
    logf = open(d / "dispatch.log", "a")
    cmd = [PYTHON, "-u", "-m", "quickstart.train_pdbbind",
           "--out_dir", str(d), "--seed", str(j["seed"])] + j["flags"] + BASE
    phys = VISIBLE[gpu]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=phys, OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[confirm] slot={gpu} phys_gpu={phys} pid={p.pid} "
          f"{j['name']}/seed{j['seed']}", flush=True)
    return p


def main():
    all_jobs = list(jobs())
    pending = [j for j in all_jobs if not job_done(j)]
    print(f"[confirm] GPUS={GPUS} | {len(pending)}/{len(all_jobs)} pending", flush=True)
    retries = {}
    slots = {}
    while pending or slots:
        for gpu in GPUS:
            if gpu in slots:
                p, j = slots[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                key = f"{j['name']}/seed{j['seed']}"
                if rc == 0 and job_done(j):
                    print(f"[confirm] DONE {key} (gpu {gpu})", flush=True)
                else:
                    n = retries.get(key, 0)
                    if n < MAX_RETRIES:
                        retries[key] = n + 1
                        pending.append(j)
                        print(f"[confirm] RETRY {key} rc={rc}", flush=True)
                    else:
                        print(f"[confirm] FAILED {key} rc={rc}", flush=True)
                del slots[gpu]
            if gpu not in slots and pending:
                j = pending.pop(0)
                slots[gpu] = (launch(j, gpu), j)
        time.sleep(30)
    print("[confirm] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
