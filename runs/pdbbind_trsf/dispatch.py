#!/usr/bin/env python3
"""Dispatcher: RSNN_TRSF (transformer + angles/dihedrals) on PDBbind-Benchmark identity30.

Parameter-matched to GET (identity30 GET = 719,356 trainable; band [683388, 755324]).
The architecture h_dim=120 / num_layers=3 / nhead=4 / ffn_mult=4 -> 692,009 params is
FIXED across all configs (that is what keeps every run inside the parameter budget), and
the grid varies only training / positional-encoding hyperparameters around the standard
config. First the standard config on 3 seeds (the head-to-head vs GET), then a one-factor
grid on seed 42.

Slot-scheduled over the GPUs visible in the allocation (adaptive: uses however many GPUs
CUDA_VISIBLE_DEVICES / torch exposes -- request 6 on dym-lab, fewer is fine). Idempotent:
a job whose metrics.json already contains test_metrics is skipped, so the script is safe
to re-run after preemption/interruption.

Layout: runs/pdbbind_trsf/<config>/seed<S>/{metrics.json, model.pt, train.log, dispatch.log}
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
    """Physical GPU ids visible in this allocation (from SLURM's
    CUDA_VISIBLE_DEVICES, e.g. ['2','4','5']); falls back to torch's count."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v not in (None, ""):
        return [x for x in v.split(",") if x != ""]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] if n > 0 else ["0"]


VISIBLE = _visible_ids()      # physical ids owned by this allocation
GPUS = list(range(len(VISIBLE)))  # logical slots 0..N-1

# Standard (param-matched) architecture + sampling shared by every config.
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

# One-factor grid around the standard config. Architecture is fixed (=> in budget);
# only pos_enc / attn_mode / lr / dropout vary.
CONFIGS = {
    "standard":   {"flags": ["--pos_enc", "rope", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.0"], "seeds": [42, 43, 44]},
    "sinusoidal": {"flags": ["--pos_enc", "sinusoidal", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.0"], "seeds": [42]},
    "causal":     {"flags": ["--pos_enc", "rope", "--attn_mode", "causal", "--lr", "1e-3", "--dropout", "0.0"], "seeds": [42]},
    "lr5e4":      {"flags": ["--pos_enc", "rope", "--attn_mode", "full", "--lr", "5e-4", "--dropout", "0.0"], "seeds": [42]},
    "dropout0.1": {"flags": ["--pos_enc", "rope", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.1"], "seeds": [42]},
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
    # gpu is a LOGICAL slot; map it to the physical id this allocation owns so a
    # child never grabs a GPU outside the SLURM allocation.
    phys = VISIBLE[gpu]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=phys, OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[dispatch] slot={gpu} phys_gpu={phys} pid={p.pid} "
          f"{j['name']}/seed{j['seed']}", flush=True)
    return p


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_jobs = list(jobs())
    pending = [j for j in all_jobs if not job_done(j)]
    print(f"[dispatch] GPUS={GPUS} | {len(pending)}/{len(all_jobs)} pending "
          f"({len(all_jobs) - len(pending)} already done)", flush=True)
    retries = {}
    slots = {}  # gpu -> (Popen, job)
    while pending or slots:
        for gpu in GPUS:
            if gpu in slots:
                p, j = slots[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                key = f"{j['name']}/seed{j['seed']}"
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
        time.sleep(30)
    print("[dispatch] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
