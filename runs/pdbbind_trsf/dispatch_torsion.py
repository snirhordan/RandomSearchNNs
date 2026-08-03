#!/usr/bin/env python3
"""Protein torsion-ordering ablation: dropout0.1/h120 + --protein_order torsion, 3 seeds.

Compares against the allatom dropout0.1/h120 baseline (Pearson 0.5847, already run;
allatom is byte-identical to the pre-torsion sampler, so it is a valid paired baseline).
Same param count (692,009) -- only the protein token ORDER changes (backbone spine ->
phi/psi/omega, side-chain chi path -> chi1..chiN, segment-gated). Requires the identity30
caches to be augmented with torsion_order/torsion_seg (generation.pdbbind --mode augment_torsion).

Idempotent, adaptive GPU count (logical slot -> physical CUDA_VISIBLE_DEVICES).
Layout: runs/pdbbind_trsf/torsion/seed<S>/{metrics.json, model.pt, train.log}
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

BASE = [
    "--data_dir", str(DATA_DIR),
    "--base", "transformer",
    "--h_dim", "120", "--num_layers", "3", "--nhead", "4", "--ffn_mult", "4",
    "--reduce", "mean", "--pos_enc", "rope", "--attn_mode", "full",
    "--lr", "1e-3", "--dropout", "0.1",
    "--m", "6", "--w", "8", "--max_len", "448", "--ligand_max_len", "72",
    "--angles", "1", "--dihedrals", "1", "--angle_K", "8", "--dihedral_K", "4",
    "--distances", "0", "--mol_edge_feat", "0",
    "--epochs", "500", "--patience", "50", "--batch_size", "8",
    "--num_workers", "8", "--standardize", "1",
]

CONFIGS = {
    "torsion": {"flags": ["--protein_order", "torsion"], "seeds": [42, 43, 44]},
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
           "--out_dir", str(d), "--seed", str(j["seed"])] + BASE + j["flags"]
    phys = VISIBLE[gpu]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=phys, OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[torsion] slot={gpu} phys_gpu={phys} pid={p.pid} "
          f"{j['name']}/seed{j['seed']}", flush=True)
    return p


def main():
    all_jobs = list(jobs())
    pending = [j for j in all_jobs if not job_done(j)]
    print(f"[torsion] GPUS={GPUS} | {len(pending)}/{len(all_jobs)} pending", flush=True)
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
                    print(f"[torsion] DONE {key} (gpu {gpu})", flush=True)
                else:
                    n = retries.get(key, 0)
                    if n < MAX_RETRIES:
                        retries[key] = n + 1
                        pending.append(j)
                        print(f"[torsion] RETRY {key} rc={rc}", flush=True)
                    else:
                        print(f"[torsion] FAILED {key} rc={rc}", flush=True)
                del slots[gpu]
            if gpu not in slots and pending:
                j = pending.pop(0)
                slots[gpu] = (launch(j, gpu), j)
        time.sleep(30)
    print("[torsion] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
