#!/usr/bin/env python3
"""Head-to-head on additional protein-ligand datasets/splits: carry the identity30
winner (dropout0.1) + grid points (causal, lr5e4) to identity60, scaffold, and LBA.

Tests whether the "param-matched plain transformer ~ GET, chemistry tweaks don't help"
finding holds on a different split (identity60/scaffold, same PDBbind complexes) and a
genuinely new dataset (Atom3D LBA). dropout0.1 gets 3 seeds (the head-to-head); the two
grid points get seed 42 for context. Same 692,009-param config; GET baselines run
separately (PDBBind/<split>_get.json ; LBA/get.json).

A dataset is skipped until its rwnn cache (data/pdbbind/<name> or data/lba/<name>) exists,
so LBA can be added to DATASETS and will start once its cache is built. Idempotent,
adaptive GPU count. Layout: runs/pdbbind_trsf/<dataset>/<config>/seed<S>/{metrics.json,...}
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
MAX_RETRIES = 1

# name -> rwnn cache dir (must contain train/valid/test.pt + vocab.json)
DATASETS = {
    "identity60": REPO / "data/pdbbind/identity60",
    "scaffold":   REPO / "data/pdbbind/scaffold",
    "lba":        REPO / "data/lba/lba30",   # skipped until built
}

# shared config (no data_dir / pos_enc / attn_mode / lr / dropout -- those vary below)
BASE = [
    "--base", "transformer",
    "--h_dim", "120", "--num_layers", "3", "--nhead", "4", "--ffn_mult", "4",
    "--reduce", "mean",
    "--m", "6", "--w", "8", "--max_len", "448", "--ligand_max_len", "72",
    "--angles", "1", "--dihedrals", "1", "--angle_K", "8", "--dihedral_K", "4",
    "--distances", "0", "--mol_edge_feat", "0",
    "--epochs", "500", "--patience", "50", "--batch_size", "8",
    "--num_workers", "8", "--standardize", "1",
]

CONFIGS = {
    "dropout0.1": {"flags": ["--pos_enc", "rope", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.1"], "seeds": [42, 43, 44]},
    "causal":     {"flags": ["--pos_enc", "rope", "--attn_mode", "causal", "--lr", "1e-3", "--dropout", "0.0"], "seeds": [42]},
    "lr5e4":      {"flags": ["--pos_enc", "rope", "--attn_mode", "full", "--lr", "5e-4", "--dropout", "0.0"], "seeds": [42]},
}


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


_ONLY = set(sys.argv[1:])  # optional dataset filter, e.g. `dispatch_multidataset.py lba`


def jobs():
    for ds, ddir in DATASETS.items():
        if _ONLY and ds not in _ONLY:
            continue
        if not (Path(ddir) / "train.pt").exists():
            print(f"[multi] SKIP dataset '{ds}' -- cache {ddir}/train.pt not built yet", flush=True)
            continue
        for cfg, spec in CONFIGS.items():
            for seed in spec["seeds"]:
                yield {"ds": ds, "ddir": str(ddir), "cfg": cfg, "seed": seed, "flags": spec["flags"]}


def job_dir(j):
    return OUT_ROOT / j["ds"] / j["cfg"] / f"seed{j['seed']}"


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
           "--out_dir", str(d), "--seed", str(j["seed"]),
           "--data_dir", j["ddir"]] + BASE + j["flags"]
    phys = VISIBLE[gpu]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=phys, OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[multi] slot={gpu} phys={phys} pid={p.pid} {j['ds']}/{j['cfg']}/seed{j['seed']}", flush=True)
    return p


def main():
    all_jobs = list(jobs())
    pending = [j for j in all_jobs if not job_done(j)]
    print(f"[multi] GPUS={GPUS} | {len(pending)}/{len(all_jobs)} pending", flush=True)
    retries = {}
    slots = {}
    while pending or slots:
        for gpu in GPUS:
            if gpu in slots:
                p, j = slots[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                key = f"{j['ds']}/{j['cfg']}/seed{j['seed']}"
                if rc == 0 and job_done(j):
                    print(f"[multi] DONE {key}", flush=True)
                else:
                    n = retries.get(key, 0)
                    if n < MAX_RETRIES:
                        retries[key] = n + 1
                        pending.append(j)
                        print(f"[multi] RETRY {key} rc={rc}", flush=True)
                    else:
                        print(f"[multi] FAILED {key} rc={rc}", flush=True)
                del slots[gpu]
            if gpu not in slots and pending:
                j = pending.pop(0)
                slots[gpu] = (launch(j, gpu), j)
        time.sleep(30)
    print("[multi] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
