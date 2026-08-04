#!/usr/bin/env python3
"""Compute-matched head-to-head: RSNN_TRSF given exactly GET's wall-clock budget.

The published comparison gives RSNN 500 epochs with early stopping (patience 50) while
GET runs its native fixed recipe (20 epochs on PDBbind, 10 on LBA). RSNN therefore spends
11-31x more wall clock per seed. This dispatcher re-runs the winning `dropout0.1` config
with `--epochs` capped to what fits inside GET's measured per-seed wall clock, so both
models get the same GPU time on the same hardware (1x A40).

Epoch caps = floor(GET_seconds_per_seed / RSNN_measured_seconds_per_epoch), taken from the
existing runs' `val_history[].dt_sec` and GET's SLURM log timestamps:

  split       GET s/seed   RSNN s/epoch   epochs that fit
  identity30      660           67               6
  identity60      618          109               5
  scaffold        642          116               5
  LBA             270           99               2

`--patience` is set above the cap so early stopping never fires; the trainer still selects
the best-validation-RMSE checkpoint within the budget and evaluates it on test, which is
the intended "train for at most T seconds, then report" protocol.

Everything else is byte-identical to the `dropout0.1` runs it is compared against.
Layout: runs/pdbbind_trsf/compute_matched/<dataset>/seed<S>/metrics.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"
MAX_RETRIES = 1
SEEDS = [42, 43, 44]

# CM_SCHEDULE=constant (default) reproduces the first compute-matched sweep; 'cosine'
# anneals the lr over the capped budget so the short run converges instead of stopping
# mid-flight at full lr. Same epoch caps either way, so compute stays matched.
SCHEDULE = os.environ.get("CM_SCHEDULE", "constant")
assert SCHEDULE in ("constant", "cosine"), f"bad CM_SCHEDULE={SCHEDULE}"
OUT_ROOT = REPO / ("runs/pdbbind_trsf/compute_matched"
                   + ("_cosine" if SCHEDULE == "cosine" else ""))

# dataset -> (rwnn cache dir, epoch cap matching GET's wall clock)
DATASETS = {
    "identity30": (REPO / "data/pdbbind/identity30", 6),
    "identity60": (REPO / "data/pdbbind/identity60", 5),
    "scaffold":   (REPO / "data/pdbbind/scaffold",   5),
    "lba":        (REPO / "data/lba/lba30",          2),
}

# the dropout0.1 winner config, verbatim apart from --epochs/--patience
BASE = [
    "--base", "transformer",
    "--h_dim", "120", "--num_layers", "3", "--nhead", "4", "--ffn_mult", "4",
    "--reduce", "mean",
    "--m", "6", "--w", "8", "--max_len", "448", "--ligand_max_len", "72",
    "--angles", "1", "--dihedrals", "1", "--angle_K", "8", "--dihedral_K", "4",
    "--distances", "0", "--mol_edge_feat", "0",
    "--batch_size", "8", "--num_workers", "8", "--standardize", "1",
    "--pos_enc", "rope", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.1",
]


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
_ONLY = set(sys.argv[1:])


def jobs():
    for ds, (ddir, cap) in DATASETS.items():
        if _ONLY and ds not in _ONLY:
            continue
        if not (Path(ddir) / "train.pt").exists():
            print(f"[cm] SKIP '{ds}' -- cache {ddir}/train.pt missing", flush=True)
            continue
        for seed in SEEDS:
            yield {"ds": ds, "ddir": str(ddir), "cap": cap, "seed": seed}


def job_dir(j):
    return OUT_ROOT / j["ds"] / f"seed{j['seed']}"


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
           "--data_dir", j["ddir"],
           "--epochs", str(j["cap"]),
           "--patience", str(j["cap"] + 10),
           "--lr_schedule", SCHEDULE] + BASE
    phys = VISIBLE[gpu]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=phys, OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[cm] slot={gpu} phys={phys} pid={p.pid} {j['ds']}/seed{j['seed']} "
          f"cap={j['cap']}ep", flush=True)
    return p


def main():
    all_jobs = list(jobs())
    pending = [j for j in all_jobs if not job_done(j)]
    print(f"[cm] GPUS={GPUS} | {len(pending)}/{len(all_jobs)} pending", flush=True)
    retries, slots = {}, {}
    while pending or slots:
        for gpu in GPUS:
            if gpu in slots:
                p, j = slots[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                key = f"{j['ds']}/seed{j['seed']}"
                if rc == 0 and job_done(j):
                    print(f"[cm] DONE {key}", flush=True)
                else:
                    n = retries.get(key, 0)
                    if n < MAX_RETRIES:
                        retries[key] = n + 1
                        pending.append(j)
                        print(f"[cm] RETRY {key} rc={rc}", flush=True)
                    else:
                        print(f"[cm] FAILED {key} rc={rc}", flush=True)
                del slots[gpu]
            if gpu not in slots and pending:
                j = pending.pop(0)
                slots[gpu] = (launch(j, gpu), j)
        time.sleep(5)
    print("[cm] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
