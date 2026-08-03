#!/usr/bin/env python3
"""Geometric-attention-bias pilot on PDBbind-Benchmark identity30.

Tests whether the geometric attention bias (torsion/distance injected into
attention, ``RSNN_TRSF_Reg(geom_bias=True)``) -- the single best lever on QM9,
never before wired into the protein-ligand trainer -- moves the param-matched
transformer past its established identity30 parity with GET.

Config = the established best (``dropout0.1``: h120/nl3/nhead4/rope/full/
dropout0.1, angles+dihedrals ON) PLUS ``--geom_bias 1`` (adds ~1,444 params ->
~693.5k, still param-matched to GET 719,356). Two phases:

  Phase A (cutoff sweep): seed 42, ``geom_rbf_cutoff in {5,10,15}``. QM9's 5 A
      default is tuned for small molecules; protein pockets span larger.
  Phase B (head-to-head): the best cutoff (selected by VALIDATION Pearson at the
      early-stopping checkpoint -- leakage-free, never touches test) x seeds
      {42,43,44}. Seed 42 for the winning cutoff is reused from Phase A.

Compare the 3-seed test Pearson against the two identity30 anchors:
GET 0.5887 +/- 0.0095 and RSNN-no-bias 0.5847 +/- 0.0119.

Idempotent (skips any run whose metrics.json already has test_metrics),
adaptive GPU count. Layout:
``runs/pdbbind_trsf/geom_bias/identity30/cut<C>/seed<S>/{metrics.json,...}``.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"
OUT_ROOT = REPO / "runs/pdbbind_trsf/geom_bias/identity30"
DATA_DIR = REPO / "data/pdbbind/identity30"
MAX_RETRIES = 1

CUTOFFS = [5, 10, 15]          # Phase A sweep, seed 42
SEEDS_FULL = [42, 43, 44]      # Phase B, best cutoff

# Established dropout0.1 config + the geometric attention bias. Only
# --geom_rbf_cutoff varies per run.
BASE = [
    "--base", "transformer",
    "--h_dim", "120", "--num_layers", "3", "--nhead", "4", "--ffn_mult", "4",
    "--reduce", "mean",
    "--m", "6", "--w", "8", "--max_len", "448", "--ligand_max_len", "72",
    "--angles", "1", "--dihedrals", "1", "--angle_K", "8", "--dihedral_K", "4",
    "--distances", "0", "--mol_edge_feat", "0",
    "--epochs", "500", "--patience", "50", "--batch_size", "8",
    "--num_workers", "8", "--standardize", "1",
    "--pos_enc", "rope", "--attn_mode", "full", "--lr", "1e-3", "--dropout", "0.1",
    "--geom_bias", "1", "--geom_rbf_K", "16",
    "--geom_angle_K", "8", "--geom_dihedral_K", "4", "--geom_hidden", "32",
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


def job_dir(j):
    return OUT_ROOT / f"cut{j['cut']}" / f"seed{j['seed']}"


def job_done(j):
    m = job_dir(j) / "metrics.json"
    if not m.exists():
        return False
    try:
        with open(m) as f:
            return "test_metrics" in json.load(f)
    except Exception:
        return False


def val_pearson(j):
    """Validation Pearson at the early-stopping (best-val-RMSE) checkpoint.

    Leakage-free model-selection signal for choosing the cutoff: never reads
    the test split. Falls back to the max val Pearson over training if the
    best_epoch row is missing.
    """
    m = job_dir(j) / "metrics.json"
    try:
        with open(m) as f:
            d = json.load(f)
        be = d.get("best_epoch")
        vh = d.get("val_history", [])
        for row in vh:
            if row.get("epoch") == be:
                return float(row.get("pearson_r"))
        return max((float(r.get("pearson_r", 0.0)) for r in vh), default=None)
    except Exception:
        return None


def launch(j, gpu):
    d = job_dir(j)
    d.mkdir(parents=True, exist_ok=True)
    logf = open(d / "dispatch.log", "a")
    cmd = [PYTHON, "-u", "-m", "quickstart.train_pdbbind",
           "--out_dir", str(d), "--seed", str(j["seed"]),
           "--data_dir", str(DATA_DIR),
           "--geom_rbf_cutoff", str(float(j["cut"]))] + BASE
    phys = VISIBLE[gpu]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=phys, OMP_NUM_THREADS="2")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO), env=env)
    print(f"[geom] slot={gpu} phys={phys} pid={p.pid} cut{j['cut']}/seed{j['seed']}",
          flush=True)
    return p


def run_jobs(jobs):
    """Slot-schedule ``jobs`` over the allocated GPUs to completion (idempotent)."""
    pending = [j for j in jobs if not job_done(j)]
    print(f"[geom] GPUS={GPUS} | {len(pending)}/{len(jobs)} pending", flush=True)
    retries = {}
    slots = {}
    while pending or slots:
        for gpu in GPUS:
            if gpu in slots:
                p, j = slots[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                key = f"cut{j['cut']}/seed{j['seed']}"
                if rc == 0 and job_done(j):
                    print(f"[geom] DONE {key}", flush=True)
                else:
                    n = retries.get(key, 0)
                    if n < MAX_RETRIES:
                        retries[key] = n + 1
                        pending.append(j)
                        print(f"[geom] RETRY {key} rc={rc}", flush=True)
                    else:
                        print(f"[geom] FAILED {key} rc={rc}", flush=True)
                del slots[gpu]
            if gpu not in slots and pending:
                j = pending.pop(0)
                slots[gpu] = (launch(j, gpu), j)
        time.sleep(30)


def main():
    # --- "full" mode: run the complete CUTOFFS x SEEDS_FULL grid (idempotent).
    # Used after the val-based cutoff selection proved too noisy at n=1 seed
    # (cut15 had best val but worst test): get 3 seeds for EVERY cutoff so the
    # geom_bias verdict rests on 3-seed means, not a single-seed selection.
    if "full" in sys.argv[1:]:
        grid = [{"cut": c, "seed": s} for c in CUTOFFS for s in SEEDS_FULL]
        print(f"[geom] === FULL GRID: {CUTOFFS} x {SEEDS_FULL} (idempotent) ===", flush=True)
        run_jobs(grid)
        print("[geom] FULL GRID DONE", flush=True)
        return 0

    # --- Phase A: cutoff sweep at seed 42 ---
    phase_a = [{"cut": c, "seed": 42} for c in CUTOFFS]
    print("[geom] === Phase A: cutoff sweep {5,10,15} @ seed42 ===", flush=True)
    run_jobs(phase_a)

    # --- select best cutoff by validation Pearson (leakage-free) ---
    scored = {c: val_pearson({"cut": c, "seed": 42}) for c in CUTOFFS}
    valid = {c: v for c, v in scored.items() if v is not None}
    if not valid:
        print("[geom] ERROR: no Phase-A val Pearson available; aborting Phase B",
              flush=True)
        print(f"[geom] scored={scored}", flush=True)
        return 1
    best_cut = max(valid, key=valid.get)
    sel = {
        "phase_a_val_pearson_by_cutoff": scored,
        "selected_cutoff": best_cut,
        "selection_metric": "validation Pearson at best-val-RMSE checkpoint",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "cutoff_selection.json", "w") as f:
        json.dump(sel, f, indent=2)
    print(f"[geom] selected cutoff={best_cut} A (val Pearson {valid[best_cut]:.4f}); "
          f"all={scored}", flush=True)

    # --- Phase B: best cutoff x remaining seeds (42 already done in Phase A) ---
    phase_b = [{"cut": best_cut, "seed": s} for s in SEEDS_FULL if s != 42]
    print(f"[geom] === Phase B: cut{best_cut} @ seeds {[s for s in SEEDS_FULL if s != 42]} ===",
          flush=True)
    run_jobs(phase_b)

    print("[geom] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
