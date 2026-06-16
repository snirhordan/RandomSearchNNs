#!/usr/bin/env python3
"""Gap apples-to-apples dispatcher: RSNN + d-RWNN × 3 seeds on Cormorant split.

Launches 6 cells (2 models × 3 seeds {42,43,44}) for the gap-only comparison
against the existing EGNN gap baselines. Each cell:
  - Cormorant fixed split (100k/17748/13083, --split cormorant).
  - EGNN-style meann/MAD normalization imported from preprocessing_audit.json.
  - CosineAnnealingLR(T_max=epochs=300), stepped 1x/epoch.
  - L1 loss (matching EGNN).
  - Adam + patience=50, batch=96, lr=7.5e-4 (linear-rescaled from 1e-3 @ batch=128).
  - m=16 walks, w=8 walk width.
  - RSNN: walk_type=search, distances=0, mol_edge_feat=0.
  - d-RWNN: walk_type=walk_ada, distances=1, mol_edge_feat=0.

Output layout (after finalize):
  runs/qm9_compare/rsnn/seed{42,43,44}/metrics.json
  runs/qm9_compare/d_rwnn/seed{42,43,44}/metrics.json

3-packed on GPUs 2,3 by default (EGNN gap baselines already done on 0,1).
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"

TARGET = "gap"
SEEDS = [42, 43, 44]
MODELS = ["rsnn", "d_rwnn"]
# Per-model flag overrides. "m"/"w"/"reduce" are optional; if absent, the
# corresponding args.* default is used.
MODEL_FLAGS = {
    "rsnn":         {"walk_type": "search",   "distances": 0, "mol_edge_feat": 0},
    "d_rwnn":       {"walk_type": "walk_ada", "distances": 1, "mol_edge_feat": 0},
    "rsnn_d1_m8":   {"walk_type": "search",   "distances": 1, "mol_edge_feat": 0, "m": 8},
    "rsnn_d1_m16":  {"walk_type": "search",   "distances": 1, "mol_edge_feat": 0, "m": 16},
    # Variant matrix exploring (m, w, mol_edge_feat, reduce) at fixed
    # h_dim=128 / num_layers=2 (same ~743k params, same 300 epochs).
    # Goal: outperform EGNN on gap (test_mae < 0.050 eV).
    "V1":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 16, "w": 8,  "reduce": "sum"},
    "V2":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 32, "w": 8,  "reduce": "sum"},
    "V3":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 64, "w": 8,  "reduce": "sum"},
    "V4":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 16, "w": 16, "reduce": "sum"},
    "V5":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 16, "w": 24, "reduce": "sum"},
    "V6":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 32, "w": 16, "reduce": "sum"},
    "V7":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 16, "w": 8,  "reduce": "mean"},
    "V8":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 0, "m": 16, "w": 8,  "reduce": "sum"},
    "V9":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8,  "w": 16, "reduce": "sum"},
    "V10": {"walk_type": "search", "distances": 1, "mol_edge_feat": 0, "m": 32, "w": 8,  "reduce": "sum"},
    "V11": {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 16, "w": 8,  "reduce": "max"},
    "V12": {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 64, "w": 16, "reduce": "sum"},
    # ---- Optimization sweep (O-series) ----
    # Base config from V9 (best 300-epoch completion at 0.0926 eV):
    # walk_type=search, distances=1, mol_edge_feat=1, m=8, w=16, reduce=sum.
    # All bidirectional (RSNN_LSTM_Reg is hardcoded bi). grad_clip=1.0 universal.
    # Width/depth combos verified at ~743k +/-5% params by count_params.py.
    # Group A: width/depth tradeoff (5 variants at constant ~743k params).
    "O_A1_h188_L1": {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 188, "num_layers": 1, "lstm_init": "orthogonal"},
    "O_A2_h128_L2": {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 128, "num_layers": 2},
    "O_A3_h104_L3": {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 104, "num_layers": 3, "lstm_init": "orthogonal"},
    "O_A4_h88_L4":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 88,  "num_layers": 4, "lstm_init": "orthogonal"},
    "O_A5_h72_L6":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 72,  "num_layers": 6, "lstm_init": "orthogonal"},
    # Group B: training-trick ablations at baseline (h=128, L=2).
    "O_B1_adamw":   {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 128, "num_layers": 2,
                     "optimizer": "adamw", "weight_decay": 1e-4},
    "O_B2_dropout": {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 128, "num_layers": 2, "dropout": 0.15},
    "O_B3_warmup":  {"walk_type": "search", "distances": 1, "mol_edge_feat": 1, "m": 8, "w": 16, "reduce": "sum",
                     "grad_clip": 1.0, "h_dim": 128, "num_layers": 2,
                     "warmup_epochs": 10, "optimizer": "adamw", "weight_decay": 1e-4},
}


def metrics_complete(metrics_path: Path) -> bool:
    if not metrics_path.exists() or metrics_path.stat().st_size < 100:
        return False
    try:
        with metrics_path.open() as f:
            data = json.load(f)
        sm = data.get("summary", {})
        v = sm.get("mean_test_mae")
        if v is not None and v == v:
            return True
        sp = data.get("splits") or []
        if sp and sp[0].get("test_mae") is not None:
            return True
    except Exception:
        pass
    return False


def build_jobs(out_root: Path, models, seeds):
    jobs = []
    for m in models:
        for s in seeds:
            final = out_root / m / f"seed{s}" / "metrics.json"
            if metrics_complete(final):
                continue
            jobs.append((m, s))
    return jobs


def launch(model: str, seed: int, gpu: int, args) -> subprocess.Popen:
    final_dir = args.out_root / model / f"seed{seed}"
    final_dir.mkdir(parents=True, exist_ok=True)
    log_path = final_dir / "train.log"
    stage_subdir = f"{model}/seed{seed}"
    stage_out_root = args.out_root / "_stage"
    flags = MODEL_FLAGS[model]

    cmd = [
        PYTHON, "-u", str(REPO / "quickstart" / "train_qm9.py"),
        "--target", TARGET,
        "--split", "cormorant",
        "--cormorant_data_dir", str(REPO / "external" / "egnn" / "qm9" / "temp" / "qm9"),
        "--lr_scheduler", "cosine",
        "--use_egnn_normalization", "1",
        "--norm_constants_json",
        str(args.out_root / "preprocessing_audit.json"),
        "--walk_type", flags["walk_type"],
        "--distances", str(flags["distances"]),
        "--mol_edge_feat", str(flags["mol_edge_feat"]),
        "--epochs", str(args.epochs),
        "--early_stopping", str(args.patience),
        "--n_splits", "1",
        "--batch_size", str(args.batch_size),
        "--h_dim", str(flags.get("h_dim", 128)),
        "--num_layers", str(flags.get("num_layers", 2)),
        "--m", str(flags.get("m", args.m)),
        "--w", str(flags.get("w", 8)),
        "--reduce", flags.get("reduce", "mean"),
        "--lr", str(args.lr),
        "--seed", str(seed),
        "--num_workers", str(args.num_workers),
        # Training-optimization flags (defaults match legacy behavior).
        "--grad_clip", str(flags.get("grad_clip", 0.0)),
        "--weight_decay", str(flags.get("weight_decay", 0.0)),
        "--optimizer", str(flags.get("optimizer", "adam")),
        "--lstm_init", str(flags.get("lstm_init", "default")),
        "--dropout", str(flags.get("dropout", 0.0)),
        "--warmup_epochs", str(flags.get("warmup_epochs", 0)),
        "--device_idx", "0",  # remapped via CUDA_VISIBLE_DEVICES below
        "--out_root", str(stage_out_root),
        "--run_subdir", stage_subdir,
        "--limit", str(args.limit),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Cap PyTorch intra-op threads — packed RSNN/d-RWNN procs are CPU-heavy.
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    # Determinism (per spec).
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    log_f = open(log_path, "wb")
    log_f.write(f"# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n".encode())
    log_f.write(f"# gpu: {gpu} seed: {seed} model: {model} "
                f"(CVD={env['CUDA_VISIBLE_DEVICES']})\n".encode())
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                            stdout=log_f, stderr=subprocess.STDOUT)
    proc._meta = {
        "model": model, "seed": seed, "gpu": gpu,
        "log_f": log_f,
        # train_qm9.py writes to <out_root>/<run_subdir>/<target>/metrics.json
        "stage_dir":
            stage_out_root / stage_subdir / TARGET,
        "final_dir": final_dir,
        "started": time.time(),
    }
    return proc


def finalize(proc: subprocess.Popen) -> bool:
    meta = proc._meta
    try:
        meta["log_f"].close()
    except Exception:
        pass
    src = meta["stage_dir"] / "metrics.json"
    if not src.exists():
        return False
    dst = meta["final_dir"] / "metrics.json"
    try:
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        # Move any ckpts the trainer left behind.
        for ckpt in meta["stage_dir"].glob("*.pt"):
            try:
                ckpt.rename(meta["final_dir"] / ckpt.name)
            except Exception:
                pass
        # Try to rmdir empty stage parents.
        for d in (meta["stage_dir"], meta["stage_dir"].parent,
                  meta["stage_dir"].parent.parent):
            try:
                d.rmdir()
            except OSError:
                pass
        return True
    except Exception as e:
        print(f"[finalize] failed for {meta['final_dir']}: {e}",
              file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path,
                    default=REPO / "runs" / "qm9_compare")
    ap.add_argument("--gpus", default="2,2,2,3,3,3")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=7.5e-4)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 to limit dataset (debug / dry-run).")
    ap.add_argument("--models", default="",
                    help="comma-separated subset, default=all (rsnn,d_rwnn).")
    ap.add_argument("--seeds", default="",
                    help="comma-separated subset, default=all (42,43,44).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Plan only, do not launch procs.")
    args = ap.parse_args()

    # Audit JSON is mandatory.
    audit = args.out_root / "preprocessing_audit.json"
    if not audit.exists():
        sys.exit(f"missing {audit} — run preprocessing_audit.py first")

    models = [m for m in args.models.split(",") if m] or MODELS
    seeds = [int(s) for s in args.seeds.split(",") if s] or SEEDS
    gpus = [int(g) for g in args.gpus.split(",") if g]

    jobs = build_jobs(args.out_root, models, seeds)
    print(f"[dispatch-compare] {len(jobs)} jobs "
          f"(gpus={gpus}, epochs={args.epochs}, patience={args.patience})")
    for m, s in jobs[:6]:
        print(f"  preview: model={m} seed={s}")
    if args.dry_run:
        return 0

    available = list(gpus)
    running: list[subprocess.Popen] = []
    idx = 0
    done = 0
    fail = 0
    t0 = time.time()

    while idx < len(jobs) or running:
        while available and idx < len(jobs):
            m, s = jobs[idx]
            idx += 1
            g = available.pop(0)
            proc = launch(m, s, g, args)
            running.append(proc)
            print(f"[dispatch-compare] [{idx}/{len(jobs)}] launch "
                  f"model={m:6s} seed={s} -> gpu={g} pid={proc.pid}")

        if not running:
            break
        time.sleep(15)
        still: list[subprocess.Popen] = []
        for proc in running:
            rc = proc.poll()
            if rc is None:
                still.append(proc)
                continue
            meta = proc._meta
            dt = time.time() - meta["started"]
            ok = (rc == 0) and finalize(proc)
            tag = "OK" if ok else f"FAIL(rc={rc})"
            print(f"[dispatch-compare] {tag} model={meta['model']:6s} "
                  f"seed={meta['seed']} gpu={meta['gpu']} "
                  f"dt={dt/60:.1f}min")
            if ok:
                done += 1
            else:
                fail += 1
            available.append(meta["gpu"])
        running = still

    total = time.time() - t0
    print(f"[dispatch-compare] DONE completed={done} failed={fail} "
          f"wall={total/3600:.2f}h")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
