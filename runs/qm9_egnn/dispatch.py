#!/usr/bin/env python3
"""GPU job dispatcher for the EGNN sweep on QM9.

Runs (target, seed) pairs through ``runs/qm9_egnn/run_one.py``, which wraps
vgsatorras/egnn's ``main_qm9.py``. Idempotent (skips populated metrics.json),
multi-GPU round-robin.

Default: 12 targets x 3 seeds (42/43/44) x 300 epochs at nf=128, n_layers=7.
Designed to share a host with the RWNN m-sweep dispatcher: set
``--gpus 0,1`` to pin EGNN to GPUs 0/1 while RWNN takes 2/3.
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
PY = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"

TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "R2", "zpve",
           "U0", "U", "H", "G", "Cv"]
SEEDS = [42, 43, 44]


def metrics_complete(metrics_path: Path) -> bool:
    if not metrics_path.exists() or metrics_path.stat().st_size < 100:
        return False
    try:
        with metrics_path.open() as f:
            data = json.load(f)
        v = data.get("splits", [{}])[0].get("test_mae")
        return v is not None and v == v
    except Exception:
        return False


def build_jobs(out_root: Path, targets, seeds):
    jobs = []
    for t in targets:
        for s in seeds:
            mp = out_root / t / f"seed{s}" / "metrics.json"
            if metrics_complete(mp):
                continue
            jobs.append((t, s))
    return jobs


def launch(target: str, seed: int, gpu: int, args) -> subprocess.Popen:
    out_dir = args.out_root / target / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "dispatch.log"
    cmd = [
        PY, "-u", str(REPO / "runs" / "qm9_egnn" / "run_one.py"),
        "--target", target,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--nf", str(args.nf),
        "--n_layers", str(args.n_layers),
        "--attention", str(args.attention),
        "--node_attr", str(args.node_attr),
        "--lr", str(args.lr),
        "--out_root", str(args.out_root),
        "--device_idx", "0",  # remapped via CUDA_VISIBLE_DEVICES below
    ]
    env = os.environ.copy()
    # When running under Slurm, CUDA_VISIBLE_DEVICES is already pinned by
    # the cgroup to the allocated device(s); do NOT override it (that would
    # target a different physical GPU and trip the cgroup denial).
    # Outside Slurm, pin children to the requested local GPU index.
    if "SLURM_JOB_ID" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_f = open(log_path, "wb")
    log_f.write(f"# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n".encode())
    log_f.write(f"# gpu: {gpu} seed: {seed} (CVD={env.get('CUDA_VISIBLE_DEVICES', 'unset')})\n".encode())
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                            stdout=log_f, stderr=subprocess.STDOUT)
    proc._meta = {
        "target": target, "seed": seed, "gpu": gpu,
        "log_f": log_f, "started": time.time(),
    }
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path,
                    default=REPO / "runs" / "qm9_egnn")
    ap.add_argument("--gpus", default="0,1",
                    help="comma-separated GPU indices")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--nf", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=7)
    ap.add_argument("--attention", type=int, default=1)
    ap.add_argument("--node_attr", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--targets", default="",
                    help="comma-separated subset of targets")
    ap.add_argument("--seeds", default="",
                    help="comma-separated subset of seeds")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    targets = [t for t in args.targets.split(",") if t] or TARGETS
    seeds = [int(s) for s in args.seeds.split(",") if s] or SEEDS
    gpus = [int(g) for g in args.gpus.split(",") if g]

    jobs = build_jobs(args.out_root, targets, seeds)
    print(f"[dispatch-egnn] {len(jobs)} jobs (gpus={gpus}, epochs={args.epochs}, "
          f"nf={args.nf}, n_layers={args.n_layers})")
    for j in jobs[:5]:
        print(f"  preview: tgt={j[0]} seed={j[1]}")
    if args.dry_run:
        return 0

    available = list(gpus)
    running: dict[int, subprocess.Popen] = {}
    idx = 0
    done = 0
    fail = 0
    t0 = time.time()

    while idx < len(jobs) or running:
        while available and idx < len(jobs):
            t, s = jobs[idx]
            idx += 1
            g = available.pop(0)
            proc = launch(t, s, g, args)
            running[g] = proc
            print(f"[dispatch-egnn] [{idx}/{len(jobs)}] launch tgt={t} "
                  f"seed={s} -> gpu={g} pid={proc.pid}")

        if not running:
            break
        time.sleep(30)
        finished = []
        for g, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            meta = proc._meta
            try:
                meta["log_f"].close()
            except Exception:
                pass
            dt = time.time() - meta["started"]
            mp = args.out_root / meta["target"] / f"seed{meta['seed']}" / "metrics.json"
            ok = (rc == 0) and metrics_complete(mp)
            tag = "OK" if ok else f"FAIL(rc={rc})"
            print(f"[dispatch-egnn] {tag} tgt={meta['target']:5s} "
                  f"seed={meta['seed']} gpu={g} dt={dt/60:.1f}min")
            finished.append(g)
            if ok:
                done += 1
            else:
                fail += 1
        for g in finished:
            del running[g]
            available.append(g)

    total = time.time() - t0
    print(f"[dispatch-egnn] DONE completed={done} failed={fail} "
          f"wall={total/3600:.2f}h")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
