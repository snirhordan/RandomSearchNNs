#!/usr/bin/env python3
"""GPU job dispatcher for the d-RWNN m-sweep on QM9.

Mirror of runs/qm9_rsnn/dispatch.py, but with walk_type=walk_ada (d-RWNN
sampler) and distances=1, mol_edge_feat=1 (best d-RWNN config from PR #3).
Outputs:  runs/qm9_rwnn/m<m>/<target>/split<split>/metrics.json

Designed to share a host with the EGNN dispatcher: --gpus 2,3 by default
so it runs alongside EGNN on GPUs 0/1.
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

TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "R2", "zpve",
           "U0", "U", "H", "G", "Cv"]
M_VALUES = [16, 8, 4]  # heaviest first
SPLITS = [0, 1, 2]


def metrics_complete(metrics_path: Path) -> bool:
    if not metrics_path.exists() or metrics_path.stat().st_size < 100:
        return False
    try:
        with metrics_path.open() as f:
            data = json.load(f)
        sm = data.get("summary", {})
        v = sm.get("mean_test_mae")
        return v is not None and v == v
    except Exception:
        return False


def build_jobs(out_root: Path, targets, ms):
    jobs = []
    for m in ms:
        for t in targets:
            for s in SPLITS:
                mp = out_root / f"m{m}" / t / f"split{s}" / "metrics.json"
                if metrics_complete(mp):
                    continue
                jobs.append((m, t, s))
    return jobs


def launch(m: int, tgt: str, split: int, gpu: int, args) -> subprocess.Popen:
    final_dir = args.out_root / f"m{m}" / tgt / f"split{split}"
    final_dir.mkdir(parents=True, exist_ok=True)
    log_path = final_dir / "train.log"
    seed = 42 + split
    stage_subdir = f"m{m}/{tgt}/split{split}"
    stage_out_root = args.out_root / "_stage"

    cmd = [
        PYTHON, "-u", str(REPO / "quickstart" / "train_qm9.py"),
        "--target", tgt,
        "--walk_type", "walk_ada",
        "--distances", "1",
        "--mol_edge_feat", "1",
        "--epochs", str(args.epochs),
        "--early_stopping", str(args.patience),
        "--n_splits", "1",
        "--batch_size", str(args.batch_size),
        "--h_dim", "128",
        "--num_layers", "2",
        "--m", str(m),
        "--w", "8",
        "--reduce", "mean",
        "--seed", str(seed),
        "--num_workers", str(args.num_workers),
        "--device_idx", "0",  # remapped via CUDA_VISIBLE_DEVICES
        "--out_root", str(stage_out_root),
        "--run_subdir", stage_subdir,
        "--limit", str(args.limit),
    ]
    env = os.environ.copy()
    # Always pin children to the requested local GPU index. Under Slurm,
    # CUDA_VISIBLE_DEVICES is a subset (the allocated cgroup GPUs); setting
    # it to a single relative index further restricts the child to one of
    # those allowed devices, which is exactly what we want for packing.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Cap PyTorch intra-op threads so packed procs don't oversubscribe the CPU.
    # Each RWNN run also spawns --num_workers DataLoader workers, so keep the
    # main thread count low to leave room for them within the per-proc budget.
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    log_f = open(log_path, "wb")
    log_f.write(f"# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n".encode())
    log_f.write(f"# gpu: {gpu} seed: {seed} (CVD={env.get('CUDA_VISIBLE_DEVICES', 'unset')})\n".encode())
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                            stdout=log_f, stderr=subprocess.STDOUT)
    proc._meta = {
        "m": m, "tgt": tgt, "split": split, "gpu": gpu,
        "log_f": log_f, "stage_dir": stage_out_root / stage_subdir / tgt,
        "final_dir": final_dir, "started": time.time(),
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
        # Move ckpts if any
        for ckpt in meta["stage_dir"].glob("*.pt"):
            try:
                ckpt.rename(meta["final_dir"] / ckpt.name)
            except Exception:
                pass
        for d in (meta["stage_dir"], meta["stage_dir"].parent,
                  meta["stage_dir"].parent.parent):
            try:
                d.rmdir()
            except OSError:
                pass
        return True
    except Exception as e:
        print(f"[finalize] failed for {meta['final_dir']}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path,
                    default=REPO / "runs" / "qm9_rwnn")
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--targets", default="")
    ap.add_argument("--ms", default="")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    targets = [t for t in args.targets.split(",") if t] or TARGETS
    ms = [int(m) for m in args.ms.split(",") if m] or M_VALUES
    gpus = [int(g) for g in args.gpus.split(",") if g]

    jobs = build_jobs(args.out_root, targets, ms)
    print(f"[dispatch-rwnn] {len(jobs)} jobs (gpus={gpus}, epochs={args.epochs}, "
          f"patience={args.patience})")
    for m, t, s in jobs[:5]:
        print(f"  preview: m={m} tgt={t} split={s}")
    if args.dry_run:
        return 0

    available = list(gpus)
    # Use a list of procs (not dict keyed by gpu) so packed configurations
    # like --gpus 2,2,2,3,3,3 don't overwrite earlier procs on the same GPU.
    # When a proc finishes we put its gpu (from meta) back on available.
    running: list[subprocess.Popen] = []
    idx = 0
    done = 0
    fail = 0
    t0 = time.time()

    while idx < len(jobs) or running:
        while available and idx < len(jobs):
            m, t, s = jobs[idx]
            idx += 1
            g = available.pop(0)
            proc = launch(m, t, s, g, args)
            running.append(proc)
            print(f"[dispatch-rwnn] [{idx}/{len(jobs)}] launch m={m} "
                  f"tgt={t:5s} split={s} -> gpu={g} pid={proc.pid}")

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
            print(f"[dispatch-rwnn] {tag} m={meta['m']} tgt={meta['tgt']:5s} "
                  f"split={meta['split']} gpu={meta['gpu']} "
                  f"dt={dt/60:.1f}min")
            if ok:
                done += 1
            else:
                fail += 1
            available.append(meta["gpu"])
        running = still

    total = time.time() - t0
    print(f"[dispatch-rwnn] DONE completed={done} failed={fail} "
          f"wall={total/3600:.2f}h")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
