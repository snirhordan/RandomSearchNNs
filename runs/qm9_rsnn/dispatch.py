#!/usr/bin/env python3
"""GPU job dispatcher for the RSNN m-sweep on QM9.

Enumerates (target, m, split) triples for the RSNN search sampler, sorts
heaviest-first (large m, then random ordering across targets so the per-GPU
load is balanced), and runs them with at most ``--max_parallel`` concurrent
processes -- one per GPU.

Each job invokes ``quickstart/train_qm9.py`` with ``--walk_type search
--n_splits 1 --seed (42+split) --run_subdir m<m>/split<split>`` so the output
ends up at ``runs/qm9_rsnn/m<m>/<target>/split<split>/{metrics.json,
train.log}``.

Jobs that already have a non-empty ``metrics.json`` are skipped (idempotent
re-runs).
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
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable  # rwnn-env python3

TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "R2", "zpve",
           "U0", "U", "H", "G", "Cv"]
M_VALUES = [16, 8, 4, 1]  # heaviest first
SPLITS = [0, 1, 2]


def metrics_complete(out_dir: Path) -> bool:
    """Return True if metrics.json exists and contains a usable test MAE."""
    mp = out_dir / "metrics.json"
    if not mp.exists():
        return False
    try:
        with mp.open() as f:
            data = json.load(f)
        sm = data.get("summary", {})
        v = sm.get("mean_test_mae")
        return v is not None and v == v  # not NaN
    except Exception:
        return False


def build_jobs(out_root: Path,
               restrict_targets: Optional[list[str]] = None,
               restrict_m: Optional[list[int]] = None):
    jobs = []
    tgts = restrict_targets or TARGETS
    ms = restrict_m or M_VALUES
    for m in ms:
        for tgt in tgts:
            for s in SPLITS:
                out_dir = out_root / f"m{m}" / tgt / f"split{s}"
                if metrics_complete(out_dir):
                    continue
                jobs.append((m, tgt, s, out_dir))
    return jobs


def launch(m: int, tgt: str, split: int, out_dir: Path,
           gpu: int, args) -> subprocess.Popen:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    seed = 42 + split
    cmd = [
        PYTHON, "-u", "quickstart/train_qm9.py",
        "--target", tgt,
        "--walk_type", "search",
        "--distances", "0",
        "--mol_edge_feat", "0",
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
        "--device_idx", "0",
        "--out_root", str(args.out_root),
        "--run_subdir", f"m{m}/{tgt}/split{split}/_run",
        "--limit", str(args.limit),
    ]
    # train_qm9 puts metrics inside <out_root>/<run_subdir>/<target>; but we
    # want them at <out_root>/m<m>/<target>/split<split>/. We sidestep that
    # by setting run_subdir to ``m<m>/<target>/split<split>/_run`` and moving
    # files after the run. However, that requires post-processing. Cleaner:
    # set run_subdir so the final path coincides with our convention.
    # train_qm9: run_dir = out_root / run_subdir / target
    # Our convention: out_root / m<m> / target / split<split>
    # => run_subdir = m<m>/<placeholder>/split<split>, then we use target in
    # run_subdir? That double-includes target.
    # Cleanest: use run_subdir = "m<m>/_dummy_/split<split>" and have script
    # write to out_root/m<m>/_dummy_/split<split>/target, then move. Too messy.
    # Simpler: pass run_subdir = f"m{m}/{tgt}/split{split}" AND override
    # by adding a sibling so run_dir = out_root/run_subdir/target with
    # target == "" -- but train_qm9 always appends target.
    # Bottom line: I will write to out_root/m<m>/<target>/split<split>/<target>
    # and then move/rename in finalize. To avoid that, just use a flat scheme:
    # use --out_root=runs/qm9_rsnn/m<m>/<target>/split<split> and
    # --run_subdir="" so final path is runs/qm9_rsnn/m<m>/<target>/split<split>/<target>.
    # Actually simplest fix: pass out_root = runs/qm9_rsnn/_stage,
    # run_subdir = m<m>/<tgt>/split<split>, then rename _stage/m<m>/<tgt>/
    # split<split>/<tgt> -> runs/qm9_rsnn/m<m>/<tgt>/split<split> after.
    #
    # Re-design: keep things simple. Run with out_root=runs/qm9_rsnn/_stage,
    # run_subdir=m<m>/<tgt>/split<split>, then after completion move
    # _stage/m<m>/<tgt>/split<split>/<tgt>/* -> m<m>/<tgt>/split<split>/.
    # See ``_finalize_job`` below.
    raise RuntimeError("see launch_v2")


def launch_v2(m: int, tgt: str, split: int, final_out_dir: Path,
              gpu: int, args) -> subprocess.Popen:
    """Launch one training job pinned to GPU ``gpu``.

    The training script writes its outputs to a staging directory; the
    parent dispatcher moves them to ``final_out_dir`` once the process
    exits successfully (see ``finalize_job``).
    """
    final_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = final_out_dir / "train.log"
    seed = 42 + split

    stage_subdir = f"m{m}/{tgt}/split{split}"
    stage_out_root = args.out_root / "_stage"

    cmd = [
        PYTHON, "-u", "quickstart/train_qm9.py",
        "--target", tgt,
        "--walk_type", "search",
        "--distances", "0",
        "--mol_edge_feat", "0",
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
        "--device_idx", "0",
        "--out_root", str(stage_out_root),
        "--run_subdir", stage_subdir,
        "--limit", str(args.limit),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_f = open(log_path, "wb")
    log_f.write(
        f"# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n".encode()
    )
    log_f.write(f"# gpu: {gpu}\n# seed: {seed}\n".encode())
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                            stdout=log_f, stderr=subprocess.STDOUT)
    proc._dispatcher_meta = {  # type: ignore[attr-defined]
        "m": m, "tgt": tgt, "split": split, "gpu": gpu,
        "log_f": log_f, "stage_dir": stage_out_root / stage_subdir / tgt,
        "final_dir": final_out_dir, "started": time.time(),
    }
    return proc


def finalize_job(proc: subprocess.Popen) -> bool:
    meta = proc._dispatcher_meta  # type: ignore[attr-defined]
    try:
        meta["log_f"].close()
    except Exception:
        pass
    stage_dir = meta["stage_dir"]
    final_dir = meta["final_dir"]
    metrics_src = stage_dir / "metrics.json"
    if not metrics_src.exists():
        return False
    # Move metrics.json and (if present) ckpt to final_dir. Keep train.log
    # as it already lives at final_dir.
    try:
        # Avoid clobbering an existing metrics.json from a prior partial run
        # by overwriting it -- this run succeeded.
        target = final_dir / "metrics.json"
        if target.exists():
            target.unlink()
        metrics_src.rename(target)
        # Move any *.pt checkpoints out of staging; we do NOT want to commit
        # them, but keep them on disk for debugging.
        for ckpt in stage_dir.glob("*.pt"):
            try:
                ckpt.rename(final_dir / ckpt.name)
            except Exception:
                pass
        # Clean up the now-empty stage subdir if possible.
        try:
            stage_dir.rmdir()
            stage_dir.parent.rmdir()  # target dir under stage
            stage_dir.parent.parent.rmdir()  # mX dir under stage
        except OSError:
            pass
        return True
    except Exception as e:
        print(f"[finalize] failed for {final_dir}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path,
                    default=REPO / "runs" / "qm9_rsnn")
    ap.add_argument("--max_parallel", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--targets", default="", help="comma-separated subset")
    ap.add_argument("--ms", default="", help="comma-separated subset of m")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    restrict_t = [t for t in args.targets.split(",") if t] or None
    restrict_m = [int(m) for m in args.ms.split(",") if m] or None

    jobs = build_jobs(args.out_root, restrict_t, restrict_m)
    print(f"[dispatch] {len(jobs)} jobs to run "
          f"(max_parallel={args.max_parallel}, epochs={args.epochs}, "
          f"patience={args.patience}, limit={args.limit})")
    for m, t, s, od in jobs[:5]:
        print(f"  preview: m={m} tgt={t} split={s} -> {od}")
    if args.dry_run:
        return 0

    available_gpus = list(range(args.max_parallel))
    running: dict[int, subprocess.Popen] = {}  # gpu -> proc
    job_idx = 0
    completed = 0
    failed = 0
    t_start = time.time()

    while job_idx < len(jobs) or running:
        # Launch while GPUs available and jobs left.
        while available_gpus and job_idx < len(jobs):
            m, tgt, split, od = jobs[job_idx]
            job_idx += 1
            gpu = available_gpus.pop(0)
            proc = launch_v2(m, tgt, split, od, gpu, args)
            running[gpu] = proc
            print(f"[dispatch] [{job_idx}/{len(jobs)}] launch m={m} "
                  f"tgt={tgt:5s} split={split} -> gpu={gpu} pid={proc.pid}")

        # Wait for at least one to finish.
        if not running:
            break
        time.sleep(15)
        finished_gpus = []
        for gpu, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            meta = proc._dispatcher_meta  # type: ignore[attr-defined]
            dt = time.time() - meta["started"]
            ok = (rc == 0) and finalize_job(proc)
            tag = "OK" if ok else f"FAIL(rc={rc})"
            print(f"[dispatch] {tag} m={meta['m']} tgt={meta['tgt']:5s} "
                  f"split={meta['split']} gpu={gpu} dt={dt/60:.1f}min")
            finished_gpus.append(gpu)
            if ok:
                completed += 1
            else:
                failed += 1
        for gpu in finished_gpus:
            del running[gpu]
            available_gpus.append(gpu)

    total_dt = time.time() - t_start
    print(f"[dispatch] DONE completed={completed} failed={failed} "
          f"wall={total_dt/3600:.2f}h")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
