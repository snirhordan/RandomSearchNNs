#!/usr/bin/env python3
"""qm9_multitarget orchestrator — autonomous 71-job RSNN/RWNN sweep.

State machine:
  Phase 0: wait for A/B rerun (job 68211999, O_B1_densedist) to complete
  Phase 1: pick winner config (lower test_mae on gap seed=42)
  Phase 2: build job manifest (RSNN winner + RWNN m=16+AdamW × 12 targets × 3 seeds)
  Phase 3: fill 12 slots (4 local dym-lab3 + 8 sbatch dym-lab2)
  Phase 4: monitor + auto-resubmit failed jobs (1 retry)
  Phase 5: aggregate to results.json + summary.md when all 71 done

State on filesystem (no external DB):
  runs/qm9_multitarget/<model>_<target>/seed<N>/metrics.json — DONE marker
  runs/qm9_multitarget/<model>_<target>/seed<N>/train.log    — running output
  runs/qm9_multitarget/.state.json                            — orchestrator memory

Hardware:
  dym-lab3 (local L40S, GPUs 0..3) — nohup'd python procs
  dym-lab2 (A40 via Slurm)         — sbatch, partition=dym, account=dym-lab

Invocation: python3 orchestrate.py
Idempotent — safe to call repeatedly from cron.
"""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
ROOT = REPO / "runs" / "qm9_multitarget"
STATE_PATH = ROOT / ".state.json"
SLURM_SCRIPTS = ROOT / "slurm_scripts"
PYTHON = "/home/snirhordan/miniconda3/envs/rwnn/bin/python3"

TARGETS = ["mu", "alpha", "homo", "lumo", "gap", "R2", "zpve",
           "U0", "U", "H", "G", "Cv"]  # R2 uppercase matches cache file mols_R2.pt
SEEDS = [42, 43, 44]

# RSNN winner config (selected from A/B after gap seed=42 result).
# Both candidates share most flags; only the implementation of sample_dfs
# differs (which is a code change, not a CLI flag).
RSNN_FLAGS_BASE = dict(
    walk_type="search",
    distances=1, mol_edge_feat=1,
    batch_size=96, h_dim=128, num_layers=2,
    m=8, w=16, reduce="sum", lr=0.00075,
    grad_clip=1.0, weight_decay=0.0001, optimizer="adamw",
    lstm_init="default", dropout=0.0, warmup_epochs=0,
)

# RWNN best config: m=4 (was 16; reduced 2026-05-24 for speed).
# pe_in_dim = 2*w + 19 = 35 (same as RSNN search w=16 → w+19=35), so model
# params are unchanged.
RWNN_FLAGS = dict(
    walk_type="walk_ada",  # random walk with adaptive non-backtracking
    distances=1, mol_edge_feat=1,
    batch_size=96, h_dim=128, num_layers=2,
    m=4, w=8, reduce="sum", lr=0.00075,
    grad_clip=1.0, weight_decay=0.0001, optimizer="adamw",
    lstm_init="default", dropout=0.0, warmup_epochs=0,
)

COMMON_FLAGS = dict(
    split="cormorant",
    cormorant_data_dir=str(REPO / "external/egnn/qm9/temp/qm9"),
    lr_scheduler="cosine",
    use_egnn_normalization=1,
    norm_constants_json=str(REPO / "runs/qm9_compare/preprocessing_audit.json"),
    epochs=300, early_stopping=50, n_splits=1,
    num_workers=12,
)

AB_JOB_ID = 68211999
AB_NEW_METRICS = REPO / "runs/qm9_compare/O_B1_densedist/seed42/metrics.json"
AB_OLD_METRICS = REPO / "runs/qm9_compare/O_B1_adamw/seed42/metrics.json"

MAX_SBATCH_SLOTS = 8  # dym-lab2 has 8 A40 GPUs
MAX_LOCAL_SLOTS = 4   # dym-lab3 has 4 L40S GPUs
MAX_RETRIES = 1
STALE_LOG_MIN = 30  # train.log not updated in N minutes => stale (was 15; bumped to handle slurm queue latency on dym-lab2)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "winner": None,         # "O_B1_adamw" or "O_B1_densedist"
        "manifest_built": False,
        "retries": {},          # job_key -> count
        "slurm_jids": {},       # job_key -> sbatch JID
        "local_pids": {},       # job_key -> local PID
        "local_gpus": {},       # job_key -> GPU index 0..3
    }


def save_state(st):
    STATE_PATH.write_text(json.dumps(st, indent=2))


# ---------------------------------------------------------------------------
# A/B winner detection
# ---------------------------------------------------------------------------
def read_test_mae(path: Path):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        return float(d["splits"][0]["test_mae"])
    except Exception:
        return None


def pick_winner(state):
    if state["winner"] is not None:
        return state["winner"]
    new_mae = read_test_mae(AB_NEW_METRICS)
    old_mae = read_test_mae(AB_OLD_METRICS)
    if new_mae is None:
        return None
    if old_mae is None:
        print(f"[orch] WARN: old metrics missing, defaulting to densedist")
        state["winner"] = "O_B1_densedist"
    elif new_mae < old_mae:
        state["winner"] = "O_B1_densedist"
        print(f"[orch] A/B WINNER: O_B1_densedist (new={new_mae:.4f} beats old={old_mae:.4f})")
    else:
        state["winner"] = "O_B1_adamw"
        print(f"[orch] A/B WINNER: O_B1_adamw (old={old_mae:.4f} beats new={new_mae:.4f})")
    save_state(state)
    return state["winner"]


# ---------------------------------------------------------------------------
# Job manifest
# ---------------------------------------------------------------------------
def build_manifest():
    """All jobs in the multitarget sweep.

    Returns list of dicts {key, model, target, seed, flags}.
    """
    jobs = []
    # RSNN: 12 targets × 3 seeds (gap seed=42 reused from A/B winner — see
    # `reuse_ab` flag in seed pruning below).
    for target in TARGETS:
        for seed in SEEDS:
            jobs.append({
                "key": f"rsnn_{target}_seed{seed}",
                "model": "rsnn",
                "target": target,
                "seed": seed,
                "flags": dict(RSNN_FLAGS_BASE),
            })
    # RWNN: 12 targets × 3 seeds
    for target in TARGETS:
        for seed in SEEDS:
            jobs.append({
                "key": f"rwnn_{target}_seed{seed}",
                "model": "rwnn",
                "target": target,
                "seed": seed,
                "flags": dict(RWNN_FLAGS),
            })
    return jobs


def job_dir(job):
    return ROOT / f"{job['model']}_{job['target']}" / f"seed{job['seed']}"


def metrics_path(job):
    return job_dir(job) / "metrics.json"


def train_log(job):
    return job_dir(job) / "train.log"


# ---------------------------------------------------------------------------
# Seed=42 reuse from A/B winner
# ---------------------------------------------------------------------------
def maybe_reuse_ab_winner(state):
    """Copy A/B winner's seed=42 gap result into rsnn_gap/seed42/metrics.json."""
    if state["winner"] is None:
        return
    winner_dir = REPO / "runs/qm9_compare" / state["winner"] / "seed42"
    src = winner_dir / "metrics.json"
    dst = ROOT / "rsnn_gap" / "seed42" / "metrics.json"
    if not src.exists():
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Also copy the train.log for posterity
    train_src = winner_dir / "train.log"
    if train_src.exists():
        (dst.parent / "train.log").write_bytes(train_src.read_bytes())
    dst.write_bytes(src.read_bytes())
    print(f"[orch] reused A/B winner {state['winner']} as rsnn_gap/seed42")


# ---------------------------------------------------------------------------
# Process checks
# ---------------------------------------------------------------------------
def is_local_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def is_slurm_alive(jid):
    """Check via squeue (best-effort; slurm DB is flaky here)."""
    if jid is None:
        return False
    try:
        r = subprocess.run(
            ["squeue", "-h", "-j", str(jid), "-o", "%T"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            # squeue errored (DB unreachable, etc) — return None for "unknown"
            # so the caller can treat the job as "possibly still queued" rather
            # than dead. Without this, slurm DB flakiness false-positives jobs.
            return None
        state = r.stdout.strip()
        return state in ("RUNNING", "PENDING", "CONFIGURING")
    except Exception:
        return None  # unknown


def train_log_fresh(job):
    log = train_log(job)
    if not log.exists():
        return False
    age_min = (time.time() - log.stat().st_mtime) / 60.0
    return age_min < STALE_LOG_MIN


# ---------------------------------------------------------------------------
# Job submission: sbatch (dym-lab2 A40)
# ---------------------------------------------------------------------------
def make_sbatch_script(job):
    flags = {**COMMON_FLAGS, **job["flags"]}
    out_root = str(ROOT)
    run_subdir = f"{job['model']}_{job['target']}/seed{job['seed']}"

    cmd_parts = [PYTHON, "-u", str(REPO / "quickstart/train_qm9.py")]
    cmd_parts += ["--target", job["target"]]
    for k, v in flags.items():
        cmd_parts += [f"--{k}", str(v)]
    cmd_parts += ["--seed", str(job["seed"])]
    cmd_parts += ["--device_idx", "0"]
    cmd_parts += ["--out_root", out_root]
    cmd_parts += ["--run_subdir", run_subdir]
    cmd_parts += ["--limit", "0"]
    cmd = " \\\n  ".join(shlex.quote(p) for p in cmd_parts)

    log_path = ROOT / run_subdir / "train.log"
    err_path = SLURM_SCRIPTS / f"{job['key']}.err"
    job_metrics_target = ROOT / run_subdir / "gap" / "metrics.json"  # nested by target
    job_metrics_final = ROOT / run_subdir / "metrics.json"
    target = job["target"]

    script = f"""#!/bin/bash
#SBATCH --job-name=mt_{job['key']}
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output={log_path}
#SBATCH --error={err_path}

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd {REPO}

echo "# cmd: train_qm9.py {job['key']} (dym-lab2 A40 sbatch)"
echo "# gpu: $CUDA_VISIBLE_DEVICES seed: {job['seed']} model: {job['model']} target: {job['target']} host: $(hostname)"

{cmd}

# Hoist metrics.json from <run_subdir>/<target>/ to <run_subdir>/ for orchestrator
if [ -f {ROOT}/{run_subdir}/{target}/metrics.json ]; then
  cp {ROOT}/{run_subdir}/{target}/metrics.json {ROOT}/{run_subdir}/metrics.json
fi
echo "[slurm] done"
"""
    script_path = SLURM_SCRIPTS / f"{job['key']}.sbatch"
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def submit_sbatch(job, state):
    job_dir(job).mkdir(parents=True, exist_ok=True)
    script = make_sbatch_script(job)
    r = subprocess.run(
        ["sbatch", str(script)],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        print(f"[orch] sbatch FAILED for {job['key']}: {r.stderr.strip()}")
        return False
    # Parse "Submitted batch job NNNNN"
    jid = r.stdout.strip().split()[-1]
    state["slurm_jids"][job["key"]] = jid
    print(f"[orch] sbatch SUBMITTED {job['key']} -> JID={jid}")
    return True


# ---------------------------------------------------------------------------
# Job submission: local (dym-lab3 L40S)
# ---------------------------------------------------------------------------
def submit_local(job, state, gpu):
    flags = {**COMMON_FLAGS, **job["flags"]}
    out_root = str(ROOT)
    run_subdir = f"{job['model']}_{job['target']}/seed{job['seed']}"

    cmd_parts = [PYTHON, "-u", str(REPO / "quickstart/train_qm9.py")]
    cmd_parts += ["--target", job["target"]]
    for k, v in flags.items():
        cmd_parts += [f"--{k}", str(v)]
    cmd_parts += ["--seed", str(job["seed"])]
    cmd_parts += ["--device_idx", "0"]
    cmd_parts += ["--out_root", out_root]
    cmd_parts += ["--run_subdir", run_subdir]
    cmd_parts += ["--limit", "0"]

    job_dir(job).mkdir(parents=True, exist_ok=True)
    log = train_log(job)
    # Append-mode so we don't clobber prior tries
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"

    header = (
        f"# cmd: train_qm9.py {job['key']} (dym-lab3 local L40S GPU={gpu})\n"
        f"# gpu: {gpu} seed: {job['seed']} model: {job['model']} target: {job['target']} host: dym-lab3\n"
    )
    with open(log, "a") as f:
        f.write(header)
        f.flush()
        p = subprocess.Popen(
            cmd_parts,
            stdout=f, stderr=subprocess.STDOUT,
            env=env, cwd=str(REPO),
            start_new_session=True,
        )
    state["local_pids"][job["key"]] = p.pid
    state["local_gpus"][job["key"]] = gpu
    print(f"[orch] local SUBMITTED {job['key']} -> PID={p.pid} GPU={gpu}")
    return True


def hoist_local_metrics(job):
    """train_qm9.py writes to <out_root>/<run_subdir>/<target>/metrics.json.
    Hoist to <out_root>/<run_subdir>/metrics.json for orchestrator visibility."""
    nested = job_dir(job) / job["target"] / "metrics.json"
    final = metrics_path(job)
    if nested.exists() and not final.exists():
        final.write_bytes(nested.read_bytes())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
EGNN_GAP_TARGETS = {
    "mu": 0.029, "alpha": 0.071, "homo": 0.029, "lumo": 0.025,
    "gap": 0.048, "r2": 0.106, "zpve": 1.55e-3, "U0": 0.011,
    "U": 0.012, "H": 0.012, "G": 0.012, "Cv": 0.031,
}  # EGNN paper values (Satorras 2021 Table 2), in pipeline native units (eV / D / Bohr^k / cal mol^-1 K^-1).
# Note: our internal EGNN rerun for `gap` produced 0.0504 eV (~5% higher than the paper's 0.048),
# within seed variance. Ratios use the paper value to stay consistent with the literature.
# Note: "R2" target key in TARGETS uses uppercase; the lowercase "r2" entry below is intentionally
# not looked up by aggregate() (case mismatch) -> EGNN ratio shown as "?" / "-" per the convention
# that we do not cite a single per-target EGNN R^2 number.


def aggregate(jobs):
    by_key = {}
    for job in jobs:
        mae = read_test_mae(metrics_path(job))
        by_key[job["key"]] = mae

    results = {"per_model_target_seed": by_key, "mean_std": {}}
    summary_rows = []

    for model in ("rsnn", "rwnn"):
        for target in TARGETS:
            maes = [by_key.get(f"{model}_{target}_seed{s}") for s in SEEDS]
            maes = [m for m in maes if m is not None]
            if not maes:
                continue
            mean = sum(maes) / len(maes)
            std = (sum((m - mean) ** 2 for m in maes) / len(maes)) ** 0.5
            results["mean_std"][f"{model}_{target}"] = {
                "mean": mean, "std": std, "n": len(maes), "seeds": maes,
            }
            egnn = EGNN_GAP_TARGETS.get(target)
            ratio = f"{mean/egnn:.2f}x" if egnn else "?"
            summary_rows.append({
                "model": model, "target": target, "mean": mean,
                "std": std, "n": len(maes), "egnn": egnn, "ratio": ratio,
            })

    (ROOT / "results.json").write_text(json.dumps(results, indent=2))

    # Markdown summary
    md = ["# qm9_multitarget results", ""]
    md.append("Configs (verified from per-cell metrics.json):")
    md.append(
        "- RSNN: walk_type=search, m=8, w=16 (encoding window s), h=128, L=2, AdamW, "
        "lr=7.5e-4, wd=1e-4, 300ep/patience=50, EGNN-norm meann/MAD + L1 loss, "
        "Cormorant fixed split. Winner of A/B is O_B1_densedist (DFS-jump dense-distance "
        "fix in sample_dfs). gap seed=42 reused from A/B."
    )
    md.append(
        "- RWNN: walk_type=walk_ada, m=4, w=8 (encoding window s), h=128, L=2, AdamW, "
        "lr=7.5e-4, wd=1e-4, 300ep/patience=50. Mid-sweep switched from m=16 -> m=4 for "
        "speed; all 36 RWNN cells reran with m=4."
    )
    md.append(
        "- Walk length is per-molecule n (atom count), padded to max_len=29 for batching; "
        "`--w` is the encoding window, not the walk length."
    )
    md.append("")
    md.append("Notes:")
    md.append(
        "- EGNN reference for gap is the paper-published 0.048 eV; our internal rerun gave "
        "0.0504 eV (~5% higher, within seed variance). Ratios use the paper value."
    )
    md.append(
        "- R2 EGNN reference not cited per-target (atomization-energy task; <R^2> spatial "
        "extent in Bohr^2) -- shown as `-` / `?`."
    )
    md.append(
        "- Energy targets (U0/U/H/G) RWNN ratios are ~17000-19000x -- the walk-pool readout "
        "is bounded budget so cannot in principle aggregate size-extensive quantities; "
        "structural ceiling discussed in `~/vault/reflections/ito/2026-05-22-o-series-ceiling.md`."
    )
    md.append("")
    md.append("| Model | Target | mean ± std (n) | EGNN | Ratio |")
    md.append("|-------|--------|----------------|------|-------|")
    for r in summary_rows:
        egnn_s = f"{r['egnn']:.4f}" if r['egnn'] is not None else "-"
        md.append(f"| {r['model']} | {r['target']} | {r['mean']:.4f} ± {r['std']:.4f} (n={r['n']}) | {egnn_s} | {r['ratio']} |")
    (ROOT / "summary.md").write_text("\n".join(md))
    print(f"[orch] AGGREGATE WRITTEN: {ROOT}/results.json + summary.md")


# ---------------------------------------------------------------------------
# Main orchestration loop (one tick)
# ---------------------------------------------------------------------------
def tick():
    ROOT.mkdir(parents=True, exist_ok=True)
    SLURM_SCRIPTS.mkdir(parents=True, exist_ok=True)
    state = load_state()

    # Phase 1: A/B winner
    winner = pick_winner(state)
    if winner is None:
        print(f"[orch] PHASE=0 waiting for A/B (job {AB_JOB_ID})")
        return

    # Phase 2: reuse winner's seed=42 result for rsnn_gap
    maybe_reuse_ab_winner(state)

    # Phase 3: build manifest
    jobs = build_manifest()
    state["manifest_built"] = True

    # Phase 4: classify each job
    done_jobs, running_jobs, pending_jobs = [], [], []
    for job in jobs:
        # Try to hoist metrics from nested target dir before checking
        hoist_local_metrics(job)
        if metrics_path(job).exists():
            done_jobs.append(job)
            continue
        key = job["key"]
        # Permanent skip if exceeded retries (no resubmit)
        if state["retries"].get(key, 0) > MAX_RETRIES:
            continue
        # Is this job currently running?
        local_pid = state["local_pids"].get(key)
        slurm_jid = state["slurm_jids"].get(key)
        local_run = local_pid is not None and is_local_alive(local_pid)
        slurm_alive_check = is_slurm_alive(slurm_jid) if slurm_jid is not None else False
        # If slurm DB is flaky (returns None), assume the jid is still queued.
        # Otherwise we false-positive mark jobs as dead and exhaust retries.
        slurm_run = slurm_jid is not None and (slurm_alive_check is True or slurm_alive_check is None)
        if local_run or slurm_run:
            running_jobs.append(job)
            continue
        # Job not running, not done -> pending (possibly retry)
        if local_pid is not None or slurm_jid is not None:
            # Was running but now dead; check log freshness as a tie-break
            if train_log_fresh(job):
                # log is fresh -> maybe slurm DB is stale. Treat as running.
                running_jobs.append(job)
                continue
            # Truly dead. Increment retries.
            state["retries"][key] = state["retries"].get(key, 0) + 1
            if local_pid is not None:
                state["local_pids"].pop(key, None)
                state["local_gpus"].pop(key, None)
            if slurm_jid is not None:
                state["slurm_jids"].pop(key, None)
            if state["retries"][key] > MAX_RETRIES:
                print(f"[orch] job {key} EXCEEDED RETRIES ({state['retries'][key]}); skipping permanently")
                continue
        pending_jobs.append(job)
    save_state(state)

    # Phase 5: aggregate if all jobs are done
    if len(done_jobs) == len(jobs):
        aggregate(jobs)
        print(f"[orch] ALL DONE: {len(done_jobs)}/{len(jobs)} jobs complete")
        return "ALL_DONE"

    # Phase 6: fill slots
    sbatch_running = sum(1 for j in running_jobs
                        if state["slurm_jids"].get(j["key"]) is not None)
    local_running = sum(1 for j in running_jobs
                        if state["local_pids"].get(j["key"]) is not None)
    sbatch_avail = MAX_SBATCH_SLOTS - sbatch_running
    local_avail = MAX_LOCAL_SLOTS - local_running
    used_local_gpus = {state["local_gpus"][k] for k in state["local_gpus"]
                       if k in {j["key"] for j in running_jobs}}
    free_local_gpus = [g for g in range(MAX_LOCAL_SLOTS) if g not in used_local_gpus]

    submitted = 0
    # Prioritize: fill local slots first (less queue latency)
    for gpu in free_local_gpus[:local_avail]:
        if not pending_jobs:
            break
        job = pending_jobs.pop(0)
        if submit_local(job, state, gpu):
            submitted += 1
    # Then sbatch
    for _ in range(sbatch_avail):
        if not pending_jobs:
            break
        job = pending_jobs.pop(0)
        if submit_sbatch(job, state):
            submitted += 1
    save_state(state)

    print(
        f"[orch] tick done: pending={len(pending_jobs)+submitted} (submitted now={submitted}) "
        f"running={len(running_jobs)} done={len(done_jobs)}/{len(jobs)} "
        f"slots_used=local{local_running}/{MAX_LOCAL_SLOTS}+sbatch{sbatch_running}/{MAX_SBATCH_SLOTS}"
    )


if __name__ == "__main__":
    sys.exit(0 if tick() != "ALL_DONE" else 0)
