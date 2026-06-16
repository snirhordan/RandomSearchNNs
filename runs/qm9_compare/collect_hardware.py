#!/usr/bin/env python3
"""Auto-collect hardware + software fingerprint for the gap comparison.

Writes runs/qm9_compare/hardware_fingerprint.json once at dispatcher start.
Captures everything needed to reproduce the run env: host, GPUs, torch
version, Slurm allocation, env vars, git commit. Idempotent — overwrites
the existing file each call.
"""
from __future__ import annotations
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def cpu_info() -> dict:
    out = {"logical_cores": os.cpu_count() or 0}
    try:
        info = Path("/proc/cpuinfo").read_text()
        for line in info.splitlines():
            if line.startswith("model name"):
                out["model"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        out["load_avg_at_start"] = list(os.getloadavg())
    except Exception:
        pass
    return out


def ram_gb_total() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return round(kib / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def gpu_info() -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        return []
    raw = _run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    out = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            out.append({
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": int(parts[3]),
                "driver_version": parts[4],
            })
    return out


def slurm_env() -> dict:
    keys = [
        "SLURM_JOB_ID", "SLURM_STEP_GPUS", "SLURM_JOB_USER",
        "SLURM_TASKS_PER_NODE", "SLURM_CPU_BIND",
    ]
    return {k: os.environ.get(k, "") for k in keys}


def env_caps() -> dict:
    keys = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES",
    ]
    return {k: os.environ.get(k, "") for k in keys}


def torch_info() -> dict:
    out = {}
    try:
        import torch
        out["version"] = torch.__version__
        out["cuda_runtime"] = torch.version.cuda
        out["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available()
            else None)
        out["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        out["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
        out["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        try:
            out["deterministic_algorithms"] = bool(
                torch.are_deterministic_algorithms_enabled())
        except Exception:
            out["deterministic_algorithms"] = None
    except Exception as e:
        out["import_error"] = str(e)
    return out


def git_info(repo: Path) -> dict:
    if not (repo / ".git").exists():
        return {}
    return {
        "branch": _run(["git", "-C", str(repo), "rev-parse",
                        "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "-C", str(repo), "status", "--porcelain"])),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    out = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": cpu_info(),
        "ram_gb_total": ram_gb_total(),
        "gpus": gpu_info(),
        "slurm": slurm_env(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "torch": torch_info(),
        "env": env_caps(),
        "git": git_info(repo),
    }
    out_path = repo / "runs" / "qm9_compare" / "hardware_fingerprint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
