#!/usr/bin/env python3
"""Move completed metrics.json from _stage/ to final out_root paths.

Background: pre-fix dispatchers used ``running: dict[int, Popen]`` keyed by
GPU index, which overwrote earlier procs in packed configs (--gpus 2,2,2).
The 4-of-6 untracked procs finished training but the dispatcher never
called ``finalize()``, leaving their metrics in _stage/. This script
sweeps that up safely.

Safe to run while the dispatcher is alive. It only moves stage metrics
when the final dst doesn't exist (preserves the dispatcher's own writes).
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]


def metrics_ok(p: Path) -> bool:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    # RWNN: summary.mean_test_mae
    v = d.get("summary", {}).get("mean_test_mae")
    if v is not None and v == v:
        return True
    # EGNN: splits[0].test_mae
    sp = d.get("splits", [{}])
    v = sp[0].get("test_mae") if sp else None
    return v is not None and v == v


def salvage(stage_root: Path, final_root_for: callable):
    moved = 0
    for stage_metrics in stage_root.rglob("metrics.json"):
        # Validate the staged file is actually a completed training result
        if not metrics_ok(stage_metrics):
            continue
        final = final_root_for(stage_metrics)
        if final is None:
            continue
        if final.exists() and metrics_ok(final):
            # Already finalized — leave stage alone (dispatcher may clean up
            # later, or we can delete stage manually).
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        # Atomic move within NFS-mounted filesystem
        stage_metrics.replace(final)
        print(f"salvaged: {final}")
        moved += 1
    return moved


def rwnn_final_for(stage_path: Path) -> Path | None:
    # stage layout: runs/qm9_rwnn/_stage/m<M>/<target>/split<S>/<target>/metrics.json
    # final layout: runs/qm9_rwnn/m<M>/<target>/split<S>/metrics.json
    parts = stage_path.parts
    try:
        i = parts.index("_stage")
    except ValueError:
        return None
    after = parts[i + 1 :]
    # after = ('m16', 'gap', 'split0', 'gap', 'metrics.json')
    if len(after) < 5:
        return None
    m, tgt, split, tgt2, _ = after[:5]
    if tgt != tgt2:
        return None
    return REPO / "runs" / "qm9_rwnn" / m / tgt / split / "metrics.json"


def egnn_final_for(stage_path: Path) -> Path | None:
    # EGNN's run_one.py writes directly to final out_dir already, but if it
    # ever stages (future use), follow the same idiom:
    # stage: runs/qm9_egnn/_stage/<target>/seed<S>/metrics.json -> final:
    # runs/qm9_egnn/<target>/seed<S>/metrics.json
    parts = stage_path.parts
    try:
        i = parts.index("_stage")
    except ValueError:
        return None
    after = parts[i + 1 :]
    if len(after) < 3:
        return None
    return REPO / "runs" / "qm9_egnn" / Path(*after)


def main():
    n = 0
    rwnn_stage = REPO / "runs" / "qm9_rwnn" / "_stage"
    if rwnn_stage.exists():
        n += salvage(rwnn_stage, rwnn_final_for)
    egnn_stage = REPO / "runs" / "qm9_egnn" / "_stage"
    if egnn_stage.exists():
        n += salvage(egnn_stage, egnn_final_for)
    print(f"total salvaged: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
