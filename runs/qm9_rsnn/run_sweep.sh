#!/usr/bin/env bash
# Launch the full RSNN m-sweep on QM9.
#
# Usage:
#   bash runs/qm9_rsnn/run_sweep.sh                # full grid, 10 ep / pat 3
#   EPOCHS=15 PATIENCE=4 bash runs/qm9_rsnn/run_sweep.sh
#   TARGETS=mu,gap MS=4,8 bash runs/qm9_rsnn/run_sweep.sh
#
# Outputs land under runs/qm9_rsnn/m{1,4,8,16}/<target>/split{0,1,2}/.
# The Python dispatcher handles GPU pinning (4-GPU round-robin), staging,
# and idempotent re-runs (jobs with a valid metrics.json are skipped).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
NW="${NW:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TARGETS="${TARGETS:-}"
MS="${MS:-}"

CMD=(python3 -u runs/qm9_rsnn/dispatch.py
     --max_parallel "$MAX_PARALLEL"
     --epochs "$EPOCHS" --patience "$PATIENCE"
     --num_workers "$NW")
if [[ -n "$TARGETS" ]]; then CMD+=("--targets" "$TARGETS"); fi
if [[ -n "$MS" ]]; then CMD+=("--ms" "$MS"); fi

echo "[run_sweep] $(date)  cmd: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a runs/qm9_rsnn/dispatch.log
