#!/usr/bin/env bash
# Re-scan the m-sweep grid and re-launch any jobs whose metrics.json is
# missing (e.g. crashed jobs, or jobs added to the matrix after the main
# dispatcher started).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
NW="${NW:-4}"
echo "[followup] $(date)"
python3 -u runs/qm9_rsnn/dispatch.py --max_parallel 4 \
    --epochs "$EPOCHS" --patience "$PATIENCE" --num_workers "$NW" \
    2>&1 | tee -a runs/qm9_rsnn/dispatch_followup.log
