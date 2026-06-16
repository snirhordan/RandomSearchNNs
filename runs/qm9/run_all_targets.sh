#!/usr/bin/env bash
# Sequentially run all 3 targets (each phase = 4 configs in parallel on GPUs).
# Aggregate results into runs/qm9/results.md at the end.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

for TGT in U0 gap mu; do
    echo "==================== TARGET: $TGT ===================="
    bash runs/qm9/run_target.sh "$TGT"
    echo "==================== TARGET $TGT DONE ===================="
done

echo "Aggregating..."
python3 runs/qm9/aggregate_results.py

echo "ALL DONE"
