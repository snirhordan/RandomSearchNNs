#!/usr/bin/env bash
# Chain target phases sequentially. Designed to be invoked AFTER an initial
# target's run_target.sh has been launched.
#
# Usage:  bash chain_phases.sh <target1> [<target2> ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

for TGT in "$@"; do
    echo "==================== TARGET: $TGT ===================="
    bash runs/qm9/run_target.sh "$TGT"
    echo "==================== TARGET $TGT DONE ===================="
done

echo "Aggregating..."
python3 runs/qm9/aggregate_results.py
echo "ALL CHAIN DONE"
