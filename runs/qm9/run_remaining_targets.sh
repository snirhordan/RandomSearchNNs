#!/usr/bin/env bash
# Run the 9 QM9 targets not yet covered (U0/gap/mu already done).
# Each phase = 4 configs in parallel on cuda:0..cuda:3 for one target.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

TARGETS=(alpha homo lumo R2 zpve U H G Cv)

for TGT in "${TARGETS[@]}"; do
    echo "==================== TARGET: $TGT ===================="
    bash runs/qm9/run_target.sh "$TGT"
    echo "==================== TARGET $TGT DONE ===================="
done

echo "Aggregating..."
python3 runs/qm9/aggregate_results.py
echo "ALL REMAINING TARGETS DONE"
