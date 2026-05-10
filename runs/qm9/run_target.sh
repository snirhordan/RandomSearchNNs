#!/usr/bin/env bash
# Launch the four (distances, mol_edge_feat) configs of a single target in
# parallel on cuda:0..cuda:3.
#
# Usage:  bash run_target.sh <target>           e.g. bash run_target.sh U0
#
# Each child writes train.log + metrics.json + best ckpt under
#   runs/qm9/d<d>_m<m>/<target>/
#
set -euo pipefail

TARGET="${1:-U0}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-4}"
BATCH="${BATCH:-128}"
H="${H:-128}"
NL="${NL:-2}"
M="${M:-8}"
W="${W:-8}"
SEED="${SEED:-42}"
RBF_K="${RBF_K:-16}"
RBF_CUT="${RBF_CUT:-5.0}"
NW="${NW:-6}"
LIMIT="${LIMIT:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

# Build the preprocessing cache once so children skip ~5min preproc.
CACHE_DIR="./data/qm9/qm9_d_rwnn_cache"
LIMIT_TAG=""
if [[ "${LIMIT}" != "0" ]]; then LIMIT_TAG="_lim${LIMIT}"; fi
CACHE_PATH="${CACHE_DIR}/mols_${TARGET}${LIMIT_TAG}.pt"
if [[ ! -f "$CACHE_PATH" ]]; then
    echo "Building cache for target=$TARGET..."
    python3 runs/qm9/build_cache.py --target "$TARGET" \
        --data_root ./data/qm9 --limit "$LIMIT" \
        > "runs/qm9/build_cache_${TARGET}.log" 2>&1
    echo "Cache built."
else
    echo "Cache exists at $CACHE_PATH"
fi

CFGS=("0 0" "1 0" "0 1" "1 1")
PIDS=()

for IDX in 0 1 2 3; do
    set -- ${CFGS[$IDX]}
    D=$1
    MEF=$2
    OUTDIR="runs/qm9/d${D}_m${MEF}/${TARGET}"
    mkdir -p "$OUTDIR"
    LOG="$OUTDIR/train.log"
    echo "Launching cfg=(d=$D, m=$MEF) -> CUDA $IDX -> $LOG"
    CUDA_VISIBLE_DEVICES="$IDX" \
        nohup python3 -u quickstart/train_qm9.py \
            --target "$TARGET" \
            --distances "$D" \
            --mol_edge_feat "$MEF" \
            --epochs "$EPOCHS" \
            --early_stopping "$PATIENCE" \
            --n_splits 1 \
            --batch_size "$BATCH" \
            --h_dim "$H" \
            --num_layers "$NL" \
            --m "$M" \
            --w "$W" \
            --seed "$SEED" \
            --rbf_K "$RBF_K" \
            --rbf_cutoff "$RBF_CUT" \
            --num_workers "$NW" \
            --device_idx 0 \
            --limit "$LIMIT" \
            --out_root ./runs/qm9 \
            > "$LOG" 2>&1 &
    PIDS+=("$!")
done

echo "Started ${#PIDS[@]} jobs: ${PIDS[*]}"
echo "Waiting on PIDs..."
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "PID $PID FAILED"
        FAIL=1
    fi
done
if [[ "$FAIL" -ne 0 ]]; then
    echo "One or more configs FAILED for target=$TARGET"
    exit 1
fi
echo "Target $TARGET: ALL 4 CONFIGS DONE"
