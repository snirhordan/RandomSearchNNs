#!/bin/bash
# Phase 2 ablation: gap target × 3 seeds × 3 feature configs = 9 runs
# Distributed across 4 local L40S GPUs with a slot scheduler.
# After it finishes: per-cell metrics.json in runs/qm9_quadruplet_ablation/<cfg>/seed<S>/.
# Pre-commit smoke verified pe_in_dim=51 and monotonic loss decrease.

set -euo pipefail

REPO=/home/snirhordan/ito/RandomSearchNNs
PYTHON=/home/snirhordan/miniconda3/envs/rwnn/bin/python3
OUT_ROOT="$REPO/runs/qm9_quadruplet_ablation"
LOG_DIR="$OUT_ROOT/scripts"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

# ---- Common args (match O_B1_densedist + smoke baseline) -------------------
COMMON_ARGS=(
  --target gap --split cormorant
  --cormorant_data_dir "$REPO/external/egnn/qm9/temp/qm9"
  --lr_scheduler cosine --use_egnn_normalization 1
  --norm_constants_json "$REPO/runs/qm9_compare/preprocessing_audit.json"
  --walk_type search --distances 1 --mol_edge_feat 1
  --max_search_len 16
  --epochs 300 --early_stopping 50 --n_splits 1
  --batch_size 96 --h_dim 128 --num_layers 2 --m 8 --w 16 --reduce sum --lr 0.00075
  --num_workers 12
  --grad_clip 1.0 --weight_decay 0.0001 --optimizer adamw
  --lstm_init default --dropout 0.0 --warmup_epochs 0
  --out_root "$OUT_ROOT" --limit 0
)

# ---- Ablation cells (9 total) ----------------------------------------------
declare -a CELLS
# (config_name, angles, dihedrals)
CONFIGS=(
  "dist_only:0:0"
  "plus_angle:1:0"
  "plus_angle_dihedral:1:1"
)
SEEDS=(42 43 44)

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r cfg_name angles dihedrals <<< "$cfg"
  for seed in "${SEEDS[@]}"; do
    CELLS+=("${cfg_name}:${seed}:${angles}:${dihedrals}")
  done
done

echo "[ablation] ${#CELLS[@]} cells to run on 4 GPUs"

# ---- Slot scheduler: 4 GPUs, fill each slot as it frees ---------------------
declare -A SLOT_PID
declare -A SLOT_CELL
NUM_GPUS=4

launch_cell() {
  local gpu=$1; local cell=$2
  IFS=":" read -r cfg_name seed angles dihedrals <<< "$cell"
  local subdir="${cfg_name}/seed${seed}"
  local logfile="$OUT_ROOT/${subdir}/train.log"
  mkdir -p "$(dirname "$logfile")"
  echo "[ablation] launch cell=$cell gpu=$gpu -> $subdir"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" -u "$REPO/quickstart/train_qm9.py" \
    "${COMMON_ARGS[@]}" \
    --angles "$angles" --dihedrals "$dihedrals" --angle_K 8 --dihedral_K 4 \
    --seed "$seed" --device_idx 0 \
    --run_subdir "$subdir" \
    > "$logfile" 2>&1 &
  SLOT_PID[$gpu]=$!
  SLOT_CELL[$gpu]="$cell"
}

# Initial fill: launch first 4 cells on 4 GPUs
PENDING=("${CELLS[@]}")
for ((g=0; g<NUM_GPUS && ${#PENDING[@]}>0; g++)); do
  launch_cell "$g" "${PENDING[0]}"
  PENDING=("${PENDING[@]:1}")
done

# Refill: when a slot's PID exits, launch the next pending cell on that slot
COMPLETED=()
while [[ ${#COMPLETED[@]} -lt ${#CELLS[@]} ]]; do
  for ((g=0; g<NUM_GPUS; g++)); do
    pid=${SLOT_PID[$g]:-}
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      cell=${SLOT_CELL[$g]}
      wait "$pid" 2>/dev/null && rc=0 || rc=$?
      echo "[ablation] cell=$cell gpu=$g done rc=$rc (${#COMPLETED[@]}/${#CELLS[@]} previously complete)"
      COMPLETED+=("$cell")
      unset SLOT_PID[$g]
      unset SLOT_CELL[$g]
      if [[ ${#PENDING[@]} -gt 0 ]]; then
        launch_cell "$g" "${PENDING[0]}"
        PENDING=("${PENDING[@]:1}")
      fi
    fi
  done
  sleep 30
done

echo "[ablation] ALL DONE. ${#COMPLETED[@]}/${#CELLS[@]} cells complete."
