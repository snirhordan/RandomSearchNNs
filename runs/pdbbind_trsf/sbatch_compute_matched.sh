#!/bin/bash
# Compute-matched RSNN_TRSF: dropout0.1 config capped to GET's wall-clock budget.
# 4 datasets x 3 seeds = 12 short runs (2-6 epochs each), slot-scheduled over 4 A40s.
#SBATCH --job-name=cm_rsnn
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-cm-%j.out

set -uo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python3 -u runs/pdbbind_trsf/dispatch_compute_matched.py
