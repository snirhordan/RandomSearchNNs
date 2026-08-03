#!/bin/bash
# 3-seed confirmation: sinusoidal + dropout0.1 (seeds 43,44) on dym-lab2.
#SBATCH --job-name=pdbbind_confirm
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --nodelist=dym-lab2
#SBATCH --gres=gpu:A40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-confirm-%j.out

set -euo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python3 -u runs/pdbbind_trsf/dispatch_confirm.py
