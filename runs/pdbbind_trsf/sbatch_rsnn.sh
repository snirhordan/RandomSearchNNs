#!/bin/bash
# RSNN_TRSF (transformer + dihedral) sweep on PDBbind-Benchmark identity30, dym-lab.
# Requests 6 A40s; the dispatcher adapts to however many GPUs it actually gets, so
# lowering --gres to gpu:A40:N (N<6) is safe if 6 are unavailable.
#SBATCH --job-name=pdbbind_rsnn
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --nodelist=dym-lab2
#SBATCH --gres=gpu:A40:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-%j.out

set -euo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python3 -u runs/pdbbind_trsf/dispatch.py
