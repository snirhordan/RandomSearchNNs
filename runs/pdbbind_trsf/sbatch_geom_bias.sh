#!/bin/bash
# Geometric-attention-bias pilot on PDBbind-Benchmark identity30 (RSNN_TRSF
# dropout0.1 + geom_bias). Phase A cutoff sweep {5,10,15} @ seed42, then best
# cutoff (by val Pearson) x seeds {42,43,44}. GET baseline is unchanged
# (identity30 GET 0.5887 already computed) so only our model runs here.
#SBATCH --job-name=geom_rsnn
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-geom-%j.out

set -euo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python3 -u runs/pdbbind_trsf/dispatch_geom_bias.py
