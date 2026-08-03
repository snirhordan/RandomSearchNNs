#!/bin/bash
# Complete the geom_bias grid to 3 seeds per cutoff (idempotent; the val-based
# cutoff selection at n=1 seed was too noisy). Runs the 4 missing cells:
# cut5/seed{43,44} and cut10/seed{43,44}; cut15 x3 and the seed42 sweep already exist.
#SBATCH --job-name=geomfull
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-geomfull-%j.out

set -euo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python3 -u runs/pdbbind_trsf/dispatch_geom_bias.py full
