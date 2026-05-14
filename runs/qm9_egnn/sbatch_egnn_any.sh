#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton1,nlp-2080-1,nlp-2080-2,ran-mashawsha
#SBATCH --mail-user=snirhordan@cs.technion.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --job-name="egnn-any"
#SBATCH -o /home/snirhordan/ito/RandomSearchNNs/runs/qm9_egnn/slurm_out_%j.txt
#SBATCH -e /home/snirhordan/ito/RandomSearchNNs/runs/qm9_egnn/slurm_err_%j.txt
# Any free GPU in the cluster (excluding 1080ti/2080ti/titanxp nodes which
# can't fit 2 packed procs). Packs 2 procs/GPU to be safe across 24-48 GB cards.

set -euo pipefail
module purge
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

cd /home/snirhordan/ito/RandomSearchNNs

echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 -u runs/qm9_egnn/dispatch.py \
    --epochs 300 \
    --gpus 0,0
