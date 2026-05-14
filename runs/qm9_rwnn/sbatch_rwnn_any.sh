#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton1,nlp-2080-1,nlp-2080-2,ran-mashawsha
#SBATCH --mail-user=snirhordan@cs.technion.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --job-name="rwnn-any"
#SBATCH -o /home/snirhordan/ito/RandomSearchNNs/runs/qm9_rwnn/slurm_out_%j.txt
#SBATCH -e /home/snirhordan/ito/RandomSearchNNs/runs/qm9_rwnn/slurm_err_%j.txt
# Any free GPU in the cluster. RWNN procs are ~2.6GB each so we can pack 3.

set -euo pipefail
module purge
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

cd /home/snirhordan/ito/RandomSearchNNs

echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 -u runs/qm9_rwnn/dispatch.py \
    --epochs 15 \
    --patience 4 \
    --gpus 0,0,0
