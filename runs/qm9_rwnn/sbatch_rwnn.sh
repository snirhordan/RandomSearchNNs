#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --nodelist=newton3,newton4,galileo1
#SBATCH --mail-user=snirhordan@cs.technion.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="rwnn-qm9"
#SBATCH -o /home/snirhordan/ito/RandomSearchNNs/runs/qm9_rwnn/slurm_out_%j.txt
#SBATCH -e /home/snirhordan/ito/RandomSearchNNs/runs/qm9_rwnn/slurm_err_%j.txt
# Runs one d-RWNN m-sweep dispatcher with 3 procs/GPU on the allocated A40.
# Idempotent NFS-shared metrics tree; multiple jobs pick disjoint
# (m, target, split) triples automatically.

set -euo pipefail
module purge
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

cd /home/snirhordan/ito/RandomSearchNNs

echo "host=$(hostname) cuda_visible=$CUDA_VISIBLE_DEVICES"

python3 -u runs/qm9_rwnn/dispatch.py \
    --epochs 15 \
    --patience 4 \
    --gpus 0,0,0
