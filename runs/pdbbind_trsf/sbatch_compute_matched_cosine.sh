#!/bin/bash
# Compute-matched RSNN_TRSF with cosine LR annealing over the capped budget.
# Same epoch caps as the constant-lr sweep (6/5/5/2), so compute stays matched to GET;
# only the lr schedule differs. Tests whether the identity30 instability under the cap
# (seeds 0.525-0.606) is caused by stopping mid-flight at full lr.
#SBATCH --job-name=cm_cos
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-cmcos-%j.out

set -uo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
export CM_SCHEDULE=cosine
python3 -u runs/pdbbind_trsf/dispatch_compute_matched.py
