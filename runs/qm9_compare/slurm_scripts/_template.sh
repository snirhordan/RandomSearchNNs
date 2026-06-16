#!/bin/bash
#SBATCH --job-name=__JOBNAME__
#SBATCH --partition=dym
#SBATCH --nodelist=dym-lab2
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/slurm_scripts/__JOBNAME__.out
#SBATCH --error=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/slurm_scripts/__JOBNAME__.err

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

cd /home/snirhordan/ito/RandomSearchNNs

echo "[slurm] node=$(hostname) gpu_visible=$CUDA_VISIBLE_DEVICES start=$(date)"

__CMD__

echo "[slurm] end=$(date)"
