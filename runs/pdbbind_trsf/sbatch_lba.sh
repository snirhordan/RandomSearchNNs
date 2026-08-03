#!/bin/bash
# RSNN head-to-head on Atom3D LBA only (dropout0.1 x3 + causal + lr5e4); disjoint from the
# id60/scaffold dispatcher so it can run concurrently.
#SBATCH --job-name=lba_rsnn
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-lba-%j.out

set -euo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python3 -u runs/pdbbind_trsf/dispatch_multidataset.py lba
