#!/bin/bash
# GET native baseline on PDBbind-Benchmark identity30 ("run what's in the code":
# scripts/exps/configs/PDBBind/identity30_get.json -> max_epoch 20, 3 seeds).
#SBATCH --job-name=pdbbind_get
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --nodelist=dym-lab
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-get-%j.out

set -uo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate GET
cd /home/snirhordan/ito/RandomSearchNNs/GET
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python scripts/exps/exps_3.py \
  --config ./scripts/exps/configs/PDBBind/identity30_get.json \
  --gpus 0
