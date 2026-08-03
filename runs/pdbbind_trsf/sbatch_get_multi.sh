#!/bin/bash
# GET native baselines for PDBbind identity60 + scaffold splits (3 seeds each, 20 epochs).
#SBATCH --job-name=get_multi
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-getmulti-%j.out

set -uo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate GET
cd /home/snirhordan/ito/RandomSearchNNs/GET
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
for cfg in identity60 scaffold; do
  echo "=== GET baseline: $cfg ==="
  python scripts/exps/exps_3.py --config ./scripts/exps/configs/PDBBind/${cfg}_get.json --gpus 0
done
