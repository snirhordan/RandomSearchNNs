#!/bin/bash
# GET native baseline on Atom3D LBA (LBA/get.json: max_epoch 10, 3 seeds), reads the LMDB directly.
#SBATCH --job-name=get_lba
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/pdbbind_trsf/slurm-getlba-%j.out

set -uo pipefail
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate GET
cd /home/snirhordan/ito/RandomSearchNNs/GET
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python scripts/exps/exps_3.py --config ./scripts/exps/configs/LBA/get.json --gpus 0
