#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --nodelist=newton3,newton4,galileo1
#SBATCH --mail-user=snirhordan@cs.technion.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="egnn-qm9"
#SBATCH -o /home/snirhordan/ito/RandomSearchNNs/runs/qm9_egnn/slurm_out_%j.txt
#SBATCH -e /home/snirhordan/ito/RandomSearchNNs/runs/qm9_egnn/slurm_err_%j.txt
# Runs one EGNN dispatcher with 3 procs/GPU on the allocated A40.
# The dispatcher is idempotent across the NFS-shared metrics tree, so
# multiple of these jobs running in parallel pick up disjoint (target, seed)
# pairs automatically.

set -euo pipefail
module purge
source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

cd /home/snirhordan/ito/RandomSearchNNs

# CUDA_VISIBLE_DEVICES is set by Slurm to a single device index — the
# dispatcher's --gpus 0,0,0 then packs 3 procs onto that device. Inside the
# dispatcher's launch(), we set env CUDA_VISIBLE_DEVICES=str(gpu) which
# overrides Slurm's. We don't want that, so unset before launching.
echo "host=$(hostname) cuda_visible=$CUDA_VISIBLE_DEVICES"

# Each EGNN run is ~21h wall (at 1 proc/GPU). With 3-way packing it's
# ~63h but 3 runs complete in that time. The Slurm job runs until its own
# walltime limit; the dispatcher keeps grabbing new (target, seed) pairs
# until the metrics tree is full or the job is preempted.
python3 -u runs/qm9_egnn/dispatch.py \
    --epochs 300 \
    --gpus 0,0,0
