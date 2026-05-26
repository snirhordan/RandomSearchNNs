#!/bin/bash
#SBATCH --job-name=mt_rwnn_mu_seed43
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rwnn_mu/seed43/train.log
#SBATCH --error=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/slurm_scripts/rwnn_mu_seed43.err

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs

echo "# cmd: train_qm9.py rwnn_mu_seed43 (dym-lab2 A40 sbatch)"
echo "# gpu: $CUDA_VISIBLE_DEVICES seed: 43 model: rwnn target: mu host: $(hostname)"

/home/snirhordan/miniconda3/envs/rwnn/bin/python3 \
  -u \
  /home/snirhordan/ito/RandomSearchNNs/quickstart/train_qm9.py \
  --target \
  mu \
  --split \
  cormorant \
  --cormorant_data_dir \
  /home/snirhordan/ito/RandomSearchNNs/external/egnn/qm9/temp/qm9 \
  --lr_scheduler \
  cosine \
  --use_egnn_normalization \
  1 \
  --norm_constants_json \
  /home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/preprocessing_audit.json \
  --epochs \
  300 \
  --early_stopping \
  50 \
  --n_splits \
  1 \
  --num_workers \
  12 \
  --walk_type \
  walk_ada \
  --distances \
  1 \
  --mol_edge_feat \
  1 \
  --batch_size \
  96 \
  --h_dim \
  128 \
  --num_layers \
  2 \
  --m \
  16 \
  --w \
  8 \
  --reduce \
  sum \
  --lr \
  0.00075 \
  --grad_clip \
  1.0 \
  --weight_decay \
  0.0001 \
  --optimizer \
  adamw \
  --lstm_init \
  default \
  --dropout \
  0.0 \
  --warmup_epochs \
  0 \
  --seed \
  43 \
  --device_idx \
  0 \
  --out_root \
  /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget \
  --run_subdir \
  rwnn_mu/seed43 \
  --limit \
  0

# Hoist metrics.json from <run_subdir>/<target>/ to <run_subdir>/ for orchestrator
if [ -f /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rwnn_mu/seed43/mu/metrics.json ]; then
  cp /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rwnn_mu/seed43/mu/metrics.json /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rwnn_mu/seed43/metrics.json
fi
echo "[slurm] done"
