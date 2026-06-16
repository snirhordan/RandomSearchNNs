#!/bin/bash
#SBATCH --job-name=mt_rsnn_lumo_seed42
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rsnn_lumo/seed42/train.log
#SBATCH --error=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/slurm_scripts/rsnn_lumo_seed42.err

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs

echo "# cmd: train_qm9.py rsnn_lumo_seed42 (dym-lab2 A40 sbatch)"
echo "# gpu: $CUDA_VISIBLE_DEVICES seed: 42 model: rsnn target: lumo host: $(hostname)"

/home/snirhordan/miniconda3/envs/rwnn/bin/python3 \
  -u \
  /home/snirhordan/ito/RandomSearchNNs/quickstart/train_qm9.py \
  --target \
  lumo \
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
  search \
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
  8 \
  --w \
  16 \
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
  42 \
  --device_idx \
  0 \
  --out_root \
  /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget \
  --run_subdir \
  rsnn_lumo/seed42 \
  --limit \
  0

# Hoist metrics.json from <run_subdir>/<target>/ to <run_subdir>/ for orchestrator
if [ -f /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rsnn_lumo/seed42/lumo/metrics.json ]; then
  cp /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rsnn_lumo/seed42/lumo/metrics.json /home/snirhordan/ito/RandomSearchNNs/runs/qm9_multitarget/rsnn_lumo/seed42/metrics.json
fi
echo "[slurm] done"
