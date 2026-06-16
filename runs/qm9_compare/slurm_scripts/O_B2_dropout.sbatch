#!/bin/bash
#SBATCH --job-name=rsnn_O_B2_dropout
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --nodelist=dym-lab2
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/O_B2_dropout/seed42/train.log
#SBATCH --error=/home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/slurm_scripts/O_B2_dropout.err

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn
cd /home/snirhordan/ito/RandomSearchNNs

echo "# cmd: train_qm9.py O_B2_dropout (1-pack restart on dym-lab2 A40)"
echo "# gpu: $CUDA_VISIBLE_DEVICES seed: 42 model: O_B2_dropout host: $(hostname)"

/home/snirhordan/miniconda3/envs/rwnn/bin/python3 -u /home/snirhordan/ito/RandomSearchNNs/quickstart/train_qm9.py \
  --target gap --split cormorant \
  --cormorant_data_dir /home/snirhordan/ito/RandomSearchNNs/external/egnn/qm9/temp/qm9 \
  --lr_scheduler cosine --use_egnn_normalization 1 \
  --norm_constants_json /home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/preprocessing_audit.json \
  --walk_type search --distances 1 --mol_edge_feat 1 \
  --epochs 300 --early_stopping 50 --n_splits 1 \
  --batch_size 96 --h_dim 128 --num_layers 2 --m 8 --w 16 --reduce sum --lr 0.00075 \
  --seed 42 --num_workers 12 \
  --grad_clip 1.0 --weight_decay 0.0 --optimizer adam --lstm_init default --dropout 0.15 --warmup_epochs 0 \
  --device_idx 0 \
  --out_root /home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare \
  --run_subdir O_B2_dropout/seed42 --limit 0

if [ -f /home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/O_B2_dropout/seed42/gap/metrics.json ]; then
  cp /home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/O_B2_dropout/seed42/gap/metrics.json \
     /home/snirhordan/ito/RandomSearchNNs/runs/qm9_compare/O_B2_dropout/seed42/metrics.json
fi
echo "[slurm] done"
