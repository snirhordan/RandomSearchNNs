#!/usr/bin/env bash
# Resume the QM9 RSNN m-sweep dispatcher.
# Run this from a shell that has the GPU SLURM allocation.
# The dispatcher is idempotent — splits with a populated metrics.json are skipped.

set -euo pipefail

REPO_ROOT="/home/snirhordan/ito/RandomSearchNNs"
LOG_DIR="${REPO_ROOT}/runs/qm9_rsnn"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/dispatch_resume_${TS}.log"

cd "${REPO_ROOT}"

source /home/snirhordan/miniconda3/etc/profile.d/conda.sh
conda activate rwnn

if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)" 2>/dev/null; then
    echo "[resume.sh] FATAL: CUDA not visible in this shell. Aborting." >&2
    python3 -c "import torch; print('cuda available:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())" >&2 || true
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2 2>&1 || true
    exit 1
fi

NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "[resume.sh] starting dispatcher with ${NGPU} GPU(s); log -> ${LOG}"

nohup python3 -u "${LOG_DIR}/dispatch.py" \
    --out_root "${LOG_DIR}" \
    --max_parallel "${NGPU}" \
    --epochs 10 \
    --patience 3 \
    > "${LOG}" 2>&1 &

PID=$!
disown "${PID}" 2>/dev/null || true
echo "[resume.sh] dispatcher pid=${PID} log=${LOG}"
echo "[resume.sh] tail -f ${LOG}"
