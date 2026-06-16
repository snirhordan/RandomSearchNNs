#!/usr/bin/env bash
# Set up vgsatorras/egnn for QM9 sweep. Idempotent.
#
# Patches applied:
#   1. figshare URLs migrated to api.figshare.com (springernature endpoints
#      return HTTP 202 with 0 bytes since ~2024).
#   2. Fix deprecated np.int -> int (numpy >= 1.20).
#   3. Add skip-if-exists guards so subsequent runs don't re-download.
#
# Run from repo root:
#   bash runs/qm9_egnn/setup_egnn.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EGNN_DIR="${REPO}/external/egnn"

if [ ! -d "${EGNN_DIR}/.git" ]; then
    mkdir -p "${REPO}/external"
    git clone --depth 1 https://github.com/vgsatorras/egnn.git "${EGNN_DIR}"
fi

PREP="${EGNN_DIR}/qm9/data/prepare/qm9.py"

# Patch 1: data URL
sed -i.bak \
  "s|https://springernature.figshare.com/ndownloader/files/3195389|https://api.figshare.com/v2/file/download/3195389|" \
  "${PREP}"
# Patch 2: excluded URL
sed -i \
  "s|https://springernature.figshare.com/ndownloader/files/3195404|https://api.figshare.com/v2/file/download/3195404|" \
  "${PREP}"
# Patch 3: thermo URL
sed -i \
  "s|https://springernature.figshare.com/ndownloader/files/3195395|https://api.figshare.com/v2/file/download/3195395|" \
  "${PREP}"
# Patch 4: np.int -> int
sed -i "s|np\.zeros(len(charges), dtype=np\.int)|np.zeros(len(charges), dtype=int)|" "${PREP}"

# Patch 5: skip-if-exists for download URLs (guard against figshare flakiness)
python3 - <<'PY'
from pathlib import Path
p = Path("__EGNN_DIR__/qm9/data/prepare/qm9.py".replace("__EGNN_DIR__", "${EGNN_DIR}"))
s = p.read_text()
# Excluded file guard
old = "    gdb9_url_excluded = 'https://api.figshare.com/v2/file/download/3195404'\n    gdb9_txt_excluded = join(gdb9dir, 'uncharacterized.txt')\n    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)"
new = "    gdb9_url_excluded = 'https://api.figshare.com/v2/file/download/3195404'\n    gdb9_txt_excluded = join(gdb9dir, 'uncharacterized.txt')\n    import os.path as _osp\n    if not (_osp.exists(gdb9_txt_excluded) and _osp.getsize(gdb9_txt_excluded) > 1000):\n        urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)"
if old in s:
    s = s.replace(old, new); p.write_text(s)
# Thermo file guard
old2 = "    gdb9_url_thermo = 'https://api.figshare.com/v2/file/download/3195395'\n    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')\n\n    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)"
new2 = "    gdb9_url_thermo = 'https://api.figshare.com/v2/file/download/3195395'\n    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')\n    import os.path as _osp\n    if not (_osp.exists(gdb9_txt_thermo) and _osp.getsize(gdb9_txt_thermo) > 100):\n        urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)"
if old2 in s:
    s = s.replace(old2, new2); p.write_text(s)
print("patches applied")
PY

echo "EGNN ready at ${EGNN_DIR}"
echo "Now ensure matplotlib is installed:"
echo "  pip install matplotlib"
