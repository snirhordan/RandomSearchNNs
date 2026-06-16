#!/usr/bin/env python3
"""Compute EGNN-style meann/MAD for all 12 QM9 targets from preprocessed caches.

Reads data.y (PyG QM9 units, post-HARTREE_TO_EV for energy targets) from each
mols_<target>.pt cache, filters to the Cormorant train fold, computes
meann = mean(values); MAD = mean(|values - meann|). Writes the augmented
normalization map back into preprocessing_audit.json.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

REPO = Path("/home/snirhordan/ito/RandomSearchNNs")
CACHE_DIR = REPO / "data/qm9/qm9_d_rwnn_cache"
AUDIT_JSON = REPO / "runs/qm9_compare/preprocessing_audit.json"
CORMORANT_DIR = REPO / "external/egnn/qm9/temp/qm9"

# Map orchestrator target name → cache file suffix (handles case differences).
# Both r2 and R2 -> mols_R2.pt; everything else matches case.
TARGET_TO_CACHE = {
    "mu": "mu", "alpha": "alpha", "homo": "homo", "lumo": "lumo",
    "gap": "gap", "R2": "R2", "r2": "R2", "zpve": "zpve",
    "U0": "U0", "U": "U", "H": "H", "G": "G", "Cv": "Cv",
}


def main():
    # Load Cormorant train indices (0-indexed gdb_idx)
    train_npz = np.load(CORMORANT_DIR / "train.npz")
    train_cormorant_idx = set(int(x - 1) for x in train_npz["index"])
    print(f"Cormorant train: {len(train_cormorant_idx)} mols")

    # Load existing audit JSON to preserve schema
    audit = json.loads(AUDIT_JSON.read_text())
    if "normalization" not in audit:
        audit["normalization"] = {}

    for target, cache_name in TARGET_TO_CACHE.items():
        cache_path = CACHE_DIR / f"mols_{cache_name}.pt"
        if not cache_path.exists():
            print(f"  SKIP {target}: cache {cache_path} not found")
            continue
        if target in audit["normalization"] and target == "gap":
            print(f"  KEEP {target}: existing meann={audit['normalization'][target]['meann']:.6f}")
            continue
        t0 = time.time()
        mols = torch.load(str(cache_path), weights_only=False)
        # Filter to Cormorant train fold via data.idx
        ys = []
        for m in mols:
            if hasattr(m, "idx") and m.idx is not None:
                if int(m.idx.item()) in train_cormorant_idx:
                    ys.append(float(m.y.item()))
        if not ys:
            # Fallback: cache mols may lack .idx; recover positionally from PyG QM9
            print(f"  {target}: cache lacks .idx, computing positionally from PyG QM9")
            from torch_geometric.datasets import QM9
            from generation.qm9 import QM9_TARGET_INDEX
            qm9 = QM9(root=str(REPO / "data/qm9"))
            pyg_idx_to_pos = {}
            for i in range(len(qm9)):
                pyg_idx_to_pos[int(qm9[i].idx.item())] = i
            # collect target values from PyG (already in eV/PyG units)
            tidx = QM9_TARGET_INDEX[target]
            for cidx in train_cormorant_idx:
                pos = pyg_idx_to_pos.get(cidx)
                if pos is None:
                    continue
                ys.append(float(qm9[pos].y[0, tidx].item()))
        arr = np.asarray(ys, dtype=np.float64)
        meann = float(arr.mean())
        mad = float(np.abs(arr - meann).mean())
        audit["normalization"][target] = {
            "meann": meann,
            "MAD": mad,
            "n_train": int(arr.size),
            "formula": "meann = train.mean(); MAD = mean(|train - meann|)",
        }
        print(f"  {target}: meann={meann:.6f} MAD={mad:.6f} n={arr.size} dt={time.time()-t0:.1f}s")

    # Also add r2 as alias of R2 (in case orchestrator switches case)
    if "R2" in audit["normalization"] and "r2" not in audit["normalization"]:
        audit["normalization"]["r2"] = audit["normalization"]["R2"]
        print(f"  r2: alias of R2")

    # Bump schema and note multi-target
    audit["target"] = "multi (gap + 11 others)"
    audit["units"] = "PyG QM9 native units (energies in eV via HARTREE_TO_EV)"
    AUDIT_JSON.write_text(json.dumps(audit, indent=2))
    print(f"Wrote {AUDIT_JSON}")
    print(f"normalization keys: {sorted(audit['normalization'].keys())}")


if __name__ == "__main__":
    sys.exit(main())
