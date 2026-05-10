#!/usr/bin/env python3
"""QM9 training CLI scaffold for d-RWNN.

This script intentionally does NOT run a training loop yet. It only:
  1. parses the d-RWNN-relevant CLI flags,
  2. loads the QM9 dataset,
  3. builds the atom-symbol vocabulary,
  4. prints a one-line status string and exits.

The full training loop will be wired in once the integration variant
(per-step distance augmentation vs. RBF edge feature) is selected and
``utils/search.py`` / ``models/rwnn.py`` are updated on the corresponding
branches.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow ``from generation.qm9 import ...`` when invoked from quickstart/.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generation.qm9 import (  # noqa: E402
    load_qm9,
    build_qm9_vocab,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QM9 training scaffold for d-RWNN (no training loop yet)."
    )
    # d-RWNN data choices
    p.add_argument("--target", choices=["U0", "gap", "mu"], default="U0")
    p.add_argument("--distances", type=int, choices=[0, 1], default=0,
                   help="If 1, attach pairwise Euclidean distance matrix.")
    p.add_argument("--mol_edge_feat", type=int, choices=[0, 1], default=0,
                   help="If 1, retain bond-type edge features.")
    p.add_argument("--rbf_K", type=int, default=16,
                   help="Number of Gaussian RBF basis functions for the "
                        "distance expansion.")
    p.add_argument("--rbf_cutoff", type=float, default=5.0,
                   help="RBF cutoff in Angstroms (centers spaced in "
                        "[0, cutoff]).")
    # ClinTox-aligned hyperparameters
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early_stopping", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--h_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--w", type=int, default=8)
    p.add_argument("--reduce", choices=["mean", "sum", "max"], default="mean")
    p.add_argument("--walk_type", default="walk_ada")
    p.add_argument("--n_splits", type=int, default=3)
    p.add_argument("--device_idx", type=int, default=0)
    # Data root
    p.add_argument("--data_root", default="./data/qm9")
    return p


def main() -> int:
    args = _build_argparser().parse_args()

    ds = load_qm9(root=args.data_root)
    vocab = build_qm9_vocab(ds, tokenizer=None)

    print(
        f"Loaded QM9 (N={len(ds)}), target={args.target}, "
        f"distances={args.distances}, mol_edge_feat={args.mol_edge_feat}, "
        f"vocab={len(vocab)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
