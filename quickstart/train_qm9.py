#!/usr/bin/env python3
"""QM9 training CLI scaffold for d-RWNN.

This script:
  1. parses the d-RWNN-relevant CLI flags,
  2. loads the QM9 dataset,
  3. builds the atom-symbol vocabulary,
  4. prints a one-line status string and exits.

It also exposes helpers used by the smoke test on this branch (variant B,
separate ``walk_edge_feat`` tensor + dedicated edge encoder MLP):

  - ``build_add_edge_feat``: assemble the per-graph ``(N, N, d_edge)`` edge
    feature tensor from optional pairwise distances (RBF-expanded) and
    optional bond features (broadcast onto the dense N x N grid).
  - ``compute_edge_in_dim``: ``K*distances + 3*mol_edge_feat``.
  - ``build_rwnn``: factory for ``models.rwnn.RWNN`` that threads
    ``edge_in_dim``/``edge_out_dim``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Allow ``from generation.qm9 import ...`` when invoked from quickstart/.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from generation.qm9 import (  # noqa: E402
    load_qm9,
    build_qm9_vocab,
    RBFExpansion,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QM9 training scaffold for d-RWNN."
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


def compute_edge_in_dim(distances: int, mol_edge_feat: int, rbf_K: int = 16) -> int:
    """Total edge-feature width: K*distances + 3*mol_edge_feat."""
    return int(rbf_K) * int(distances) + 3 * int(mol_edge_feat)


def build_add_edge_feat(
    data,
    *,
    distances: int,
    mol_edge_feat: int,
    rbf: Optional[RBFExpansion] = None,
    rbf_K: int = 16,
    rbf_cutoff: float = 5.0,
) -> Optional[torch.Tensor]:
    """Build the per-graph (N, N, d_edge) edge-feature tensor for variant B.

    Channels are concatenated in the order
        [0:K]      RBF(distances)        if distances=1
        [K:K+3]    bond edge_attr (dense) if mol_edge_feat=1

    When both flags are 0 this returns ``None`` so callers can pass
    ``add_edge_feat=None`` to the sampler (preserving the baseline RWNN code
    path bit-for-bit).
    """
    d_edge = compute_edge_in_dim(distances, mol_edge_feat, rbf_K)
    if d_edge == 0:
        return None
    n = int(data.x.shape[0]) if hasattr(data, 'x') else int(data.num_nodes)
    out = torch.zeros((n, n, d_edge), dtype=torch.float)
    cursor = 0
    if distances:
        if rbf is None:
            rbf = RBFExpansion(K=rbf_K, cutoff=rbf_cutoff)
        if hasattr(data, 'distances') and data.distances is not None:
            dist = data.distances
        else:
            pos = data.pos.float()
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = diff.norm(dim=-1)
        with torch.no_grad():
            rbf_feat = rbf(dist)  # (N, N, K)
        out[..., cursor:cursor + rbf_K] = rbf_feat
        cursor += rbf_K
    if mol_edge_feat:
        # Dense N x N x 3 from sparse edge_index/edge_attr (shape (E, 3)).
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.numel() > 0:
            ei = data.edge_index
            ea = data.edge_attr.float()
            d3 = ea.shape[-1]
            tmp = torch.zeros((n, n, d3), dtype=torch.float)
            for k in range(ei.shape[1]):
                i, j = int(ei[0, k].item()), int(ei[1, k].item())
                tmp[i, j] = ea[k]
                tmp[j, i] = ea[k]
            out[..., cursor:cursor + 3] = tmp
        cursor += 3
    return out


def build_rwnn(
    *,
    pe_in_dim: int,
    pe_out_dim: int,
    hid_dim: int,
    out_dim: int,
    num_layers: int,
    n_emb: int,
    reduce: str,
    edge_in_dim: int = 0,
    edge_out_dim: int = 16,
):
    """Construct a variant-B RWNN with optional edge encoder."""
    from models.rwnn import RWNN
    return RWNN(
        pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce,
        edge_in_dim=edge_in_dim, edge_out_dim=edge_out_dim,
    )


def main() -> int:
    args = _build_argparser().parse_args()

    ds = load_qm9(root=args.data_root)
    vocab = build_qm9_vocab(ds, tokenizer=None)
    edge_in_dim = compute_edge_in_dim(args.distances, args.mol_edge_feat, args.rbf_K)

    print(
        f"Loaded QM9 (N={len(ds)}), target={args.target}, "
        f"distances={args.distances}, mol_edge_feat={args.mol_edge_feat}, "
        f"edge_in_dim={edge_in_dim}, vocab={len(vocab)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
