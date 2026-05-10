#!/usr/bin/env python3
"""QM9 training CLI scaffold for d-RWNN (variant A: PE-append).

This module:
  1. parses the d-RWNN-relevant CLI flags,
  2. loads the QM9 dataset,
  3. builds the atom-symbol vocabulary,
  4. exposes helpers (``build_add_edge_feat``, ``compute_pe_in_dim``,
     ``edge_attr_to_dense``) that wire the new ``add_edge_feat`` plumbing
     through to ``utils.search.sample_walks_adaptive`` and friends.

The full training loop is intentionally NOT included yet -- it will be wired
in a follow-up commit once the OTHER agent's variant is benchmarked against
this one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# Allow ``from generation.qm9 import ...`` when invoked from quickstart/.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generation.qm9 import (  # noqa: E402
    RBFExpansion,
    build_qm9_vocab,
    load_qm9,
    qm9_to_data,
)


# ---------------------------------------------------------------------------
# Helpers wiring distances + bond features through into add_edge_feat
# ---------------------------------------------------------------------------

# OGB-style bond feature width used throughout the QM9 pipeline:
# (bond_type_idx, stereo, is_conjugated). Centralised so add_edge_feat
# bonus accounting and the dense scatter agree.
BOND_FEAT_DIM: int = 3


def edge_attr_to_dense(data) -> torch.Tensor:
    """Scatter ``data.edge_attr`` (E x BOND_FEAT_DIM) into a dense
    ``(N, N, BOND_FEAT_DIM)`` tensor.

    Non-edge entries are zero. The input ``data`` must expose ``edge_index``
    of shape ``(2, E)``, ``edge_attr`` of shape ``(E, BOND_FEAT_DIM)`` and
    ``num_nodes`` (or ``x``). The output tensor is placed on the same
    device as ``data.edge_index`` so downstream samplers (which index into
    it) do not silently incur a CPU<->GPU round-trip.
    """
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        n = int(data.num_nodes)
    elif hasattr(data, "x") and data.x is not None:
        n = int(data.x.size(0))
    else:
        n = int(data.edge_index.max()) + 1
    e_attr = data.edge_attr
    device = data.edge_index.device if hasattr(data, "edge_index") else torch.device("cpu")
    if e_attr is None or e_attr.numel() == 0:
        return torch.zeros((n, n, BOND_FEAT_DIM), dtype=torch.float, device=device)
    if e_attr.dim() != 2 or e_attr.size(-1) != BOND_FEAT_DIM:
        raise ValueError(
            f"edge_attr_to_dense expects (E, {BOND_FEAT_DIM}); "
            f"got shape {tuple(e_attr.shape)}"
        )
    out = torch.zeros((n, n, BOND_FEAT_DIM), dtype=torch.float, device=device)
    src = data.edge_index[0].long()
    dst = data.edge_index[1].long()
    out[src, dst] = e_attr.float()
    # symmetrize (just in case the bond graph isn't fully bidirected already)
    out[dst, src] = e_attr.float()
    return out


def build_add_edge_feat(
    data,
    distances: int,
    mol_edge_feat: int,
    rbf: Optional[RBFExpansion] = None,
) -> Optional[torch.Tensor]:
    """Build the ``add_edge_feat`` tensor for one ``Data``.

    Returns ``None`` when both flags are 0 (backwards-compat path: the
    sampler is called with ``add_edge_feat=None`` and produces a
    bit-identical ``walk_pe`` to the baseline).

    Otherwise returns a ``(N, N, B)`` float tensor with::

        B = K * distances + BOND_FEAT_DIM * mol_edge_feat

    where the first ``K`` channels (when ``distances=1``) are the Gaussian
    RBF expansion of pairwise Euclidean distances, and the trailing
    ``BOND_FEAT_DIM`` channels (when ``mol_edge_feat=1``) hold the dense
    bond-feature expansion of ``data.edge_attr``.
    """
    parts = []
    if distances:
        if rbf is None:
            raise ValueError("distances=1 requires rbf=RBFExpansion(...)")
        if not hasattr(data, "distances") or data.distances is None:
            raise ValueError("distances=1 requires data.distances")
        # data.distances: (N, N) -> RBF -> (N, N, K)
        parts.append(rbf(data.distances))
    if mol_edge_feat:
        parts.append(edge_attr_to_dense(data))
    if not parts:
        return None
    return torch.cat(parts, dim=-1).float()


def compute_pe_in_dim(walk_type: str, w: int, distances: int,
                       mol_edge_feat: int, rbf_K: int = 16) -> int:
    """Return the model's ``pe_in_dim`` for the given config.

    Baseline (``distances=0`` and ``mol_edge_feat=0``) returns the same
    value the legacy pipeline used:

    - ``walk``-style samplers (``walk``, ``walk_ada``) use ``2 * w``
      because they emit ``[encoding_repeat | encoding_edge]``.
    - DFS / BFS samplers use ``w`` (single ``encoding_edge`` block).
    - MDLR / RUM (no walk_pe in baseline) get ``0`` -- but the model
      should not be constructed in that mode unless the user explicitly
      supplies a non-trivial bonus B.
    """
    bonus = rbf_K * distances + BOND_FEAT_DIM * mol_edge_feat
    if walk_type in ("walk", "walk_ada"):
        return 2 * w + bonus
    if walk_type in ("dfs", "bfs"):
        return w + bonus
    if walk_type in ("walk_mdlr", "walk_rum",
                     "walk_mdlr_ada", "walk_rum_ada"):
        if bonus == 0:
            raise ValueError(
                f"walk_type={walk_type!r} produces no walk_pe in the baseline "
                "(distances=0 and mol_edge_feat=0); pe_in_dim would be 0 and "
                "the PE Linear layer is not constructible. Enable distances "
                "and/or mol_edge_feat, or pick a walk-style sampler."
            )
        return bonus
    raise ValueError(f"unknown walk_type {walk_type!r}")


# ---------------------------------------------------------------------------
# CLI scaffold (training loop intentionally TBD)
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QM9 training scaffold for d-RWNN (variant A)."
    )
    # d-RWNN data choices
    p.add_argument("--target", choices=["U0", "gap", "mu"], default="U0")
    p.add_argument("--distances", type=int, choices=[0, 1], default=0,
                   help="If 1, attach pairwise Euclidean distance matrix and "
                        "feed an RBF expansion into walk_pe.")
    p.add_argument("--mol_edge_feat", type=int, choices=[0, 1], default=0,
                   help="If 1, append the 3-channel bond-feature stream to "
                        "walk_pe.")
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

    pe_in_dim = compute_pe_in_dim(
        walk_type=args.walk_type,
        w=args.w,
        distances=args.distances,
        mol_edge_feat=args.mol_edge_feat,
        rbf_K=args.rbf_K,
    )
    bonus = args.rbf_K * args.distances + BOND_FEAT_DIM * args.mol_edge_feat

    print(
        f"Loaded QM9 (N={len(ds)}), target={args.target}, "
        f"distances={args.distances}, mol_edge_feat={args.mol_edge_feat}, "
        f"bonus={bonus}, pe_in_dim={pe_in_dim}, vocab={len(vocab)}"
    )
    return 0


__all__ = [
    "BOND_FEAT_DIM",
    "build_add_edge_feat",
    "compute_pe_in_dim",
    "edge_attr_to_dense",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
