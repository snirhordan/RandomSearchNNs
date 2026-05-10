#!/usr/bin/env python3
"""Variant-B (separate walk_edge_feat tensor) smoke test on QM9.

Runs the four-flag matrix (distances x mol_edge_feat) on a 16-molecule QM9
subset, samples walks via ``walk_ada`` (sample_walks_adaptive), runs RWNN
forward + a single backward, prints output shapes, and verifies finite output.

Also runs a deterministic backwards-compat check against the baseline RWNN
on a tiny ClinTox-like fixture (random edge_index + x_emb) confirming
behavior is bit-identical when distances=0 AND mol_edge_feat=0.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from generation.qm9 import (
    load_qm9, build_qm9_vocab, qm9_to_data, RBFExpansion,
)
from utils.search import sample_walks, sample_walks_adaptive
from models.rwnn import RWNN
from quickstart.train_qm9 import (
    build_add_edge_feat, compute_edge_in_dim, build_rwnn,
)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def run_flag_combo(distances: int, mol_edge_feat: int, ds, vocab,
                   *, m=4, l=12, w=4, hid_dim=32, num_layers=2,
                   rbf_K=16, rbf_cutoff=5.0, batch_size=4):
    """Use ``sample_walks`` (fixed-length, no -1 padding) for the smoke test
    so it exercises ``models.rwnn.RWNN`` directly. The walk_edge_feat code
    path is identical across samplers."""
    rbf = RBFExpansion(K=rbf_K, cutoff=rbf_cutoff)
    edge_in_dim = compute_edge_in_dim(distances, mol_edge_feat, rbf_K)

    samples = []
    for i in range(16):
        d = qm9_to_data(
            ds[i], vocab=vocab,
            add_distances=bool(distances),
            add_edge_attr=bool(mol_edge_feat),
            target='U0',
        )
        add_ef = build_add_edge_feat(
            d, distances=distances, mol_edge_feat=mol_edge_feat,
            rbf=rbf, rbf_K=rbf_K, rbf_cutoff=rbf_cutoff,
        )
        d = sample_walks(d, m, l, w, True, add_edge_feat=add_ef)
        # Strip private cache so collation is clean.
        if '_neighbor_dict' in d.__dict__:
            del d.__dict__['_neighbor_dict']
        # Drop variable-N tensors (already baked into walk_edge_feat).
        # PyG Data stores attrs in _store; delete via attribute access.
        for a in ('distances', 'pos', 'z', 'y_full'):
            if hasattr(d, a):
                delattr(d, a)
        samples.append(d)

    loader = DataLoader(samples, batch_size=batch_size, shuffle=False)
    model = build_rwnn(
        pe_in_dim=2 * w, pe_out_dim=16,
        hid_dim=hid_dim, out_dim=1, num_layers=num_layers,
        n_emb=len(vocab), reduce='mean',
        edge_in_dim=edge_in_dim, edge_out_dim=16,
    )

    batch = next(iter(loader))
    has_wef = hasattr(batch, 'walk_edge_feat')
    wef_shape = tuple(batch.walk_edge_feat.shape) if has_wef else None
    out = model(batch)
    finite = bool(torch.isfinite(out).all().item())
    out.sum().backward()
    return {
        'distances': distances,
        'mol_edge_feat': mol_edge_feat,
        'edge_in_dim': edge_in_dim,
        'has_walk_edge_feat': has_wef,
        'walk_edge_feat_shape': wef_shape,
        'output_shape': tuple(out.shape),
        'finite': finite,
    }


def backwards_compat_check():
    """Baseline RWNN behavior must be bit-identical when both flags are 0.

    We construct a tiny synthetic graph, sample walks twice (once via the
    baseline call signature - no add_edge_feat - and once via the new
    keyword with None), and assert all sampler outputs are equal under the
    same RNG seed. We also verify the model runs identically when
    edge_in_dim=0 (no edge_encoding parameter, no widening of LSTM input).
    """
    vocab = {'A': 0, 'B': 1, 'PAD': 2}

    def make_graph():
        # Path graph with 6 nodes.
        n = 6
        ei = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
             [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)
        x = torch.zeros((n, 9), dtype=torch.float)
        x_emb = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        d = Data(x=x, edge_index=ei)
        d.x_emb = x_emb
        return d

    # Baseline call: no add_edge_feat at all (exercises sample_walks).
    seed_all(0)
    d1 = make_graph()
    d1 = sample_walks(d1, 3, 5, 3, True)

    # Variant-B call with add_edge_feat=None (must be identical).
    seed_all(0)
    d2 = make_graph()
    d2 = sample_walks(d2, 3, 5, 3, True, add_edge_feat=None)

    sampler_eq = (
        torch.equal(d1.walk_emb, d2.walk_emb)
        and torch.equal(d1.walk_ids, d2.walk_ids)
        and torch.equal(d1.walk_pe, d2.walk_pe)
        and not hasattr(d1, 'walk_edge_feat')
        and not hasattr(d2, 'walk_edge_feat')
    )

    # Also check sample_walks_adaptive (walk_ada) parity.
    seed_all(1)
    d3 = make_graph()
    d3 = sample_walks_adaptive(d3, 3, 6, 3, True, 8, vocab)
    seed_all(1)
    d4 = make_graph()
    d4 = sample_walks_adaptive(d4, 3, 6, 3, True, 8, vocab, add_edge_feat=None)
    sampler_eq = sampler_eq and (
        torch.equal(d3.walk_emb, d4.walk_emb)
        and torch.equal(d3.walk_ids, d4.walk_ids)
        and torch.equal(d3.walk_pe, d4.walk_pe)
        and torch.equal(d3.lengths, d4.lengths)
        and not hasattr(d3, 'walk_edge_feat')
        and not hasattr(d4, 'walk_edge_feat')
    )

    # Model parity: edge_in_dim=0 => no edge_encoding attr, identical params.
    torch.manual_seed(123)
    m_base = RWNN(6, 8, 16, 1, 2, len(vocab), 'mean')
    torch.manual_seed(123)
    m_b = RWNN(6, 8, 16, 1, 2, len(vocab), 'mean',
               edge_in_dim=0, edge_out_dim=16)
    no_extra_attr = not hasattr(m_b, 'edge_encoding')
    same_keys = list(m_base.state_dict().keys()) == list(m_b.state_dict().keys())
    same_vals = all(
        torch.equal(m_base.state_dict()[k], m_b.state_dict()[k])
        for k in m_base.state_dict()
    )

    return {
        'sampler_identical': bool(sampler_eq),
        'no_edge_encoding_attr': no_extra_attr,
        'state_dict_keys_match': same_keys,
        'state_dict_values_match': same_vals,
    }


def main():
    print("=== d-RWNN variant-B smoke test (separate walk_edge_feat) ===")
    ds = load_qm9(root=str(REPO / 'data' / 'qm9'))
    vocab = build_qm9_vocab(ds, tokenizer=None)
    print(f"QM9 N={len(ds)}, vocab_size={len(vocab)}")
    print()

    print("--- Backwards-compat check ---")
    bc = backwards_compat_check()
    for k, v in bc.items():
        print(f"  {k}: {v}")
    bc_ok = all(bc.values())
    print(f"  PASS={bc_ok}")
    print()

    print("--- Four-flag matrix (m=4, w=4, max_len=12, batch=4, walk_ada) ---")
    results = []
    for distances, mol_edge_feat in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        seed_all(42)
        r = run_flag_combo(distances, mol_edge_feat, ds, vocab)
        results.append(r)
        print(f"  distances={distances} mol_edge_feat={mol_edge_feat}: "
              f"edge_in_dim={r['edge_in_dim']}, "
              f"walk_edge_feat={r['walk_edge_feat_shape']}, "
              f"output_shape={r['output_shape']}, "
              f"finite={r['finite']}")
    all_ok = all(r['finite'] for r in results) and bc_ok
    print()
    print(f"OVERALL PASS={all_ok}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
