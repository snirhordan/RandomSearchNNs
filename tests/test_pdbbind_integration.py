"""End-to-end integration: PDBbind reader -> protein-ligand sampler -> RSNN_TRSF_Reg.

Validates that the atom-level Data built by generation/pdbbind.build_data,
sampled by utils.search.sample_protein_ligand, batched by the same PyG
DataLoader train_qm9 uses, forwards and backwards through the shared
transformer regression head with NO shape/contract mismatch. Synthetic data
only -- no real dataset needed. This is the contract that quickstart/
train_pdbbind.py will rely on.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.pdbbind import build_data, build_pdbbind_vocab
from utils.search import sample_protein_ligand
from quickstart.train_qm9 import RSNN_TRSF_Reg, compute_pe_in_dim

# keys the dataset strips after sampling so variable-N node attrs collate cleanly
_STRIP = ("x", "pos", "z", "edge_index", "edge_attr", "segment",
          "residue_id", "num_nodes", "pdb_id")


def _complex(seed, n_lig=4):
    """Synthetic complex: 2 protein residues (3+2 atoms) + a connected ligand chain."""
    rng = np.random.default_rng(seed)
    # protein: residue 0 (atoms 0,1,2), residue 1 (atoms 3,4), spread out
    prot = np.array([[0, 0, 0], [1.4, 0, 0], [2.6, 0.6, 0],
                     [4.0, 0.4, 0], [5.2, -0.3, 0]], dtype=np.float64)
    # ligand: a chain ~1.5A spacing, placed so at least one atom is < 4.5A from protein
    lig = np.array([[3.0, 2.0, 0], [3.0, 3.4, 0], [3.0, 4.8, 0], [4.2, 5.4, 0]],
                   dtype=np.float64)[:n_lig]
    lig += rng.normal(0, 0.05, lig.shape)  # tiny jitter per seed
    coords = np.vstack([prot, lig])
    elements = ['C', 'N', 'O', 'C', 'N'] + ['C', 'O', 'N', 'C'][:n_lig]
    segment = [0, 0, 0, 0, 0] + [1] * n_lig
    residue_id = [0, 0, 0, 1, 1] + [-1] * n_lig
    return {
        'coords': coords, 'elements': elements,
        'segment': np.array(segment), 'residue_id': np.array(residue_id),
        'label': float(seed), 'pdb_id': f'synth{seed}',
    }


def test_reader_sampler_model_end_to_end():
    torch.manual_seed(0)
    import random
    random.seed(0)

    decoded = [_complex(1), _complex(2, n_lig=3)]
    vocab = build_pdbbind_vocab(decoded)
    assert vocab['PAD'] == len(vocab) - 1  # embedding padding_idx contract

    s, m, max_len = 8, 4, 24
    angle_K, dihedral_K = 8, 4

    data_list = []
    for d in decoded:
        data = build_data(d, vocab, cutoff=4.5, eps=1e-5)
        # ligand atoms must be connected for a meaningful DFS
        data = sample_protein_ligand(
            data, m=m, s=s, max_len=max_len, vocab=vocab,
            angles=True, dihedrals=True, angle_K=angle_K, dihedral_K=dihedral_K,
        )
        for k in _STRIP:
            if hasattr(data, k):
                delattr(data, k)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    pe_in_dim = compute_pe_in_dim("search", w=s, distances=0, mol_edge_feat=0,
                                  angles=1, dihedrals=1,
                                  angle_K=angle_K, dihedral_K=dihedral_K)
    # sanity: the sampler produced exactly this many pe channels
    assert batch.walk_pe.shape[-1] == pe_in_dim

    pe_out_dim, h_dim, num_layers, nhead = 16, 32, 2, 4  # d_model=48, 48%4==0
    model = RSNN_TRSF_Reg(pe_in_dim, pe_out_dim, h_dim, 1, num_layers,
                          len(vocab), reduce="mean", dropout=0.0, nhead=nhead,
                          ffn_mult=4, attn_mode="full", pos_enc="sinusoidal")

    out = model(batch)
    assert out.shape == (2, 1), out.shape
    assert torch.isfinite(out).all()

    loss = torch.nn.functional.mse_loss(out.squeeze(-1), batch.y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_rope_and_causal_variants_forward():
    """The other pos_enc / attn_mode combos also forward cleanly on sampler output."""
    import random
    torch.manual_seed(1); random.seed(1)
    decoded = [_complex(3), _complex(4)]
    vocab = build_pdbbind_vocab(decoded)
    s, m, max_len = 8, 3, 20
    data_list = []
    for d in decoded:
        data = build_data(d, vocab, cutoff=4.5, eps=1e-5)
        data = sample_protein_ligand(data, m=m, s=s, max_len=max_len, vocab=vocab,
                                     angles=False, dihedrals=False)
        for k in _STRIP:
            if hasattr(data, k):
                delattr(data, k)
        data_list.append(data)
    batch = next(iter(DataLoader(data_list, batch_size=2)))
    pe_in_dim = compute_pe_in_dim("search", w=s, distances=0, mol_edge_feat=0)
    for pos_enc, attn_mode in [("rope", "full"), ("none", "causal")]:
        model = RSNN_TRSF_Reg(pe_in_dim, 16, 32, 1, 2, len(vocab), reduce="sum",
                              nhead=4, pos_enc=pos_enc, attn_mode=attn_mode)
        out = model(batch)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()
