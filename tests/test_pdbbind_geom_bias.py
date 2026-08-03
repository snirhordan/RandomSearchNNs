"""Tests for the geometric attention bias on the protein-ligand (PDBbind) path.

Covers the wiring added to ``quickstart/train_pdbbind.py`` so the transformer
can consume the E(3)-invariant geometric attention bias (``GeometricAttentionBias``,
already used by the QM9 trainer) on protein-ligand complexes:

  1. ``sample_protein_ligand(emit_xyz=True)`` walk_xyz correctness -- shape,
     zero-padding, and per-position xyz == data.pos[node] over BOTH the protein
     prefix and the ligand walk.
  2. ``PDBbindWalkDataset(emit_xyz=1)`` attaches ``walk_xyz`` and it survives
     PyG collation as (B*m, max_len, 3) (while ``pos`` is stripped).
  3. ``RSNN_TRSF_Reg(geom_bias=True)`` runs forward+backward over a real
     protein-ligand batch with finite loss and gradients into the bias MLP.
  4. Enabling the bias adds exactly +1,444 params for the default geom config
     (rbf_K=16, angle_K=8, dihedral_K=4, hidden=32, nhead=4).
  5. End-to-end ``main(--geom_bias 1)`` smoke (auto-enables emit_xyz) yields
     finite metrics and records geom_bias=1; ``--geom_bias 1 --base lstm`` fails
     fast.

Run::

    source /home/snirhordan/miniconda3/etc/profile.d/conda.sh && conda activate rwnn
    cd /home/snirhordan/ito/RandomSearchNNs
    python -m pytest tests/test_pdbbind_geom_bias.py -v --tb=short
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.pdbbind import build_data, build_pdbbind_vocab  # noqa: E402
from quickstart.train_pdbbind import (  # noqa: E402
    PDBbindWalkDataset,
    main as train_main,
)
from quickstart.train_qm9 import RSNN_TRSF_Reg, compute_pe_in_dim  # noqa: E402
from utils.search import sample_protein_ligand  # noqa: E402


# ---------------------------------------------------------------------------
# Raw-Data fixture (keeps data.pos) for the sampler-level walk_xyz check.
# Mirrors tests/test_protein_ligand_sampler.py: only protein atom 2 contacts
# the ligand, ligand is a fully-connected star (5,6,7,8).
# ---------------------------------------------------------------------------
CUTOFF = 4.5
VOCAB_RAW = {"PAD": 100}
PROT_IDX = [0, 1, 2, 3, 4]
LIG_IDX = [5, 6, 7, 8]


def _complex_pos():
    return torch.tensor([
        [0.0, 0.0, 0.0],    # 0  protein residue 0
        [2.0, 0.3, 0.0],    # 1  protein residue 0
        [4.0, 0.0, 0.5],    # 2  protein residue 0 (interface atom)
        [20.0, 1.0, 0.0],   # 3  protein residue 1
        [22.0, 0.0, 1.0],   # 4  protein residue 1
        [7.0, 0.0, 0.0],    # 5  ligand (star center)
        [8.0, 1.0, 0.0],    # 6  ligand
        [8.0, -1.0, 0.0],   # 7  ligand
        [7.0, 0.0, 2.0],    # 8  ligand
    ])


def _radius_graph_edge_index(pos, cutoff=CUTOFF):
    n = pos.shape[0]
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and torch.norm(pos[i] - pos[j]).item() <= cutoff:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def _make_raw_complex():
    pos = _complex_pos()
    n = pos.shape[0]
    data = Data(edge_index=_radius_graph_edge_index(pos), pos=pos, num_nodes=n)
    data.x_emb = torch.arange(n, dtype=torch.long)
    data.segment = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    data.residue_id = torch.tensor([0, 0, 0, 1, 1, -1, -1, -1, -1], dtype=torch.long)
    return data


# ---------------------------------------------------------------------------
# Cache-shaped fixture (via build_data/build_pdbbind_vocab) for the dataset /
# trainer-level tests. Same recipe as tests/test_train_pdbbind_smoke.py.
# ---------------------------------------------------------------------------
def _decoded_complex(seed, n_lig=4):
    rng = np.random.default_rng(seed)
    prot = np.array([[0, 0, 0], [1.4, 0, 0], [2.6, 0.6, 0],
                     [4.0, 0.4, 0], [5.2, -0.3, 0]], dtype=np.float64)
    lig = np.array([[3.0, 2.0, 0], [3.0, 3.4, 0], [3.0, 4.8, 0], [4.2, 5.4, 0]],
                   dtype=np.float64)[:n_lig]
    lig += rng.normal(0, 0.05, lig.shape)
    coords = np.vstack([prot, lig])
    elements = ['C', 'N', 'O', 'C', 'N'] + ['C', 'O', 'N', 'C'][:n_lig]
    segment = [0, 0, 0, 0, 0] + [1] * n_lig
    residue_id = [0, 0, 0, 1, 1] + [-1] * n_lig
    return {
        'coords': coords, 'elements': elements,
        'segment': np.array(segment), 'residue_id': np.array(residue_id),
        'label': 4.0 + 0.5 * seed, 'pdb_id': f'synth{seed}',
    }


def _data_list_and_vocab(n=6):
    decoded = [_decoded_complex(s) for s in range(1, n + 1)]
    vocab = build_pdbbind_vocab(decoded)
    assert vocab['PAD'] == len(vocab) - 1 and vocab['UNK'] == 0
    data_list = [build_data(d, vocab, cutoff=4.5, eps=1e-5) for d in decoded]
    return data_list, vocab


# small, fast walk config reused across dataset/model tests
_M, _S, _MAXLEN, _LIGMAX = 3, 4, 24, 8
_ANGLE_K, _DIH_K = 4, 2


def _dataset(data_list, vocab, emit_xyz):
    return PDBbindWalkDataset(
        data_list, vocab, m=_M, s=_S, max_len=_MAXLEN, ligand_max_len=_LIGMAX,
        angles=1, dihedrals=1, angle_K=_ANGLE_K, dihedral_K=_DIH_K,
        distances=0, mol_edge_feat=0, rbf=None, emit_xyz=emit_xyz,
    )


# ===========================================================================
# 1. Sampler-level walk_xyz correctness (raw Data with pos).
# ===========================================================================
def test_walk_xyz_matches_pos_at_realized_positions():
    random.seed(3)
    data = _make_raw_complex()
    pos = _complex_pos()
    m, s, max_len = 5, 2, 20
    out = sample_protein_ligand(data, m, s, max_len, VOCAB_RAW, emit_xyz=True)

    assert out.walk_xyz.shape == (m, max_len, 3)
    saw_protein = saw_ligand = False
    for i in range(m):
        L = int(out.lengths[i])
        assert L >= 1
        for p in range(L):
            node = int(out.walk_ids[0, i, p])
            assert node >= 0
            assert torch.allclose(out.walk_xyz[i, p], pos[node], atol=1e-6)
            saw_protein |= node in PROT_IDX
            saw_ligand |= node in LIG_IDX
        # every padded position is exactly zero
        if L < max_len:
            assert torch.all(out.walk_xyz[i, L:] == 0)
    # coverage spans BOTH sub-sequences, not just the protein prefix
    assert saw_protein and saw_ligand


def test_emit_xyz_off_leaves_no_walk_xyz():
    data = _make_raw_complex()
    out = sample_protein_ligand(data, 3, 2, 20, VOCAB_RAW, emit_xyz=False)
    assert not hasattr(out, "walk_xyz")


# ===========================================================================
# 2. Dataset emission + collation.
# ===========================================================================
def test_dataset_emits_walk_xyz_and_strips_pos():
    data_list, vocab = _data_list_and_vocab()
    d = _dataset(data_list, vocab, emit_xyz=1)[0]
    assert hasattr(d, "walk_xyz")
    assert d.walk_xyz.shape == (_M, _MAXLEN, 3)
    # pos is a per-atom attr and must be stripped for clean collation. It is a
    # canonical PyG key (hasattr stays True), so assert it holds no data.
    assert d.pos is None
    assert "pos" not in set(d.keys())
    # emit_xyz=0 -> attribute absent (0 extra bytes on the default path)
    d0 = _dataset(data_list, vocab, emit_xyz=0)[0]
    assert not hasattr(d0, "walk_xyz")


def test_walk_xyz_survives_collation():
    data_list, vocab = _data_list_and_vocab()
    ds = _dataset(data_list, vocab, emit_xyz=1)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert hasattr(batch, "walk_xyz")
    # PyG concatenates the per-item (m, max_len, 3) blocks along dim 0.
    assert batch.walk_xyz.shape == (2 * _M, _MAXLEN, 3)
    # lengths collate the same way and index the same rows as walk_xyz.
    assert batch.lengths.shape == (2 * _M,)


# ===========================================================================
# 3. Model forward/backward with the bias enabled.
# ===========================================================================
def _build_trsf(vocab_size, geom_bias, nhead=4, h_dim=16, num_layers=2):
    pe_in_dim = compute_pe_in_dim(
        "search", w=_S, distances=0, mol_edge_feat=0,
        angles=1, dihedrals=1, angle_K=_ANGLE_K, dihedral_K=_DIH_K,
    )
    return RSNN_TRSF_Reg(
        pe_in_dim, 16, h_dim, 1, num_layers, vocab_size, "mean",
        dropout=0.1, nhead=nhead, ffn_mult=2, attn_mode="full", pos_enc="rope",
        geom_bias=geom_bias,
    )


def test_geom_bias_forward_backward_finite():
    torch.manual_seed(0)
    data_list, vocab = _data_list_and_vocab()
    ds = _dataset(data_list, vocab, emit_xyz=1)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = _build_trsf(len(vocab), geom_bias=True)
    assert model.geom_bias_mod is not None
    out = model(batch).squeeze(-1)
    loss = torch.nn.functional.mse_loss(out, batch.y.view(-1).float())
    assert torch.isfinite(loss)
    loss.backward()
    # gradient actually flows into the geometric-bias MLP
    grads = [p.grad for p in model.geom_bias_mod.parameters() if p.requires_grad]
    assert grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                         for g in grads)


def test_geom_bias_param_delta_is_1444():
    _, vocab = _data_list_and_vocab()
    n_off = sum(p.numel() for p in _build_trsf(len(vocab), geom_bias=False).parameters()
                if p.requires_grad)
    n_on = sum(p.numel() for p in _build_trsf(len(vocab), geom_bias=True).parameters()
               if p.requires_grad)
    # Linear(16+2*8+2*4=40, 32) + Linear(32, nhead=4) = 1312 + 132 = 1444.
    # rbf_centers is a buffer, so geom_bias=False adds exactly 0 params.
    assert n_on - n_off == 1444


# ===========================================================================
# 4. End-to-end trainer smoke + fail-fast guard.
# ===========================================================================
def _write_cache(tmp_path):
    data_list, vocab = _data_list_and_vocab()
    data_dir = tmp_path / "pdbbind_synth"
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data_list[0:2], data_dir / "train.pt")
    torch.save(data_list[2:4], data_dir / "valid.pt")
    torch.save(data_list[4:6], data_dir / "test.pt")
    with open(data_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)
    return data_dir


def test_trainer_smoke_geom_bias(tmp_path):
    data_dir = _write_cache(tmp_path)
    out_dir = tmp_path / "run_geom"
    argv = [
        "--data_dir", str(data_dir), "--out_dir", str(out_dir),
        "--base", "transformer",
        "--h_dim", "16", "--num_layers", "2", "--nhead", "2", "--ffn_mult", "2",
        "--pos_enc", "rope", "--attn_mode", "full", "--reduce", "mean",
        "--epochs", "2", "--batch_size", "2", "--num_workers", "0",
        "--patience", "50", "--seed", "0",
        "--m", "3", "--w", "4", "--max_len", "24", "--ligand_max_len", "8",
        "--angles", "1", "--dihedrals", "1", "--angle_K", "4", "--dihedral_K", "2",
        "--distances", "0", "--mol_edge_feat", "0", "--standardize", "1",
        # geom bias on: emit_xyz is auto-enabled through main()'s _ds_kw.
        "--geom_bias", "1", "--geom_rbf_cutoff", "10.0",
    ]
    assert train_main(argv) == 0
    with open(out_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["config"]["geom_bias"] == 1
    assert metrics["config"]["emit_xyz"] == 0        # explicit flag stays 0...
    assert metrics["config"]["geom_rbf_cutoff"] == 10.0
    for key in ("pearson_r", "spearman_rho", "rmse", "mae"):
        assert np.isfinite(metrics["test_metrics"][key])


def test_geom_bias_requires_transformer_base(tmp_path):
    data_dir = _write_cache(tmp_path)
    argv = [
        "--data_dir", str(data_dir), "--out_dir", str(tmp_path / "run_bad"),
        "--base", "lstm", "--geom_bias", "1",
        "--epochs", "1", "--batch_size", "2", "--num_workers", "0",
        "--m", "3", "--w", "4", "--max_len", "24", "--ligand_max_len", "8",
    ]
    with pytest.raises(SystemExit):
        train_main(argv)
