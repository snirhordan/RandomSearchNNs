"""Tests for the Phase-2 geometric pairwise attention bias.

Covers ``models.seq_encoder.GeometricAttentionBias`` and its integration into
``TransformerSeqLayer.forward`` / ``RSNN_TRSF_Reg``:

- forward/backward finite + grads flow through the bias MLP;
- E(3): SE(3) (rotation+translation) invariance of the bias and model output;
- reflection flips only the dihedral sin channels (chirality), matching
  ``utils.search._dihedral_basis``;
- padding/boundary handling: no NaN, pad pairs and missing-neighbor channels
  are zeroed;
- vectorized ``_basis`` matches a scalar reference built from the trusted
  ``utils.search`` helpers;
- adjacent ``(i, i+1)`` pair features equal the consecutive ``sample_dfs``
  angle/dihedral features;
- param-count budget (geom_bias=False adds 0 params; the ON delta = MLP size);
- backward-compat: geom_bias=False model is bit-identical to a model built
  without the new kwargs;
- optional CUDA parity.

Run::

    source /home/snirhordan/miniconda3/etc/profile.d/conda.sh && conda activate rwnn
    cd /home/snirhordan/ito/RandomSearchNNs
    python -m pytest tests/test_geom_bias.py -v --tb=short
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.seq_encoder import GeometricAttentionBias, TransformerSeqLayer
from quickstart.train_qm9 import RSNN_TRSF_Reg
from utils.search import (
    sample_dfs,
    _bond_angle,
    _dihedral,
    _angle_basis,
    _dihedral_basis,
)


HID_DIM = 16
PE_OUT_DIM = 8
D_MODEL = HID_DIM + PE_OUT_DIM   # 24
NHEAD = 4                        # head_dim = 6
OUT_DIM = 1
REDUCE = "mean"
RBF_K = 16
ANGLE_K = 8
DIHEDRAL_K = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_bias(nhead=NHEAD, **kw):
    torch.manual_seed(0)
    return GeometricAttentionBias(nhead, rbf_K=RBF_K, angle_K=ANGLE_K,
                                  dihedral_K=DIHEDRAL_K, **kw)


def _make_walk_xyz(BN, L, seed=0):
    torch.manual_seed(seed)
    return torch.randn(BN, L, 3)


def _random_rotation(seed=0):
    """A proper rotation (det +1) via QR of a random matrix."""
    torch.manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3))
    q = q * torch.sign(torch.diagonal(r))          # fix sign ambiguity
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]                          # force det +1
    return q


def _make_chain_data(pos_xyz):
    """Build a Data with an N-atom chain (edges 0-1-2-...-N-1)."""
    N = pos_xyz.shape[0]
    edges = []
    for i in range(N - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    edge_index = torch.tensor(edges, dtype=torch.long).T
    x = torch.zeros((N, 1), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data.pos = pos_xyz.float()
    data.x_emb = torch.zeros(N, dtype=torch.long)
    return data


def _synthetic_batch_xyz(pe_in_dim=4, seed=0, L=10):
    """Two graphs x 3 walks, padded length L, vocab size 6 (PAD = 5), w/ xyz."""

    class B:
        pass

    torch.manual_seed(seed)
    b = B()
    b.lengths = torch.tensor([L, 7, 5, L, 3, 8])
    b.walk_emb = torch.randint(0, 5, (6, L))
    b.walk_pe = torch.randn(6, L, pe_in_dim)
    b.walk_xyz = torch.randn(6, L, 3)
    wi = torch.full((2, 3, L), -1, dtype=torch.long)
    for g in range(2):
        for s in range(3):
            Ls = int(b.lengths[g * 3 + s])
            wi[g, s, :Ls] = torch.randint(0, 4, (Ls,))
    b.walk_ids = wi
    for i in range(6):
        Li = int(b.lengths[i])
        b.walk_emb[i, Li:] = 5
        b.walk_xyz[i, Li:] = 0.0          # zero padding, matches sample_dfs
    return b


# ---------------------------------------------------------------------------
# 1. forward / backward finite + grads through the bias MLP
# ---------------------------------------------------------------------------
def test_bias_forward_shape_finite():
    mod = _make_bias()
    BN, L = 2, 7
    walk_xyz = _make_walk_xyz(BN, L)
    pad_mask = torch.zeros(BN, L, dtype=torch.bool)
    pad_mask[1, 5:] = True
    bias = mod(walk_xyz, pad_mask)
    assert bias.shape == (BN, NHEAD, L, L)
    assert torch.isfinite(bias).all()


def test_model_forward_backward_with_bias():
    batch = _synthetic_batch_xyz(pe_in_dim=4)
    model = RSNN_TRSF_Reg(4, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6, "sum",
                          nhead=NHEAD, attn_mode="full", pos_enc="rope",
                          geom_bias=True, geom_rbf_K=RBF_K,
                          geom_angle_K=ANGLE_K, geom_dihedral_K=DIHEDRAL_K)
    out = model(batch)
    assert out.shape == (2, OUT_DIM)
    assert torch.isfinite(out).all()
    out.sum().backward()
    bias_params = list(model.geom_bias_mod.parameters())
    assert len(bias_params) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in bias_params)


# ---------------------------------------------------------------------------
# 2. SE(3) invariance (rotation + translation)
# ---------------------------------------------------------------------------
def test_bias_se3_invariance():
    mod = _make_bias().eval()
    BN, L = 2, 8
    walk_xyz = _make_walk_xyz(BN, L)
    pad_mask = torch.zeros(BN, L, dtype=torch.bool)
    pad_mask[0, 6:] = True
    R = _random_rotation(seed=1)
    t = torch.randn(3)
    moved = walk_xyz @ R.T + t
    with torch.no_grad():
        b0 = mod(walk_xyz, pad_mask)
        b1 = mod(moved, pad_mask)
    assert torch.allclose(b0, b1, atol=1e-5), "bias not SE(3)-invariant"


def test_model_output_se3_invariance():
    R = _random_rotation(seed=2)
    t = torch.randn(3)
    model = RSNN_TRSF_Reg(4, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6, "sum",
                          nhead=NHEAD, attn_mode="full", pos_enc="none",
                          geom_bias=True, geom_rbf_K=RBF_K,
                          geom_angle_K=ANGLE_K,
                          geom_dihedral_K=DIHEDRAL_K).eval()
    batch = _synthetic_batch_xyz(pe_in_dim=4, seed=5)
    moved = _synthetic_batch_xyz(pe_in_dim=4, seed=5)
    real = ~( torch.arange(moved.walk_xyz.shape[1])[None, :]
              >= moved.lengths[:, None] )
    rotated = moved.walk_xyz @ R.T + t
    moved.walk_xyz = torch.where(real[..., None], rotated,
                                 torch.zeros_like(rotated))
    with torch.no_grad():
        o0 = model(batch)
        o1 = model(moved)
    assert torch.allclose(o0, o1, atol=1e-5), "model output not SE(3)-invariant"


# ---------------------------------------------------------------------------
# 3. Reflection flips only the dihedral sin channels (chirality)
# ---------------------------------------------------------------------------
def test_reflection_flips_dihedral_sin_only():
    mod = _make_bias().eval()
    BN, L = 1, 9
    walk_xyz = _make_walk_xyz(BN, L, seed=3)
    pad_mask = torch.zeros(BN, L, dtype=torch.bool)
    refl = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])  # det -1
    reflected = walk_xyz @ refl.T
    with torch.no_grad():
        f0 = mod._basis(walk_xyz, pad_mask)
        f1 = mod._basis(reflected, pad_mask)
    # channel layout: [rbf(K), angle_i(K), angle_j(K), dih_sin(K), dih_cos(K)]
    n_inv = RBF_K + 2 * ANGLE_K
    inv0, inv1 = f0[..., :n_inv], f1[..., :n_inv]
    sin0 = f0[..., n_inv:n_inv + DIHEDRAL_K]
    sin1 = f1[..., n_inv:n_inv + DIHEDRAL_K]
    cos0 = f0[..., n_inv + DIHEDRAL_K:]
    cos1 = f1[..., n_inv + DIHEDRAL_K:]
    assert torch.allclose(inv0, inv1, atol=1e-5), "distance/angles not invariant"
    assert torch.allclose(cos0, cos1, atol=1e-5), "dihedral cos not invariant"
    assert torch.allclose(sin0, -sin1, atol=1e-5), "dihedral sin must flip sign"
    # and there is genuine chirality signal somewhere (non-trivial dihedral)
    assert sin0.abs().max() > 1e-3


# ---------------------------------------------------------------------------
# 4. Padding / boundary: no NaN, masked channels are zero
# ---------------------------------------------------------------------------
def test_boundary_and_padding_zeroing():
    mod = _make_bias().eval()
    BN, L = 1, 6
    walk_xyz = _make_walk_xyz(BN, L, seed=4)
    pad_mask = torch.zeros(BN, L, dtype=torch.bool)
    pad_mask[0, 4:] = True                     # real length 4
    with torch.no_grad():
        f = mod._basis(walk_xyz, pad_mask)
    assert torch.isfinite(f).all()
    n_inv = RBF_K + 2 * ANGLE_K
    angle_i = f[..., RBF_K:RBF_K + ANGLE_K]
    angle_j = f[..., RBF_K + ANGLE_K:n_inv]
    dih = f[..., n_inv:]
    rbf = f[..., :RBF_K]
    # position i=0 has no i-1 -> angle_i (and dih) row 0 must be zero
    assert torch.allclose(angle_i[0, 0, :, :], torch.zeros_like(angle_i[0, 0]))
    assert torch.allclose(dih[0, 0, :, :], torch.zeros_like(dih[0, 0]))
    # last real position j=3 has no j+1 -> angle_j (and dih) col 3 must be zero
    assert torch.allclose(angle_j[0, :, 3, :], torch.zeros_like(angle_j[0, :, 3]))
    assert torch.allclose(dih[0, :, 3, :], torch.zeros_like(dih[0, :, 3]))
    # any pair touching a padded position -> all channels (incl. rbf) zero
    assert torch.allclose(rbf[0, 4:, :, :], torch.zeros_like(rbf[0, 4:, :, :]))
    assert torch.allclose(rbf[0, :, 4:, :], torch.zeros_like(rbf[0, :, 4:, :]))


def test_coord_gradient_finite():
    """d(bias)/d(walk_xyz) must be finite even though every grid has degenerate
    cells (the i==j diagonal and j==i+-1 adjacency make acos hit +/-1 and the
    dihedral atan2 hit (0, 0)). Guards against the latent NaN that would surface
    if coordinates are ever made differentiable (e.g. force prediction)."""
    mod = _make_bias()
    BN, L = 2, 7
    # (a) fully-real walk: exercises every degenerate diagonal/adjacency cell.
    xyz = _make_walk_xyz(BN, L, seed=12).requires_grad_(True)
    pad = torch.zeros(BN, L, dtype=torch.bool)
    mod(xyz, pad).sum().backward()
    assert xyz.grad is not None and torch.isfinite(xyz.grad).all()
    # (b) padded walk (zeroed padding, matching collation): padded coords get a
    # finite (zero) gradient through the masks.
    xyz2 = _make_walk_xyz(BN, L, seed=13)
    xyz2[1, 5:] = 0.0
    xyz2.requires_grad_(True)
    pad2 = torch.zeros(BN, L, dtype=torch.bool)
    pad2[1, 5:] = True
    mod(xyz2, pad2).sum().backward()
    assert torch.isfinite(xyz2.grad).all()


def test_padded_layer_output_finite_with_bias():
    """Bias must compose with the eye-guard: fully-padded rows stay finite."""
    layer = TransformerSeqLayer(D_MODEL, NHEAD, attn_mode="causal",
                                pos_enc="rope").eval()
    mod = _make_bias().eval()
    BN, L = 2, 6
    x = torch.randn(BN, L, D_MODEL)
    walk_xyz = _make_walk_xyz(BN, L, seed=6)
    pad_mask = torch.zeros(BN, L, dtype=torch.bool)
    pad_mask[1, 3:] = True
    walk_xyz[1, 3:] = 0.0
    with torch.no_grad():
        bias = mod(walk_xyz, pad_mask)
        y = layer(x, pad_mask=pad_mask, attn_bias=bias)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# 5. Vectorized _basis vs scalar reference (trusted utils.search helpers)
# ---------------------------------------------------------------------------
def _scalar_reference_basis(walk_xyz, pad_mask, angle_K, dihedral_K,
                            rbf_centers, rbf_width):
    """Triple Python loop using the scalar helpers, with the same boundary
    zeroing the module applies. Returns (BN, L, L, in_dim)."""
    BN, L, _ = walk_xyz.shape
    valid = ~pad_mask
    in_dim = rbf_centers.numel() + 2 * angle_K + 2 * dihedral_K
    out = torch.zeros(BN, L, L, in_dim)
    K = rbf_centers.numel()
    for n in range(BN):
        length = int(valid[n].sum())
        xs = walk_xyz[n]
        for i in range(L):
            for j in range(L):
                feat = torch.zeros(in_dim)
                if valid[n, i] and valid[n, j]:
                    d = (xs[i] - xs[j]).norm()
                    feat[:K] = torch.exp(-((d - rbf_centers) ** 2)
                                         / (2 * rbf_width ** 2))
                # angle_i needs i-1, i, j real and i != j; angle_j needs
                # i, j, j+1 real and i != j; the dihedral needs all four and
                # i != j (self-pairs are masked off the diagonal).
                vi = (valid[n, i] and i >= 1 and valid[n, i - 1]
                      and valid[n, j] and i != j)
                vj = (valid[n, j] and j + 1 < L and valid[n, j + 1]
                      and valid[n, i] and i != j)
                vd = (valid[n, i] and i >= 1 and valid[n, i - 1]
                      and valid[n, j] and j + 1 < L and valid[n, j + 1]
                      and i != j)
                if vi:
                    th_i = _bond_angle(xs[i - 1], xs[i], xs[j])
                    feat[K:K + angle_K] = _angle_basis(th_i, angle_K)
                if vj:
                    th_j = _bond_angle(xs[i], xs[j], xs[j + 1])
                    feat[K + angle_K:K + 2 * angle_K] = \
                        _angle_basis(th_j, angle_K)
                if vd:
                    phi = _dihedral(xs[i - 1], xs[i], xs[j], xs[j + 1])
                    feat[K + 2 * angle_K:] = _dihedral_basis(phi, dihedral_K)
                out[n, i, j] = feat
    return out


def test_vectorized_matches_scalar_reference():
    mod = _make_bias().eval()
    BN, L = 2, 7
    walk_xyz = _make_walk_xyz(BN, L, seed=8)
    pad_mask = torch.zeros(BN, L, dtype=torch.bool)
    pad_mask[0, 5:] = True
    walk_xyz[0, 5:] = 0.0
    with torch.no_grad():
        f_vec = mod._basis(walk_xyz, pad_mask)
    f_ref = _scalar_reference_basis(walk_xyz, pad_mask, ANGLE_K, DIHEDRAL_K,
                                    mod.rbf_centers, mod.rbf_width)
    assert torch.allclose(f_vec, f_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Adjacent (i, i+1) pair features == consecutive sample_dfs features
# ---------------------------------------------------------------------------
def test_adjacent_pair_reduces_to_consecutive_features():
    # Deterministic canonical chain so the DFS order is fixed and matches
    # walk-position == atom index.
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
    ])
    data = _make_chain_data(pos)
    nw, w, max_len = 1, 2, pos.shape[0]
    vocab = {"PAD": 0}
    data = sample_dfs(data, nw, w, max_len, vocab, angles=True, dihedrals=True,
                      angle_K=ANGLE_K, dihedral_K=DIHEDRAL_K,
                      canonical=True, emit_xyz=True)
    L = int(data.lengths[0])
    walk_xyz = data.walk_xyz[:, :L, :]           # (1, L, 3)
    pad_mask = torch.zeros(1, L, dtype=torch.bool)

    mod = _make_bias().eval()
    with torch.no_grad():
        feats = mod._basis(walk_xyz, pad_mask)   # (1, L, L, in_dim)

    n_inv = RBF_K + 2 * ANGLE_K
    angle_i = feats[..., RBF_K:RBF_K + ANGLE_K]
    angle_j = feats[..., RBF_K + ANGLE_K:n_inv]
    dih = feats[..., n_inv:]

    # consecutive features from sample_dfs walk_pe layout:
    #   [edge(w) | angle(angle_K) | dihedral(2*dihedral_K)]
    wpe = data.walk_pe[0]                        # (L, edge + aK + 2dK)
    cons_angle = wpe[:, w:w + ANGLE_K]
    cons_dih = wpe[:, w + ANGLE_K:]

    for i in range(L - 1):
        j = i + 1
        # vertex-i angle of pair (i, i+1) == consecutive angle at step i+1
        if i >= 1:
            assert torch.allclose(angle_i[0, i, j], cons_angle[i + 1], atol=1e-5)
        # vertex-j angle of pair (i, i+1) == consecutive angle at step i+2
        if j + 1 < L:
            assert torch.allclose(angle_j[0, i, j], cons_angle[i + 2], atol=1e-5)
        # dihedral of pair (i, i+1) == consecutive dihedral at step i+2
        if i >= 1 and j + 1 < L:
            assert torch.allclose(dih[0, i, j], cons_dih[i + 2], atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Param-count budget
# ---------------------------------------------------------------------------
def test_param_count_budget():
    def mk(geom):
        torch.manual_seed(0)
        return RSNN_TRSF_Reg(4, 16, 128, OUT_DIM, 2, 6, "mean",
                             nhead=8, ffn_mult=4, attn_mode="full",
                             pos_enc="sinusoidal", geom_bias=geom,
                             geom_rbf_K=16, geom_angle_K=8, geom_dihedral_K=4,
                             geom_hidden=32)
    n_off = sum(p.numel() for p in mk(False).parameters())
    n_on = sum(p.numel() for p in mk(True).parameters())
    # analytic MLP size: in_dim = 16 + 2*8 + 2*4 = 40
    in_dim = 16 + 2 * 8 + 2 * 4
    expected = (in_dim * 32 + 32) + (32 * 8 + 8)   # Linear1 + Linear2
    assert expected == 1576
    assert n_on - n_off == expected
    # geom_bias=False must add exactly 0 params (checkpoint compatibility)
    torch.manual_seed(0)
    base = RSNN_TRSF_Reg(4, 16, 128, OUT_DIM, 2, 6, "mean", nhead=8,
                         ffn_mult=4, attn_mode="full", pos_enc="sinusoidal")
    assert sum(p.numel() for p in base.parameters()) == n_off


# ---------------------------------------------------------------------------
# 8. Backward-compat: geom_bias=False bit-identical to no-kwargs model
# ---------------------------------------------------------------------------
def test_backward_compat_geom_bias_off():
    batch = _synthetic_batch_xyz(pe_in_dim=4, seed=11)
    torch.manual_seed(123)
    m_kw = RSNN_TRSF_Reg(4, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6, "sum",
                         nhead=NHEAD, attn_mode="full", pos_enc="rope",
                         geom_bias=False).eval()
    torch.manual_seed(123)
    m_nokw = RSNN_TRSF_Reg(4, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6, "sum",
                           nhead=NHEAD, attn_mode="full",
                           pos_enc="rope").eval()
    assert m_kw.geom_bias_mod is None
    with torch.no_grad():
        o_kw = m_kw(batch)
        o_nokw = m_nokw(batch)
    assert torch.equal(o_kw, o_nokw)


def test_layer_no_attn_bias_matches_prior_path():
    """TransformerSeqLayer.forward with attn_bias=None == pre-Phase-2 path."""
    torch.manual_seed(7)
    layer = TransformerSeqLayer(D_MODEL, NHEAD, attn_mode="causal",
                                pos_enc="rope").eval()
    x = torch.randn(2, 6, D_MODEL)
    pad_mask = torch.zeros(2, 6, dtype=torch.bool)
    pad_mask[1, 3:] = True
    with torch.no_grad():
        y_default = layer(x, pad_mask=pad_mask)
        y_none = layer(x, pad_mask=pad_mask, attn_bias=None)
    assert torch.equal(y_default, y_none)


# ---------------------------------------------------------------------------
# 9. CUDA parity (skip if unavailable)
# ---------------------------------------------------------------------------
def test_bias_cuda_parity(device):
    if device.type != "cuda":
        pytest.skip("no CUDA available")
    mod = _make_bias().eval()
    BN, L = 2, 7
    walk_xyz = _make_walk_xyz(BN, L, seed=9)
    pad_mask = torch.zeros(BN, L, dtype=torch.bool)
    pad_mask[0, 5:] = True
    walk_xyz[0, 5:] = 0.0
    with torch.no_grad():
        b_cpu = mod(walk_xyz, pad_mask)
        mod_gpu = mod.to(device)
        b_gpu = mod_gpu(walk_xyz.to(device), pad_mask.to(device))
    assert torch.isfinite(b_gpu).all()
    assert torch.allclose(b_cpu, b_gpu.cpu(), atol=1e-4)
