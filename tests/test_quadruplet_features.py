"""Unit tests for the quadruplet (bond angle + dihedral) feature path in
``utils/search.py::sample_dfs``.

Covers:
- Numerical correctness on hand-constructed geometries (right angle,
  planar trans dihedral, gauche dihedral, eclipsed dihedral).
- The Fourier basis values (cos(l*theta), sin(l*phi), cos(l*phi)) for those
  angles match closed-form values to float tolerance.
- ``max_search_len`` truncates the DFS at the requested cap.
- ``pe_in_dim`` agrees with the actual ``walk_pe.shape[-1]``.

Run::

    source /home/snirhordan/miniconda3/etc/profile.d/conda.sh && conda activate rwnn
    cd /home/snirhordan/ito/RandomSearchNNs
    python -m pytest tests/test_quadruplet_features.py -v --tb=short
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

# Repo root on sys.path so ``utils`` and ``quickstart`` import cleanly under
# pytest (matches the convention used in test_d_rwnn_invariance.py).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.search import (  # noqa: E402
    sample_dfs,
    _bond_angle,
    _dihedral,
    _angle_basis,
    _dihedral_basis,
)
from quickstart.train_qm9 import compute_pe_in_dim  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build a tiny synthetic geometric graph with known coordinates.
# ---------------------------------------------------------------------------


def _make_chain_data(pos_xyz, vocab_size=4):
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
    data.x_emb = torch.zeros(N, dtype=torch.long)  # all atoms = type 0
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bond_angle_right_angle():
    """Three atoms at (1,0,0)-(0,0,0)-(0,1,0) form a 90-degree angle."""
    p_prev2 = torch.tensor([1.0, 0.0, 0.0])
    p_prev1 = torch.tensor([0.0, 0.0, 0.0])
    p_curr = torch.tensor([0.0, 1.0, 0.0])
    theta = _bond_angle(p_prev2, p_prev1, p_curr)
    assert math.isclose(float(theta), math.pi / 2, abs_tol=1e-6)


def test_bond_angle_straight():
    """Three colinear atoms form a 180-degree angle (no kink)."""
    p_prev2 = torch.tensor([-1.0, 0.0, 0.0])
    p_prev1 = torch.tensor([0.0, 0.0, 0.0])
    p_curr = torch.tensor([1.0, 0.0, 0.0])
    theta = _bond_angle(p_prev2, p_prev1, p_curr)
    assert math.isclose(float(theta), math.pi, abs_tol=1e-5)


def test_dihedral_trans_planar():
    """Planar trans (anti) configuration of four atoms in the xy-plane = ±pi."""
    p0 = torch.tensor([0.0, 1.0, 0.0])
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])
    p3 = torch.tensor([1.0, -1.0, 0.0])
    phi = _dihedral(p0, p1, p2, p3)
    assert math.isclose(abs(float(phi)), math.pi, abs_tol=1e-5)


def test_dihedral_cis_planar():
    """Planar cis (syn) configuration of four atoms in the xy-plane = 0."""
    p0 = torch.tensor([0.0, 1.0, 0.0])
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])
    p3 = torch.tensor([1.0, 1.0, 0.0])
    phi = _dihedral(p0, p1, p2, p3)
    assert math.isclose(float(phi), 0.0, abs_tol=1e-5)


def test_dihedral_gauche():
    """Gauche-like configuration: 90 degrees out of the xy-plane = ±pi/2."""
    p0 = torch.tensor([0.0, 1.0, 0.0])
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])
    p3 = torch.tensor([1.0, 0.0, 1.0])
    phi = _dihedral(p0, p1, p2, p3)
    assert math.isclose(abs(float(phi)), math.pi / 2, abs_tol=1e-5)


def test_angle_basis_values():
    """cos(l * pi/2) for l=1..K matches closed-form."""
    K = 8
    theta = torch.tensor(math.pi / 2)
    basis = _angle_basis(theta, K)
    expected = torch.tensor([math.cos(l * math.pi / 2) for l in range(1, K + 1)])
    assert torch.allclose(basis, expected, atol=1e-6)


def test_dihedral_basis_values_pi():
    """sin(l*pi)=0 and cos(l*pi)=(-1)^l for the trans configuration."""
    K = 4
    phi = torch.tensor(math.pi)
    basis = _dihedral_basis(phi, K)
    # Layout: [sin l*phi for l=1..K] then [cos l*phi for l=1..K]
    sin_part = basis[:K]
    cos_part = basis[K:]
    assert torch.allclose(sin_part, torch.zeros(K), atol=1e-6)
    expected_cos = torch.tensor([(-1.0) ** l for l in range(1, K + 1)])
    assert torch.allclose(cos_part, expected_cos, atol=1e-6)


def test_dihedral_basis_sign_under_reflection():
    """Reflecting the molecule flips dihedral sign; sin l*phi flips, cos l*phi stays."""
    K = 4
    phi = torch.tensor(math.pi / 3)
    b_pos = _dihedral_basis(phi, K)
    b_neg = _dihedral_basis(-phi, K)
    sin_pos, cos_pos = b_pos[:K], b_pos[K:]
    sin_neg, cos_neg = b_neg[:K], b_neg[K:]
    assert torch.allclose(sin_neg, -sin_pos, atol=1e-6)
    assert torch.allclose(cos_neg, cos_pos, atol=1e-6)


def test_sample_dfs_max_search_len_caps_length():
    """When max_search_len < n_atoms, the DFS terminates at max_search_len."""
    # 6-atom linear chain
    pos = torch.zeros((6, 3))
    pos[:, 0] = torch.arange(6, dtype=torch.float)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    # Without cap: DFS visits all 6 atoms (chain is connected)
    d_full = sample_dfs(data.clone(), nw=1, s=4, max_len=10, vocab=vocab)
    assert int(d_full.lengths[0]) == 6
    # With cap=3: DFS terminates at 3 atoms
    d_capped = sample_dfs(data.clone(), nw=1, s=4, max_len=10, vocab=vocab,
                          max_search_len=3)
    assert int(d_capped.lengths[0]) == 3


def test_sample_dfs_angles_pe_in_dim_consistent():
    """walk_pe.shape[-1] matches compute_pe_in_dim's prediction (with angles)."""
    pos = torch.zeros((6, 3))
    pos[:, 0] = torch.arange(6, dtype=torch.float)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    s = 4
    angle_K = 8
    dihedral_K = 4
    d = sample_dfs(data.clone(), nw=2, s=s, max_len=10, vocab=vocab,
                   add_edge_feat=None, angles=True, dihedrals=True,
                   angle_K=angle_K, dihedral_K=dihedral_K)
    # No add_edge_feat path: walk_pe = [encoding_edge(s), angle(angle_K),
    # dihedral(2*dihedral_K)] = s + 8 + 8 = 20
    expected = s + angle_K + 2 * dihedral_K
    assert d.walk_pe.shape[-1] == expected
    # Cross-check against compute_pe_in_dim for the same config:
    # pe_in_dim for search = s + 0 (no distances/mol_edge) + angle_K + 2*dihedral_K
    pred = compute_pe_in_dim(walk_type="search", w=s, distances=0,
                             mol_edge_feat=0, rbf_K=16,
                             angles=1, dihedrals=1,
                             angle_K=angle_K, dihedral_K=dihedral_K)
    assert pred == expected


def test_sample_dfs_zero_padding_at_short_positions():
    """For pos < 2, angle is undefined and features should be zero.
    For pos < 3, dihedral is undefined and features should be zero.
    """
    pos = torch.zeros((6, 3))
    pos[:, 0] = torch.arange(6, dtype=torch.float)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    s = 4
    angle_K = 4
    dihedral_K = 2
    d = sample_dfs(data.clone(), nw=1, s=s, max_len=10, vocab=vocab,
                   add_edge_feat=None, angles=True, dihedrals=True,
                   angle_K=angle_K, dihedral_K=dihedral_K)
    # walk_pe layout: [encoding_edge(s) | angle(angle_K) | dihedral(2*dihedral_K)]
    angle_slice = d.walk_pe[0, :, s : s + angle_K]
    dihedral_slice = d.walk_pe[0, :, s + angle_K :]
    # positions 0 and 1 have no valid bond angle (need pos >= 2)
    assert torch.allclose(angle_slice[0], torch.zeros(angle_K))
    assert torch.allclose(angle_slice[1], torch.zeros(angle_K))
    # positions 0, 1, 2 have no valid dihedral (need pos >= 3)
    assert torch.allclose(dihedral_slice[0], torch.zeros(2 * dihedral_K))
    assert torch.allclose(dihedral_slice[1], torch.zeros(2 * dihedral_K))
    assert torch.allclose(dihedral_slice[2], torch.zeros(2 * dihedral_K))


def test_sample_dfs_no_angle_flags_matches_baseline():
    """With angles=False dihedrals=False, behaviour is byte-identical to the
    original sample_dfs (backward-compat guarantee for existing checkpoints).
    """
    pos = torch.zeros((6, 3))
    pos[:, 0] = torch.arange(6, dtype=torch.float)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    import random
    random.seed(42)
    torch.manual_seed(42)
    d_baseline = sample_dfs(data.clone(), nw=2, s=4, max_len=10, vocab=vocab)
    random.seed(42)
    torch.manual_seed(42)
    d_new = sample_dfs(data.clone(), nw=2, s=4, max_len=10, vocab=vocab,
                       angles=False, dihedrals=False)
    assert torch.equal(d_baseline.walk_pe, d_new.walk_pe)
    assert torch.equal(d_baseline.walk_emb, d_new.walk_emb)
    assert torch.equal(d_baseline.lengths, d_new.lengths)


# ---------------------------------------------------------------------------
# E(3) invariance/equivariance tests on the quadruplet features.
# ---------------------------------------------------------------------------


def _random_rotation(seed: int = 0):
    """Random rotation matrix in SO(3) via QR of a Gaussian matrix."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    # Make a proper rotation (det +1)
    q = q * torch.sign(torch.diag(r))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_bond_angle_rotation_translation_invariant():
    """theta(R p_prev2 + t, R p_prev1 + t, R p_curr + t) == theta(...)."""
    p0 = torch.tensor([1.0, 0.0, 0.0])
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    R = _random_rotation(0)
    t = torch.tensor([5.0, -3.0, 2.0])
    theta = _bond_angle(p0, p1, p2)
    theta_t = _bond_angle(R @ p0 + t, R @ p1 + t, R @ p2 + t)
    assert torch.isclose(theta, theta_t, atol=1e-5)


def test_bond_angle_reflection_invariant():
    """Bond angle is invariant under reflection (it's the angle at the apex)."""
    p0 = torch.tensor([1.0, 0.0, 0.0])
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 1.0, 0.0])
    M = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))  # reflect x
    theta = _bond_angle(p0, p1, p2)
    theta_r = _bond_angle(M @ p0, M @ p1, M @ p2)
    assert torch.isclose(theta, theta_r, atol=1e-5)


def test_dihedral_rotation_translation_invariant():
    """phi is invariant under proper rotation + translation."""
    p0 = torch.tensor([0.0, 1.0, 0.0])
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])
    p3 = torch.tensor([1.0, 0.0, 1.0])
    R = _random_rotation(1)
    t = torch.tensor([2.0, 5.0, -1.0])
    phi = _dihedral(p0, p1, p2, p3)
    phi_t = _dihedral(R @ p0 + t, R @ p1 + t, R @ p2 + t, R @ p3 + t)
    assert torch.isclose(phi, phi_t, atol=1e-5)


def test_dihedral_reflection_flips_sign():
    """phi flips sign under improper reflection: phi(M·) == -phi(·).
    (This is exactly the chirality we want sin(l*phi) features to capture.)
    """
    p0 = torch.tensor([0.0, 1.0, 0.0])
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])
    p3 = torch.tensor([1.0, 0.0, 1.0])
    M = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))  # reflect x
    phi = _dihedral(p0, p1, p2, p3)
    phi_r = _dihedral(M @ p0, M @ p1, M @ p2, M @ p3)
    assert torch.isclose(phi, -phi_r, atol=1e-5)
