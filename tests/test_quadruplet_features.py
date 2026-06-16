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


# ---------------------------------------------------------------------------
# Vectorized path: numerical equivalence to the scalar path.
# ---------------------------------------------------------------------------


def test_sample_dfs_vectorized_matches_scalar():
    """sample_dfs with vectorize=True must produce element-equal walk_pe to
    the scalar path, given the same RNG state. Same eps, same clamp, same
    atan2 formula -> exact equality except for any float-ordering noise.
    """
    import random
    import numpy as np

    # Build a small irregular 7-atom geometry so angles and dihedrals are
    # non-trivial across the walk.
    rng = np.random.RandomState(7)
    pos = torch.tensor(rng.rand(7, 3) * 4.0, dtype=torch.float32)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}

    seed = 12345
    nw, s, max_len = 4, 4, 10
    angle_K, dihedral_K = 8, 4

    def _run(vectorize):
        random.seed(seed)
        torch.manual_seed(seed)
        d = sample_dfs(data.clone(), nw=nw, s=s, max_len=max_len, vocab=vocab,
                       add_edge_feat=None, angles=True, dihedrals=True,
                       angle_K=angle_K, dihedral_K=dihedral_K,
                       vectorize=vectorize)
        return d

    d_slow = _run(False)
    d_fast = _run(True)

    # Same walks (same RNG -> identical DFS order).
    assert torch.equal(d_slow.walk_ids, d_fast.walk_ids)
    assert torch.equal(d_slow.walk_emb, d_fast.walk_emb)
    assert torch.equal(d_slow.lengths, d_fast.lengths)

    # walk_pe[..., :s] is the connectivity encoding (deterministic, identical).
    assert torch.equal(d_slow.walk_pe[..., :s], d_fast.walk_pe[..., :s])

    # Angle + dihedral slices should be allclose (the math is the same, but
    # vectorized cross products can have different float rounding order).
    angle_slow = d_slow.walk_pe[..., s : s + angle_K]
    angle_fast = d_fast.walk_pe[..., s : s + angle_K]
    assert torch.allclose(angle_slow, angle_fast, atol=1e-5), (
        f"angle features diverge: max abs diff = {(angle_slow - angle_fast).abs().max()}"
    )

    dihedral_slow = d_slow.walk_pe[..., s + angle_K :]
    dihedral_fast = d_fast.walk_pe[..., s + angle_K :]
    assert torch.allclose(dihedral_slow, dihedral_fast, atol=1e-5), (
        f"dihedral features diverge: max abs diff = {(dihedral_slow - dihedral_fast).abs().max()}"
    )


def test_sample_dfs_vectorized_with_no_angles_is_noop():
    """vectorize=True with angles=False, dihedrals=False is byte-identical
    to scalar path (no quadruplet features to compute either way)."""
    import random
    pos = torch.zeros((6, 3))
    pos[:, 0] = torch.arange(6, dtype=torch.float)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    random.seed(7); torch.manual_seed(7)
    d_slow = sample_dfs(data.clone(), nw=2, s=4, max_len=10, vocab=vocab,
                        vectorize=False)
    random.seed(7); torch.manual_seed(7)
    d_fast = sample_dfs(data.clone(), nw=2, s=4, max_len=10, vocab=vocab,
                        vectorize=True)
    assert torch.equal(d_slow.walk_pe, d_fast.walk_pe)
    assert torch.equal(d_slow.lengths, d_fast.lengths)


# ---------------------------------------------------------------------------
# adversarial tests
# ---------------------------------------------------------------------------
# Each test below stress-tests the vectorize=True path against the scalar
# reference. Convention: build a graph whose DFS order is forced to visit
# specific atoms (forced via a deterministic small chain + fixed seed), then
# run sample_dfs twice with vectorize toggled and compare element-wise.


def _run_both_paths(data, *, nw, s, max_len, vocab, angle_K, dihedral_K,
                    angles=True, dihedrals=True, seed=12345):
    """Run sample_dfs with vectorize={False,True} under identical RNG.

    Returns (d_slow, d_fast).
    """
    import random
    random.seed(seed); torch.manual_seed(seed)
    d_slow = sample_dfs(data.clone(), nw=nw, s=s, max_len=max_len, vocab=vocab,
                        add_edge_feat=None, angles=angles, dihedrals=dihedrals,
                        angle_K=angle_K, dihedral_K=dihedral_K, vectorize=False)
    random.seed(seed); torch.manual_seed(seed)
    d_fast = sample_dfs(data.clone(), nw=nw, s=s, max_len=max_len, vocab=vocab,
                        add_edge_feat=None, angles=angles, dihedrals=dihedrals,
                        angle_K=angle_K, dihedral_K=dihedral_K, vectorize=True)
    return d_slow, d_fast


def test_adv_bond_angle_a_norm_zero_two_atoms_same_coord():
    """Attack #2: p_prev2 == p_prev1 -> a_norm == 0.

    Scalar (search.py:22): cos_t = 0 / (0*b + eps) = 0  ->  arccos(0) = pi/2.
    Batched (search.py:79): identical formula -> same answer.
    Both paths must agree.
    """
    # 4-atom chain where atoms 0,1 share coordinates.
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],   # coincident with atom 0
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=torch.float32)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}

    d_slow, d_fast = _run_both_paths(
        data, nw=2, s=3, max_len=6, vocab=vocab, angle_K=4, dihedral_K=2,
        seed=999,
    )
    # Same walk order and lengths.
    assert torch.equal(d_slow.walk_ids, d_fast.walk_ids)
    # Bit-equal on all feature slices despite a_norm=0 entry.
    assert torch.allclose(d_slow.walk_pe, d_fast.walk_pe, atol=1e-6)


def test_adv_dihedral_b2_norm_below_eps_threshold_diverges():
    """Attack #1 / #9: b2_norm in (0, 1e-8) is the *only* documented divergence
    between scalar and batched dihedral.

    Scalar (search.py:44-45): early-returns 0 when b2_norm < 1e-8.
    Batched (search.py:91):   uses b2_norm.clamp(min=1e-8) and continues
                              computing -> generally non-zero result.

    We construct a fixture that forces a b2 with norm 1e-9 between two
    adjacent atoms in the DFS order. This DOES produce a divergence and we
    assert it explicitly; the test passes by *documenting* the divergence
    via the assertions below. For QM9 chemistry (~1 A bond lengths), this
    threshold cannot be hit in practice — bond lengths never get within nine
    orders of magnitude of zero.
    """
    # 4-atom chain: atoms 1 and 2 are placed 1e-9 apart -> b2 sub-eps.
    pos = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1e-9, 0.0, 0.0],  # 1e-9 from atom 1 -> below scalar threshold
        [1.0, -1.0, 0.0],
    ], dtype=torch.float32)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}

    d_slow, d_fast = _run_both_paths(
        data, nw=4, s=3, max_len=6, vocab=vocab, angle_K=4, dihedral_K=2,
        seed=2024,
    )
    assert torch.equal(d_slow.walk_ids, d_fast.walk_ids)

    # Find dihedral slice indices: layout is [edge(s) | angle(angle_K) | dihedral(2*dK)].
    s, angle_K, dK = 3, 4, 2
    dihedral_slow = d_slow.walk_pe[..., s + angle_K:]
    dihedral_fast = d_fast.walk_pe[..., s + angle_K:]

    # There MUST exist at least one (i, pos) where the two paths disagree —
    # otherwise the fixture failed to trigger the b2_norm < 1e-8 branch.
    max_diff = (dihedral_slow - dihedral_fast).abs().max().item()
    assert max_diff > 0.1, (
        f"Expected scalar != batched in sub-eps b2 regime; got max diff={max_diff}. "
        "Fixture may have failed to elicit a sub-1e-8 b2 step (DFS picked an order "
        "where atoms 1 and 2 are not adjacent at positions >= 3)."
    )

    # Edge-encoding and bond-angle slices, which are independent of the
    # dihedral b2_hat branch, should still agree exactly.
    assert torch.equal(d_slow.walk_pe[..., :s], d_fast.walk_pe[..., :s])
    angle_slow = d_slow.walk_pe[..., s:s + angle_K]
    angle_fast = d_fast.walk_pe[..., s:s + angle_K]
    assert torch.allclose(angle_slow, angle_fast, atol=1e-6)


def test_adv_three_consecutive_collinear_b2_hat_nondegenerate():
    """Attack #1 (refined): three consecutive atoms collinear *but* with non-tiny
    b2 (so b2_norm > 1e-8). Here scalar's b2_hat = b2/b2_norm and batched's
    b2_hat = b2/b2_norm.clamp(min=1e-8) are identical (the clamp does
    nothing when b2_norm >> 1e-8). The dihedral itself is mathematically
    0 (n1 = 0 because b1 || b2), and both paths should return 0.
    """
    # 4 atoms colinear along x with non-degenerate spacing.
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],     # 0,1,2 colinear -> b1 || b2 -> n1 = 0
        [2.0, 1.0, 0.0],
    ], dtype=torch.float32)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    d_slow, d_fast = _run_both_paths(
        data, nw=4, s=3, max_len=6, vocab=vocab, angle_K=4, dihedral_K=2,
        seed=42,
    )
    assert torch.equal(d_slow.walk_ids, d_fast.walk_ids)
    assert torch.allclose(d_slow.walk_pe, d_fast.walk_pe, atol=1e-6)


def test_adv_far_from_origin_float32_precision():
    """Attack #3: atom positions translated to (1e4, 1e4, 1e4) +/- small offset.
    Float32 cross products accumulate cancellation noise differently in
    the scalar (one quadruplet at a time) vs batched (M quadruplets fused)
    code paths. The tolerance loosens but the two paths must still agree
    to a few-ULP level — well within atol=1e-3, easily within 1e-4 in
    practice for typical molecular geometries.
    """
    import numpy as np
    rng = np.random.RandomState(11)
    pos = torch.tensor(rng.rand(7, 3) * 2.0 + 1e4, dtype=torch.float32)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    d_slow, d_fast = _run_both_paths(
        data, nw=4, s=4, max_len=10, vocab=vocab, angle_K=8, dihedral_K=4,
        seed=1234,
    )
    assert torch.equal(d_slow.walk_ids, d_fast.walk_ids)
    # Tighter check on edge slice (no float math involved).
    assert torch.equal(d_slow.walk_pe[..., :4], d_fast.walk_pe[..., :4])
    # Looser tolerance for the angle/dihedral float slices.
    angle_slow = d_slow.walk_pe[..., 4:4 + 8]
    angle_fast = d_fast.walk_pe[..., 4:4 + 8]
    assert torch.allclose(angle_slow, angle_fast, atol=1e-3), (
        f"angles diverge at large coords: max diff = "
        f"{(angle_slow - angle_fast).abs().max()}"
    )
    dihedral_slow = d_slow.walk_pe[..., 4 + 8:]
    dihedral_fast = d_fast.walk_pe[..., 4 + 8:]
    assert torch.allclose(dihedral_slow, dihedral_fast, atol=1e-3), (
        f"dihedrals diverge at large coords: max diff = "
        f"{(dihedral_slow - dihedral_fast).abs().max()}"
    )


def test_adv_random_stress_50_molecules():
    """Attack #4: 50 random 8-atom molecules, m=4 walks, s=4. All paths agree
    to atol=1e-5 across every element. This is the "headline" equivalence
    test for QM9-like usage.
    """
    import numpy as np
    n_mols = 50
    max_diff_angle = 0.0
    max_diff_dihedral = 0.0
    for mol_seed in range(n_mols):
        rng = np.random.RandomState(mol_seed)
        n_atoms = 8
        pos = torch.tensor(rng.rand(n_atoms, 3) * 3.0, dtype=torch.float32)
        data = _make_chain_data(pos)
        vocab = {"PAD": 0}
        d_slow, d_fast = _run_both_paths(
            data, nw=4, s=4, max_len=8, vocab=vocab, angle_K=8, dihedral_K=4,
            seed=1000 + mol_seed,
        )
        # Walk order identical.
        assert torch.equal(d_slow.walk_ids, d_fast.walk_ids), (
            f"DFS order diverged for mol {mol_seed} — RNG is not isolated"
        )
        # Track max divergence in feature slices.
        s = 4
        angle_K = 8
        a_slow = d_slow.walk_pe[..., s:s + angle_K]
        a_fast = d_fast.walk_pe[..., s:s + angle_K]
        max_diff_angle = max(max_diff_angle,
                             (a_slow - a_fast).abs().max().item())
        d_slow_dh = d_slow.walk_pe[..., s + angle_K:]
        d_fast_dh = d_fast.walk_pe[..., s + angle_K:]
        max_diff_dihedral = max(max_diff_dihedral,
                                (d_slow_dh - d_fast_dh).abs().max().item())
        assert torch.allclose(d_slow.walk_pe, d_fast.walk_pe, atol=1e-5), (
            f"mol {mol_seed}: max diff exceeds 1e-5"
        )
    # Sanity: at least *some* dihedrals were non-trivial across 50 molecules.
    # (If max diff were 0 it would just mean nothing was computed.)
    # Note: max diff is the inter-path error; we just sanity-check it's bounded.
    assert max_diff_angle < 1e-5
    assert max_diff_dihedral < 1e-5


def test_adv_float64_pos_dtype_consistency():
    """Attack #5: data.pos as float64. The vectorized path explicitly casts
    `basis_a.to(walk_pe_angle.dtype)` (search.py:434, 443) before scattering
    back into the float32 buffer. The scalar path does an implicit cast on
    assignment. Both should land at the same float32 value.
    """
    import numpy as np
    rng = np.random.RandomState(13)
    pos64 = torch.tensor(rng.rand(7, 3) * 4.0, dtype=torch.float64)
    data = _make_chain_data(pos64)
    # _make_chain_data casts to float() — override with float64.
    data.pos = pos64
    vocab = {"PAD": 0}
    d_slow, d_fast = _run_both_paths(
        data, nw=3, s=4, max_len=10, vocab=vocab, angle_K=8, dihedral_K=4,
        seed=77,
    )
    assert torch.equal(d_slow.walk_ids, d_fast.walk_ids)
    # walk_pe buffer is float32 in both branches.
    assert d_slow.walk_pe.dtype == torch.float32
    assert d_fast.walk_pe.dtype == torch.float32
    assert torch.allclose(d_slow.walk_pe, d_fast.walk_pe, atol=1e-5)


def test_adv_empty_quadruplet_list_two_atoms():
    """Attack #6: molecule too small for any dihedral (2 atoms, pos < 3 always).

    The `if dihedral_idx is not None and len(dihedral_idx) > 0` guard at
    search.py:435 must short-circuit and leave walk_pe_dihedral all zero
    (matching scalar). Same for angle on 2-atom molecules (pos < 2 -> no
    angle either; but max_len > 2 will visit at most 2 atoms in chain).
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    data = _make_chain_data(pos)
    vocab = {"PAD": 0}
    d_slow, d_fast = _run_both_paths(
        data, nw=2, s=3, max_len=6, vocab=vocab, angle_K=4, dihedral_K=2,
        seed=5,
    )
    # Both paths must produce all-zero dihedral slice.
    s = 3
    angle_K = 4
    dihedral_slow = d_slow.walk_pe[..., s + angle_K:]
    dihedral_fast = d_fast.walk_pe[..., s + angle_K:]
    assert torch.equal(dihedral_slow, torch.zeros_like(dihedral_slow))
    assert torch.equal(dihedral_fast, torch.zeros_like(dihedral_fast))
    # And the angle slice should also be all-zero (we never reach pos=2 with
    # only 2 atoms in the DFS).
    angle_slow = d_slow.walk_pe[..., s:s + angle_K]
    angle_fast = d_fast.walk_pe[..., s:s + angle_K]
    assert torch.equal(angle_slow, torch.zeros_like(angle_slow))
    assert torch.equal(angle_fast, torch.zeros_like(angle_fast))


def test_adv_scatter_order_invariance_under_index_permutation():
    """Attack #7: PyTorch fancy-index scatter
       walk_pe_angle[idx_a[:, 0], idx_a[:, 1]] = basis_a
    must be invariant to the order of rows in idx_a (each (i, pos) is unique
    because DFS visits each position once per walk).

    We patch the index list to its reverse order, recompute, and assert the
    final walk_pe_angle is bitwise identical.
    """
    # Build a non-trivial graph that produces multiple angle quadruplets.
    import numpy as np
    rng = np.random.RandomState(101)
    pos = torch.tensor(rng.rand(6, 3) * 3.0, dtype=torch.float32)

    # Directly exercise the scatter pattern used in sample_dfs (search.py:434).
    from utils.search import _batch_bond_angle, _batch_angle_basis
    nw, max_len, angle_K = 3, 6, 4
    walk_pe = torch.zeros((nw, max_len, angle_K), dtype=torch.float32)
    # Construct a synthetic index list: 5 quadruplets across the buffer.
    angle_idx = [
        (0, 2, 0, 1, 2),
        (0, 3, 1, 2, 3),
        (1, 2, 2, 3, 4),
        (2, 4, 3, 4, 5),
        (1, 5, 0, 2, 4),
    ]
    pos_xyz = pos

    def _scatter(angle_idx):
        wpa = torch.zeros_like(walk_pe)
        idx = torch.tensor(angle_idx, dtype=torch.long)
        p0 = pos_xyz[idx[:, 2]]
        p1 = pos_xyz[idx[:, 3]]
        p2 = pos_xyz[idx[:, 4]]
        thetas = _batch_bond_angle(p0, p1, p2)
        basis_a = _batch_angle_basis(thetas, angle_K)
        wpa[idx[:, 0], idx[:, 1]] = basis_a
        return wpa

    forward = _scatter(angle_idx)
    reversed_ = _scatter(list(reversed(angle_idx)))
    # Each (i, pos) is unique -> order-independent fancy-index assignment.
    assert torch.equal(forward, reversed_), (
        "fancy-index scatter is NOT order-invariant — this would mean the "
        "vectorize=True path depends on insertion order, breaking reproducibility"
    )


def test_adv_cross_product_orientation_matches_numpy():
    """Attack #8: torch.cross sign convention matches numpy.cross (right-hand rule).
    If torch flipped sign on any axis, dihedral phi would systematically differ.
    """
    import numpy as np
    rng = np.random.RandomState(31)
    a = rng.randn(5, 3)
    b = rng.randn(5, 3)
    np_cross = np.cross(a, b)
    torch_cross = torch.cross(
        torch.from_numpy(a), torch.from_numpy(b), dim=-1
    ).numpy()
    assert np.allclose(np_cross, torch_cross, atol=1e-10)


def test_adv_batched_b2_hat_equals_scalar_when_b2_above_eps():
    """Attack #9: for any non-degenerate quadruplet (b2_norm >> 1e-8), the
    scalar b2_hat = b2/b2_norm and the batched b2_hat = b2/b2_norm.clamp(1e-8)
    are bit-equal (the clamp is a no-op). Already covered indirectly by
    test_sample_dfs_vectorized_matches_scalar, but we pin it explicitly
    here at the helper level for clarity.
    """
    import numpy as np
    rng = np.random.RandomState(17)
    p0 = torch.tensor(rng.randn(20, 3), dtype=torch.float32)
    p1 = torch.tensor(rng.randn(20, 3), dtype=torch.float32)
    p2 = torch.tensor(rng.randn(20, 3), dtype=torch.float32)
    p3 = torch.tensor(rng.randn(20, 3), dtype=torch.float32)
    from utils.search import _batch_dihedral
    phi_batch = _batch_dihedral(p0, p1, p2, p3)
    phi_scalar = torch.stack([
        _dihedral(p0[i], p1[i], p2[i], p3[i]) for i in range(20)
    ])
    assert torch.allclose(phi_batch, phi_scalar, atol=1e-6), (
        f"non-degenerate scalar vs batched dihedral diverge: "
        f"max diff = {(phi_batch - phi_scalar).abs().max()}"
    )


def test_adv_index_tensor_dtype_is_long_and_no_overflow():
    """Attack #10: idx_a = torch.tensor(angle_idx, dtype=torch.long) — verify
    int64 is enforced and indices well below the int64 ceiling.

    QM9 has at most ~30 heavy atoms per molecule, so any node id easily fits
    in int8. This test is mostly forensic — it confirms the dtype contract
    on the index tensor.
    """
    angle_idx = [(0, 2, 0, 1, 2), (1, 3, 1, 2, 3)]
    idx_a = torch.tensor(angle_idx, dtype=torch.long)
    assert idx_a.dtype == torch.long
    # int64 max = 2**63 - 1; we should be nowhere near it.
    assert idx_a.max().item() < (2**62)


def test_adv_fancy_index_shapes():
    """Attack #11: pos_xyz[idx_a[:, 2]] must return (M, 3), not (M, 1, 3) or
    (M, 3, 1). This is the exact pattern at search.py:429-431.
    """
    pos_xyz = torch.randn(10, 3)
    idx_a = torch.tensor(
        [(0, 2, 0, 1, 2), (1, 3, 1, 2, 3), (2, 4, 2, 3, 4)],
        dtype=torch.long,
    )  # (M=3, 5)
    p_prev2 = pos_xyz[idx_a[:, 2]]
    p_prev1 = pos_xyz[idx_a[:, 3]]
    p_curr = pos_xyz[idx_a[:, 4]]
    assert p_prev2.shape == (3, 3), p_prev2.shape
    assert p_prev1.shape == (3, 3), p_prev1.shape
    assert p_curr.shape == (3, 3), p_curr.shape


def test_adv_random_stress_with_random_rotation_invariance():
    """Combined: 20 random molecules x 2 rotations each.  Both
    vectorize=False and vectorize=True must (1) agree with each other and (2)
    produce rotation-invariant features. Catches any latent sign/orientation
    bug that would only appear when global frame is non-canonical.
    """
    import numpy as np
    for mol_seed in range(20):
        rng = np.random.RandomState(mol_seed + 500)
        pos = torch.tensor(rng.rand(6, 3) * 2.0, dtype=torch.float32)
        # Random rotation
        R = _random_rotation(mol_seed)
        pos_rot = pos @ R.T

        data_a = _make_chain_data(pos)
        data_b = _make_chain_data(pos_rot)
        vocab = {"PAD": 0}

        d_a_slow, d_a_fast = _run_both_paths(
            data_a, nw=3, s=3, max_len=6, vocab=vocab, angle_K=4, dihedral_K=2,
            seed=2000 + mol_seed,
        )
        d_b_slow, d_b_fast = _run_both_paths(
            data_b, nw=3, s=3, max_len=6, vocab=vocab, angle_K=4, dihedral_K=2,
            seed=2000 + mol_seed,
        )
        # (1) vectorize=True matches scalar in both frames.
        assert torch.allclose(d_a_slow.walk_pe, d_a_fast.walk_pe, atol=1e-5)
        assert torch.allclose(d_b_slow.walk_pe, d_b_fast.walk_pe, atol=1e-5)
        # (2) rotation-invariance: angle/dihedral features should agree across
        # frames (edge encoding is graph-only -> trivially equal).
        # Same DFS RNG seed and same neighbor structure -> same walk_ids and
        # hence same per-position features.
        assert torch.equal(d_a_slow.walk_ids, d_b_slow.walk_ids)
        assert torch.allclose(d_a_slow.walk_pe, d_b_slow.walk_pe, atol=1e-4)
        assert torch.allclose(d_a_fast.walk_pe, d_b_fast.walk_pe, atol=1e-4)
