"""Tests for the protein-ligand sampler (``utils.search.sample_protein_ligand``).

Synthetic complex: 2 protein residues (residue 0: atoms 0, 1, 2; residue 1:
atoms 3, 4) as segment 0, plus 4 ligand atoms (5, 6, 7, 8) as segment 1. The
ligand forms a fully-connected star (branchy) subgraph within 4.5 A, and
only protein atom 2 sits within 4.5 A of the ligand -- a clean, single
interface contact with a strict interface-distance ranking among protein
atoms (2 < 1 < 0 < 3 < 4, no ties), which lets the truncation test assert
an exact expected kept set.

Run::

    source /home/snirhordan/miniconda3/etc/profile.d/conda.sh && conda activate rwnn
    cd /home/snirhordan/ito/RandomSearchNNs
    python -m pytest tests/test_protein_ligand_sampler.py -v --tb=short
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.search import sample_protein_ligand  # noqa: E402
from quickstart.train_qm9 import compute_pe_in_dim  # noqa: E402


CUTOFF = 4.5
VOCAB = {'PAD': 100}
PROT_IDX = [0, 1, 2, 3, 4]
LIG_IDX = [5, 6, 7, 8]


def _radius_graph_edge_index(pos, cutoff=CUTOFF):
    """Undirected radius-graph edge_index (both (i, j) and (j, i))."""
    n = pos.shape[0]
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if torch.norm(pos[i] - pos[j]).item() <= cutoff:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def _complex_pos():
    """9-atom synthetic protein-ligand complex coordinates.

    Protein prefix atoms 0-4; ligand atoms 5-8 (star around atom 5, all
    pairwise ligand distances < CUTOFF so DFS start/branch order varies).
    Only atom 2 is within CUTOFF of any ligand atom; interface-distance
    ranking among protein atoms is strictly 2 < 1 < 0 < 3 < 4.
    """
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


def _make_complex():
    pos = _complex_pos()
    n = pos.shape[0]
    edge_index = _radius_graph_edge_index(pos)
    segment = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    residue_id = torch.tensor([0, 0, 0, 1, 1, -1, -1, -1, -1], dtype=torch.long)
    x_emb = torch.arange(n, dtype=torch.long)
    data = Data(edge_index=edge_index, pos=pos, num_nodes=n)
    data.x_emb = x_emb
    data.segment = segment
    data.residue_id = residue_id
    return data


def _random_rotation(seed=0):
    """Random proper rotation matrix in SO(3) via QR of a Gaussian matrix."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# --- sanity check on the fixture itself (not part of the contract, but
# catches a broken synthetic complex before it silently breaks every test).
def test_fixture_topology_sanity():
    data = _make_complex()
    edges = set(map(tuple, data.edge_index.t().tolist()))
    # Only atom 2 (protein) touches the ligand.
    for p in [0, 1, 3, 4]:
        for l in LIG_IDX:
            assert (p, l) not in edges and (l, p) not in edges
    assert (2, 5) in edges and (5, 2) in edges
    # Ligand subgraph is fully connected (a walk from any start covers all).
    for a in LIG_IDX:
        for b in LIG_IDX:
            if a != b:
                assert (a, b) in edges


# ---------------------------------------------------------------------------
# 1. Shapes
# ---------------------------------------------------------------------------
def test_shapes():
    data = _make_complex()
    m, s, max_len = 4, 2, 20
    out = sample_protein_ligand(data, m, s, max_len, VOCAB)
    assert out.walk_emb.shape == (m, max_len)
    assert out.walk_ids.shape == (1, m, max_len)
    assert out.walk_pe.shape == (m, max_len, s)
    assert out.lengths.shape == (m,)


# ---------------------------------------------------------------------------
# 2. Deterministic prefix (identical across walks and across repeated,
#    re-seeded calls).
# ---------------------------------------------------------------------------
def test_deterministic_prefix_across_walks_and_calls():
    m, s, max_len = 6, 2, 20
    expected_prefix = torch.tensor(PROT_IDX, dtype=torch.long)

    random.seed(7)
    out1 = sample_protein_ligand(_make_complex(), m, s, max_len, VOCAB)

    random.seed(7)
    out2 = sample_protein_ligand(_make_complex(), m, s, max_len, VOCAB)

    for i in range(m):
        assert torch.equal(out1.walk_ids[0, i, :5], expected_prefix)
        assert torch.equal(out2.walk_ids[0, i, :5], expected_prefix)
    assert torch.equal(out1.walk_ids, out2.walk_ids)


# ---------------------------------------------------------------------------
# 3. Segment separation: prefix region only segment 0, ligand suffix only
#    segment 1, no overlap.
# ---------------------------------------------------------------------------
def test_prefix_and_ligand_segments_disjoint():
    data = _make_complex()
    m, s, max_len = 5, 2, 20
    out = sample_protein_ligand(data, m, s, max_len, VOCAB)
    for i in range(m):
        L = int(out.lengths[i])
        ids = out.walk_ids[0, i, :L].tolist()
        prefix_part = ids[:5]
        ligand_part = ids[5:]
        assert all(v in PROT_IDX for v in prefix_part)
        assert all(v in LIG_IDX for v in ligand_part)
        assert len(set(prefix_part) & set(ligand_part)) == 0


# ---------------------------------------------------------------------------
# 4. Ensemble variability: ligand suffixes are not all identical across walks.
# ---------------------------------------------------------------------------
def test_ligand_suffixes_vary_across_walks():
    random.seed(42)
    data = _make_complex()
    m, s, max_len = 8, 2, 20
    out = sample_protein_ligand(data, m, s, max_len, VOCAB)
    suffixes = []
    for i in range(m):
        L = int(out.lengths[i])
        suffixes.append(tuple(out.walk_ids[0, i, 5:L].tolist()))
    assert len(set(suffixes)) > 1


# ---------------------------------------------------------------------------
# 5. walk_pe width matches compute_pe_in_dim("search", ...).
# ---------------------------------------------------------------------------
def test_pe_width_matches_compute_pe_in_dim_no_flags():
    data = _make_complex()
    m, s, max_len = 3, 2, 20
    out = sample_protein_ligand(data, m, s, max_len, VOCAB)
    expected = compute_pe_in_dim("search", w=s, distances=0, mol_edge_feat=0)
    assert out.walk_pe.shape[-1] == expected == s


def test_pe_width_matches_compute_pe_in_dim_with_angles_dihedrals():
    data = _make_complex()
    m, s, max_len = 3, 2, 20
    angle_K, dihedral_K = 6, 3
    out = sample_protein_ligand(data, m, s, max_len, VOCAB,
                                angles=True, dihedrals=True,
                                angle_K=angle_K, dihedral_K=dihedral_K)
    expected = compute_pe_in_dim("search", w=s, distances=0, mol_edge_feat=0,
                                 angles=1, dihedrals=1,
                                 angle_K=angle_K, dihedral_K=dihedral_K)
    assert out.walk_pe.shape[-1] == expected
    assert expected == s + angle_K + 2 * dihedral_K


def test_pe_width_matches_compute_pe_in_dim_with_edge_feat():
    data = _make_complex()
    m, s, max_len = 3, 2, 20
    n = data.num_nodes
    d_edge = 5
    add_edge_feat = torch.rand(n, n, d_edge)
    out = sample_protein_ligand(data, m, s, max_len, VOCAB,
                                add_edge_feat=add_edge_feat)
    # compute_pe_in_dim's "distances" bonus term (rbf_K * distances) is the
    # generic width contribution of an RBF-expanded per-edge feature stream;
    # setting rbf_K=d_edge, distances=1 reproduces the width of an arbitrary
    # add_edge_feat stream of that dimensionality (mol_edge_feat=0 to avoid
    # double-counting the unrelated 3-channel bond-feature bonus).
    expected = compute_pe_in_dim("search", w=s, distances=1, mol_edge_feat=0,
                                 rbf_K=d_edge)
    assert out.walk_pe.shape[-1] == expected == s + d_edge


# ---------------------------------------------------------------------------
# 6. angle/dihedral: zero below threshold and at padding; not all zero at
#    some valid position at/above threshold.
# ---------------------------------------------------------------------------
def test_angle_dihedral_zero_below_threshold_and_padding():
    random.seed(3)
    data = _make_complex()
    m, s, max_len = 1, 2, 20
    angle_K, dihedral_K = 6, 3
    out = sample_protein_ligand(data, m, s, max_len, VOCAB,
                                angles=True, dihedrals=True,
                                angle_K=angle_K, dihedral_K=dihedral_K)
    L = int(out.lengths[0])
    angle_slice = out.walk_pe[0, :, s: s + angle_K]
    dihedral_slice = out.walk_pe[0, :, s + angle_K:]

    # Below threshold: exactly zero.
    assert torch.allclose(angle_slice[0], torch.zeros(angle_K))
    assert torch.allclose(angle_slice[1], torch.zeros(angle_K))
    assert torch.allclose(dihedral_slice[0], torch.zeros(2 * dihedral_K))
    assert torch.allclose(dihedral_slice[1], torch.zeros(2 * dihedral_K))
    assert torch.allclose(dihedral_slice[2], torch.zeros(2 * dihedral_K))

    # Padding: zero beyond the realized length.
    assert torch.allclose(angle_slice[L:], torch.zeros(max_len - L, angle_K))
    assert torch.allclose(dihedral_slice[L:],
                          torch.zeros(max_len - L, 2 * dihedral_K))

    # At/above threshold, within the realized length: not all zero somewhere.
    assert any(
        not torch.allclose(angle_slice[p], torch.zeros(angle_K))
        for p in range(2, L)
    )
    assert any(
        not torch.allclose(dihedral_slice[p], torch.zeros(2 * dihedral_K))
        for p in range(3, L)
    )


# ---------------------------------------------------------------------------
# 7. Padding contract: walk_emb == PAD, walk_ids == -1, walk_pe == 0 beyond
#    the realized length.
# ---------------------------------------------------------------------------
def test_padding_contract():
    data = _make_complex()
    m, s, max_len = 4, 2, 20
    out = sample_protein_ligand(data, m, s, max_len, VOCAB)
    for i in range(m):
        L = int(out.lengths[i])
        assert torch.all(out.walk_emb[i, L:] == VOCAB['PAD'])
        assert torch.all(out.walk_ids[0, i, L:] == -1)
        assert torch.allclose(out.walk_pe[i, L:], torch.zeros(max_len - L, s))


# ---------------------------------------------------------------------------
# 8. Truncation policy: ligand fully retained; kept prefix atoms are the
#    interface-nearest ones, re-sorted by (residue_id, atom_index).
# ---------------------------------------------------------------------------
def test_truncation_keeps_ligand_and_interface_nearest_prefix():
    random.seed(11)
    data = _make_complex()
    m, s, max_len = 4, 2, 6   # prefix(5) + ligand(4) cannot fit in 6.
    out = sample_protein_ligand(data, m, s, max_len, VOCAB)
    for i in range(m):
        L = int(out.lengths[i])
        ids = out.walk_ids[0, i, :L].tolist()
        assert L == 6
        lig_part = [v for v in ids if v in LIG_IDX]
        prot_part = [v for v in ids if v in PROT_IDX]
        # Ligand fully retained: it is a single connected component of 4
        # atoms, so a DFS from any start covers all 4 -- never truncated.
        assert set(lig_part) == set(LIG_IDX)
        assert len(lig_part) == 4
        # Interface-nearest protein atoms retained: prefix_cap = 6 - 4 = 2,
        # and the strict interface-distance ranking is 2 < 1 < 0 < 3 < 4,
        # so the kept set must be exactly {1, 2}.
        assert set(prot_part) == {1, 2}
        # Kept atoms remain sorted by (residue_id, atom_index): both are
        # residue 0, so index order 1 < 2 must be preserved, at the front.
        assert ids[:2] == [1, 2]


# ---------------------------------------------------------------------------
# 9. SE(3) invariance for a fixed walk order: same RNG draws + identical
#    (invariant) graph topology => identical order; angle/dihedral values
#    invariant under rotation + translation up to float tolerance.
# ---------------------------------------------------------------------------
def test_se3_invariance_of_encoding_for_fixed_order():
    data1 = _make_complex()

    R = _random_rotation(5)
    t = torch.tensor([5.0, -2.0, 3.0])
    pos2 = data1.pos @ R.T + t
    edge_index2 = _radius_graph_edge_index(pos2)

    # Rotation+translation preserves pairwise distances exactly (up to fp
    # noise), so the radius-graph edge set must be identical.
    e1 = set(map(tuple, data1.edge_index.t().tolist()))
    e2 = set(map(tuple, edge_index2.t().tolist()))
    assert e1 == e2

    data2 = Data(edge_index=edge_index2, pos=pos2, num_nodes=data1.num_nodes)
    data2.x_emb = data1.x_emb.clone()
    data2.segment = data1.segment.clone()
    data2.residue_id = data1.residue_id.clone()

    # max_len large enough that no truncation occurs, so the kept prefix
    # never depends on the (fp-sensitive) interface-distance argsort.
    m, s, max_len = 3, 2, 20
    angle_K, dihedral_K = 6, 3

    random.seed(99)
    out1 = sample_protein_ligand(data1, m, s, max_len, VOCAB,
                                 angles=True, dihedrals=True,
                                 angle_K=angle_K, dihedral_K=dihedral_K)
    random.seed(99)
    out2 = sample_protein_ligand(data2, m, s, max_len, VOCAB,
                                 angles=True, dihedrals=True,
                                 angle_K=angle_K, dihedral_K=dihedral_K)

    # Same graph topology + same RNG draws -> byte-identical walk order.
    assert torch.equal(out1.walk_ids, out2.walk_ids)
    assert torch.equal(out1.lengths, out2.lengths)

    # encoding_edge depends only on (invariant) graph adjacency -> exact.
    assert torch.equal(out1.walk_pe[..., :s], out2.walk_pe[..., :s])
    # angle/dihedral depend on raw coordinates -> invariant up to fp noise.
    assert torch.allclose(out1.walk_pe[..., s:], out2.walk_pe[..., s:],
                          atol=1e-4)
