"""Tests for the PROTEIN TORSION ORDER ablation (additive, default-off).

Covers:
  (a) build_protein_torsion_order chi-path construction (linear Lys,
      branched Tyr first-per-level tiebreak, Gly len2, Ala len3).
  (b) build_protein_torsion_order fragment splitting (bonded vs gapped
      C-N), a hand-verified spine dihedral, and segment discontinuity at
      the fragment gap (no quadruplet silently spans it).
  (c) sample_protein_ligand(protein_order='torsion') segment gating:
      angle/dihedral/edge zero at segment starts, nonzero within a long
      enough segment; a spine dihedral value cross-checked against an
      independently implemented dihedral formula.
  (d) REGRESSION: protein_order='allatom' (default AND explicit) is
      byte-identical to a no-kwarg call, same seed.
  (e) integration: build_data + build_protein_torsion_order -> sample
      (protein_order='torsion') -> RSNN_TRSF_Reg forward is finite.

Run::

    source /home/snirhordan/miniconda3/etc/profile.d/conda.sh && conda activate rwnn
    cd /home/snirhordan/ito/RandomSearchNNs
    python -m pytest tests/test_protein_torsion_order.py -v --tb=short
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.pdbbind import (  # noqa: E402
    build_data,
    build_pdbbind_vocab,
    build_protein_torsion_order,
)
from utils.search import sample_protein_ligand, _dihedral  # noqa: E402
from quickstart.train_qm9 import RSNN_TRSF_Reg, compute_pe_in_dim  # noqa: E402


# ---------------------------------------------------------------------------
# Independent dihedral formula (praxeolitic form) -- deliberately a
# DIFFERENT derivation path than utils.search._dihedral's n1xn2/atan2
# formula, used as a cross-check in (b)/(c). Verified by hand against
# utils.search._dihedral on random quadruplets to fix the sign convention
# (this formula's raw sign is opposite _dihedral's; negated below).
# ---------------------------------------------------------------------------
def _independent_dihedral(p0, p1, p2, p3):
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return -float(np.arctan2(y, x))


# ---------------------------------------------------------------------------
# (a) chi-path construction: linear Lys, branched Tyr, Gly, Ala.
# ---------------------------------------------------------------------------

def _single_residue_atoms(element, greek):
    n = len(element)
    global_idx = list(range(n))
    residue_id = [0] * n
    coords = np.zeros((n, 3))
    return global_idx, element, greek, residue_id, coords


def test_chi_path_linear_lys():
    # N, CA, C, O, CB, CG, CD, CE, NZ (real Lys atom order/greek codes).
    element = ['N', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'N']
    greek = ['', 'A', '', '', 'B', 'G', 'D', 'E', 'Z']
    order, seg = build_protein_torsion_order(*_single_residue_atoms(element, greek))

    N, CA, C, O, CB, CG, CD, CE, NZ = range(9)
    assert set(order) == set(range(9))  # coverage: every atom appears

    spine_seg = seg[order.index(N)]
    spine_positions = [i for i, s in enumerate(seg) if s == spine_seg]
    assert [order[i] for i in spine_positions] == [N, CA, C]

    # chi-path segment: find the segment containing CB (a side-chain atom).
    chi_seg = seg[order.index(CB)]
    chi_positions = [i for i, s in enumerate(seg) if s == chi_seg]
    chi_atoms = [order[i] for i in chi_positions]
    assert chi_atoms == [N, CA, CB, CG, CD, CE, NZ]

    # chi1..chi4 are exactly the consecutive-4 windows of the chi-path.
    assert chi_atoms[0:4] == [N, CA, CB, CG]      # chi1
    assert chi_atoms[1:5] == [CA, CB, CG, CD]     # chi2
    assert chi_atoms[2:6] == [CB, CG, CD, CE]     # chi3
    assert chi_atoms[3:7] == [CG, CD, CE, NZ]     # chi4

    tail_seg = seg[order.index(O)]
    tail_atoms = [order[i] for i, s in enumerate(seg) if s == tail_seg]
    assert tail_atoms == [O]


def test_chi_path_branched_tyr_picks_first_per_level():
    # N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ, OH -- CD/CE each appear
    # TWICE (branch); chi-path must pick the FIRST occurrence at each level.
    element = ['N', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'O']
    greek = ['', 'A', '', '', 'B', 'G', 'D', 'D', 'E', 'E', 'Z', 'H']
    order, seg = build_protein_torsion_order(*_single_residue_atoms(element, greek))

    N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ, OH = range(12)
    assert set(order) == set(range(12))

    chi_seg = seg[order.index(CB)]
    chi_atoms = [order[i] for i, s in enumerate(seg) if s == chi_seg]
    assert chi_atoms == [N, CA, CB, CG, CD1, CE1, CZ, OH]  # first-per-level

    tail_seg = seg[order.index(CD2)]
    tail_atoms = [order[i] for i, s in enumerate(seg) if s == tail_seg]
    assert set(tail_atoms) == {O, CD2, CE2}  # duplicates + O fall to tail


def test_chi_path_gly_len2():
    element = ['N', 'C', 'C', 'O']
    greek = ['', 'A', '', '']
    order, seg = build_protein_torsion_order(*_single_residue_atoms(element, greek))
    N, CA, C, O = range(4)
    chi_seg = seg[order.index(N)]
    # N's segment could be spine (len3) or chi (len2); pick chi via CA membership check
    # -- easier: enumerate all segments and find the length-2 one containing N,CA only.
    from collections import defaultdict
    blocks = defaultdict(list)
    for pos, s in enumerate(seg):
        blocks[s].append(order[pos])
    chi_blocks = [b for b in blocks.values() if b == [N, CA]]
    assert len(chi_blocks) == 1
    tail_blocks = [b for b in blocks.values() if b == [O]]
    assert len(tail_blocks) == 1


def test_chi_path_ala_len3():
    element = ['N', 'C', 'C', 'O', 'C']
    greek = ['', 'A', '', '', 'B']
    order, seg = build_protein_torsion_order(*_single_residue_atoms(element, greek))
    N, CA, C, O, CB = range(5)
    from collections import defaultdict
    blocks = defaultdict(list)
    for pos, s in enumerate(seg):
        blocks[s].append(order[pos])
    chi_blocks = [b for b in blocks.values() if b == [N, CA, CB]]
    assert len(chi_blocks) == 1
    tail_blocks = [b for b in blocks.values() if b == [O]]
    assert len(tail_blocks) == 1


# ---------------------------------------------------------------------------
# (b) fragment splitting: 3 Gly-like residues, res0-res1 bonded, res1-res2
#     gapped; a spine quadruplet matches an independent dihedral; no
#     quadruplet spans the gap (segment discontinuity at the boundary).
# ---------------------------------------------------------------------------

def _three_residue_backbone():
    """3 Gly-like residues (N, CA, C, O each); res0-C to res1-N ~1.06A
    (bonded, < 1.8A cutoff); res1-C to res2-N = 10A (gapped)."""
    N0, CA0, C0 = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 1.0, 0.0)
    O0 = (1.5, 2.0, 0.0)
    N1 = (2.5, 1.2, 0.3)  # dist(C0, N1) ~= 1.063 -- bonded
    CA1, C1 = (3.3, 2.0, 0.6), (4.3, 2.3, 1.0)
    O1 = (4.3, 3.3, 1.0)
    N2 = (14.3, 2.3, 1.0)  # dist(C1, N2) == 10.0 -- gapped
    CA2, C2 = (15.3, 2.3, 1.0), (16.3, 2.3, 1.0)
    O2 = (16.3, 3.3, 1.0)

    coords = np.array([N0, CA0, C0, O0, N1, CA1, C1, O1, N2, CA2, C2, O2])
    element = ['N', 'C', 'C', 'O'] * 3
    greek = ['', 'A', '', ''] * 3
    residue_id = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    global_idx = list(range(12))
    return global_idx, element, greek, residue_id, coords, coords


def test_fragment_split_and_spine_dihedral_no_gap_quadruplet():
    global_idx, element, greek, residue_id, coords, coords_arr = _three_residue_backbone()
    order, seg = build_protein_torsion_order(global_idx, element, greek, residue_id, coords)

    N0, CA0, C0, O0, N1, CA1, C1, O1, N2, CA2, C2, O2 = range(12)
    assert set(order) == set(range(12))

    # --- 2 fragments: {res0, res1} bonded, {res2} alone. ---
    from collections import defaultdict
    blocks = defaultdict(list)
    for pos, s in enumerate(seg):
        blocks[s].append(order[pos])
    spine_blocks = [b for b in blocks.values() if len(b) >= 3 and b[0] in (N0, N2)]
    frag01 = [b for b in spine_blocks if b[0] == N0][0]
    frag2 = [b for b in spine_blocks if b[0] == N2][0]
    assert frag01 == [N0, CA0, C0, N1, CA1, C1]
    assert frag2 == [N2, CA2, C2]

    # --- a spine quadruplet inside fragment0 (N0,CA0,C0,N1 == psi(res0))
    # matches an independent dihedral computation. ---
    expected = _independent_dihedral(coords_arr[N0], coords_arr[CA0],
                                     coords_arr[C0], coords_arr[N1])
    got = _dihedral(torch.tensor(coords_arr[N0]), torch.tensor(coords_arr[CA0]),
                    torch.tensor(coords_arr[C0]), torch.tensor(coords_arr[N1])).item()
    assert got == pytest.approx(expected, abs=1e-6)

    # --- no quadruplet spans the fragment gap: the 4 raw `order` positions
    # that WOULD form (N1,CA1,C1,N2) if segments were ignored have
    # NON-UNIFORM seg (the boundary is caught by the sampler's gating). ---
    # The raw `order` window spanning the fragment0/fragment1 spine
    # boundary is [.., N1, CA1, C1, N2, ..].
    boundary_start = order.index(C1) - 2
    window_seg = seg[boundary_start:boundary_start + 4]
    assert len(set(window_seg)) > 1, "fragment boundary must break segment uniformity"


# ---------------------------------------------------------------------------
# (c) sample_protein_ligand(protein_order='torsion'): segment gating.
# ---------------------------------------------------------------------------

def _torsion_complex():
    """3-residue protein backbone (test (b)'s fixture) + a small connected
    ligand star, with torsion_order/torsion_seg pre-attached."""
    global_idx, element, greek, residue_id, coords, coords_arr = _three_residue_backbone()
    order, seg = build_protein_torsion_order(global_idx, element, greek, residue_id, coords)

    # Ligand: 4-atom star (fully connected within 4.5A), placed far from
    # the protein except one contact atom (not required for this test, but
    # keeps the fixture consistent with test_protein_ligand_sampler.py's
    # conventions).
    lig = np.array([
        [7.0, 0.0, 0.0],
        [8.0, 1.0, 0.0],
        [8.0, -1.0, 0.0],
        [7.0, 0.0, 2.0],
    ])
    all_pos = np.vstack([coords_arr, lig])
    n = all_pos.shape[0]
    segment = torch.tensor([0] * 12 + [1] * 4, dtype=torch.long)
    residue_id_t = torch.tensor(residue_id + [-1] * 4, dtype=torch.long)

    # Radius-graph edges (4.5A) for get_neighbor_dict / edge-window feature.
    pos_t = torch.tensor(all_pos, dtype=torch.float32)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and torch.norm(pos_t[i] - pos_t[j]).item() < 4.5:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(edge_index=edge_index, pos=pos_t, num_nodes=n)
    data.x_emb = torch.arange(n, dtype=torch.long)
    data.segment = segment
    data.residue_id = residue_id_t
    data.torsion_order = torch.tensor(order, dtype=torch.long)
    data.torsion_seg = torch.tensor(seg, dtype=torch.long)
    return data, order, seg, coords_arr


def test_torsion_segment_gating_zero_at_starts_nonzero_within():
    random.seed(17)
    data, prot_order, prot_seg, coords_arr = _torsion_complex()
    vocab = {'PAD': 100}
    m, s, max_len = 1, 2, 40
    angle_K, dihedral_K = 4, 2

    out = sample_protein_ligand(
        data, m, s, max_len, vocab, angles=True, dihedrals=True,
        angle_K=angle_K, dihedral_K=dihedral_K, protein_order='torsion',
    )
    L = int(out.lengths[0])
    ids = out.walk_ids[0, 0, :L].tolist()
    assert ids[:len(prot_order)] == prot_order  # torsion prefix used verbatim (no truncation)

    angle_slice = out.walk_pe[0, :, s: s + angle_K]
    dihedral_slice = out.walk_pe[0, :, s + angle_K:]

    # Build the FULL per-position segment array exactly as the sampler does
    # internally (protein segs ++ one sentinel seg for the ligand walk).
    lig_len = L - len(prot_order)
    seg_full = prot_seg + [-1] * lig_len

    for pos in range(2, L):
        same3 = seg_full[pos - 2] == seg_full[pos - 1] == seg_full[pos]
        if not same3:
            assert torch.allclose(angle_slice[pos], torch.zeros(angle_K)), (
                f"angle at pos={pos} should be gated to 0 (segment discontinuity)"
            )
    for pos in range(3, L):
        same4 = (seg_full[pos - 3] == seg_full[pos - 2] ==
                 seg_full[pos - 1] == seg_full[pos])
        if not same4:
            assert torch.allclose(dihedral_slice[pos], torch.zeros(2 * dihedral_K)), (
                f"dihedral at pos={pos} should be gated to 0 (segment discontinuity)"
            )

    # fragment0 spine (positions 0-5, all seg 0) is long enough (6 atoms)
    # that SOME position within it has a nonzero angle/dihedral.
    assert any(
        not torch.allclose(angle_slice[p], torch.zeros(angle_K)) for p in range(2, 6)
    )
    assert any(
        not torch.allclose(dihedral_slice[p], torch.zeros(2 * dihedral_K))
        for p in range(3, 6)
    )

    # Cross-check: the dihedral at pos=3 (N0,CA0,C0,N1, all seg 0) recovers
    # the same phi via its l=1 (sin, cos) slot as an independent formula.
    # _dihedral_basis layout is [sin(l=1..K), cos(l=1..K)] (NOT interleaved
    # per-l): l=1's sin is at offset 0, l=1's cos is at offset dihedral_K.
    N0, CA0, C0, N1 = 0, 1, 2, 4
    expected_phi = _independent_dihedral(coords_arr[N0], coords_arr[CA0],
                                         coords_arr[C0], coords_arr[N1])
    sin1 = dihedral_slice[3, 0].item()
    cos1 = dihedral_slice[3, dihedral_K].item()
    recovered_phi = float(np.arctan2(sin1, cos1))
    assert recovered_phi == pytest.approx(expected_phi, abs=1e-4)


def test_torsion_missing_attrs_falls_back_to_allatom_with_warning():
    data, _, _, _ = _torsion_complex()
    del data.torsion_order
    del data.torsion_seg
    vocab = {'PAD': 100}
    with pytest.warns(RuntimeWarning, match="falling back to the 'allatom'"):
        out = sample_protein_ligand(data, 2, 2, 30, vocab, protein_order='torsion')
    # Flat (residue_id, atom_index) order == allatom fallback.
    assert out.walk_ids[0, 0, :12].tolist() == list(range(12))


# ---------------------------------------------------------------------------
# (d) REGRESSION: protein_order='allatom' (default AND explicit) is
#     byte-identical to a no-kwarg call, same seed.
# ---------------------------------------------------------------------------

def _plain_complex():
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [2.0, 0.3, 0.0], [4.0, 0.0, 0.5],
        [20.0, 1.0, 0.0], [22.0, 0.0, 1.0],
        [7.0, 0.0, 0.0], [8.0, 1.0, 0.0], [8.0, -1.0, 0.0], [7.0, 0.0, 2.0],
    ])
    n = pos.shape[0]
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and torch.norm(pos[i] - pos[j]).item() <= 4.5:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(edge_index=edge_index, pos=pos, num_nodes=n)
    data.x_emb = torch.arange(n, dtype=torch.long)
    data.segment = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    data.residue_id = torch.tensor([0, 0, 0, 1, 1, -1, -1, -1, -1], dtype=torch.long)
    return data


def test_allatom_default_and_explicit_byte_identical_to_no_kwarg():
    vocab = {'PAD': 100}
    m, s, max_len = 5, 2, 20
    angle_K, dihedral_K = 6, 3

    random.seed(123)
    out_no_kwarg = sample_protein_ligand(
        _plain_complex(), m, s, max_len, vocab,
        angles=True, dihedrals=True, angle_K=angle_K, dihedral_K=dihedral_K,
    )
    random.seed(123)
    out_default = sample_protein_ligand(
        _plain_complex(), m, s, max_len, vocab,
        angles=True, dihedrals=True, angle_K=angle_K, dihedral_K=dihedral_K,
        protein_order='allatom',
    )
    random.seed(123)
    out_explicit = sample_protein_ligand(
        _plain_complex(), m, s, max_len, vocab,
        angles=True, dihedrals=True, angle_K=angle_K, dihedral_K=dihedral_K,
        protein_order='allatom',
    )

    for other in (out_default, out_explicit):
        assert torch.equal(out_no_kwarg.walk_emb, other.walk_emb)
        assert torch.equal(out_no_kwarg.walk_ids, other.walk_ids)
        assert torch.equal(out_no_kwarg.walk_pe, other.walk_pe)
        assert torch.equal(out_no_kwarg.lengths, other.lengths)


def test_allatom_no_deprecation_warning_emitted():
    vocab = {'PAD': 100}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here is a bug
        sample_protein_ligand(_plain_complex(), 2, 2, 20, vocab)
        sample_protein_ligand(_plain_complex(), 2, 2, 20, vocab, protein_order='allatom')


# ---------------------------------------------------------------------------
# (e) integration: build_data + build_protein_torsion_order -> sample
#     (protein_order='torsion') -> RSNN_TRSF_Reg forward is finite.
# ---------------------------------------------------------------------------

def test_integration_torsion_through_model_forward():
    torch.manual_seed(0)
    random.seed(0)

    global_idx, element, greek, residue_id, coords, coords_arr = _three_residue_backbone()
    lig = np.array([[3.0, 2.0, 0.0], [3.0, 3.4, 0.0], [3.0, 4.8, 0.0], [4.2, 5.4, 0.0]])
    all_coords = np.vstack([coords_arr, lig])
    elements_full = element + ['C', 'O', 'N', 'C']
    segment_full = np.array([0] * 12 + [1] * 4)
    residue_id_full = np.array(residue_id + [-1] * 4)

    decoded = {
        'coords': all_coords, 'elements': elements_full,
        'segment': segment_full, 'residue_id': residue_id_full,
        'label': 3.0, 'pdb_id': 'synth_torsion',
    }
    vocab = build_pdbbind_vocab([decoded])
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)

    order, seg = build_protein_torsion_order(global_idx, element, greek, residue_id, coords)
    data.torsion_order = torch.tensor(order, dtype=torch.long)
    data.torsion_seg = torch.tensor(seg, dtype=torch.long)

    s, m, max_len = 8, 3, 30
    angle_K, dihedral_K = 6, 3
    data = sample_protein_ligand(
        data, m=m, s=s, max_len=max_len, vocab=vocab,
        angles=True, dihedrals=True, angle_K=angle_K, dihedral_K=dihedral_K,
        protein_order='torsion',
    )
    for k in ("x", "pos", "z", "edge_index", "edge_attr", "segment",
             "residue_id", "num_nodes", "pdb_id", "torsion_order", "torsion_seg"):
        if hasattr(data, k):
            delattr(data, k)

    from torch_geometric.loader import DataLoader
    loader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader))

    pe_in_dim = compute_pe_in_dim("search", w=s, distances=0, mol_edge_feat=0,
                                  angles=1, dihedrals=1,
                                  angle_K=angle_K, dihedral_K=dihedral_K)
    assert batch.walk_pe.shape[-1] == pe_in_dim

    model = RSNN_TRSF_Reg(pe_in_dim, 16, 32, 1, 2, len(vocab), reduce="mean",
                          dropout=0.0, nhead=4, ffn_mult=4, attn_mode="full",
                          pos_enc="sinusoidal")
    out = model(batch)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()

    loss = torch.nn.functional.mse_loss(out.squeeze(-1), batch.y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
