"""Tests for the ligand bond-graph ABLATION (``ligand_graph='bond'``).

Covers:
  (a) ``extract_ligand_bond_edges`` on a synthetic 3-atom chain (permuted
      indexing, to catch any accidental identity-order assumption) and on a
      REAL PDBbind complex (exact coordinate match + sane bond count).
  (b) ``sample_protein_ligand(..., ligand_graph='bond')`` walks ONLY bonded
      ligand atoms: a synthetic ligand where two atoms are < 4.5A apart (so
      connected in the distance graph) but NOT bonded -- distance-mode can
      place them adjacent in a walk, bond-mode never co-occurs them.
  (c) ``ligand_graph='distance'`` (default, and the byte-identical
      regression against calling WITHOUT the new kwarg at all).
  (d) End-to-end: ``build_data`` + ``extract_ligand_bond_edges`` (real
      rdkit-authored raw sdf files) -> ``sample_protein_ligand(ligand_graph=
      'bond')`` -> ``RSNN_TRSF_Reg`` forward, finite (2, 1) output.

Run::

    source /home/snirhordan/miniconda3/etc/profile.d/conda.sh && conda activate rwnn
    cd /home/snirhordan/ito/RandomSearchNNs
    python -m pytest tests/test_ligand_bond_graph.py -v --tb=short
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.pdbbind import (  # noqa: E402
    build_data,
    build_pdbbind_vocab,
    extract_ligand_bond_edges,
    load_pdbbind_cache,
)
from utils.search import sample_protein_ligand  # noqa: E402
from quickstart.train_qm9 import RSNN_TRSF_Reg, compute_pe_in_dim  # noqa: E402


REAL_CACHE = ROOT / "data" / "pdbbind" / "identity30" / "test.pt"
REAL_RAW_DIR = ROOT / "GET" / "datasets" / "PDBBind" / "pdbbind" / "pdb_files"


# ---------------------------------------------------------------------------
# Helpers: hand-authored V2000 molblocks (no external rdkit-dependent writer
# needed -- full control over coordinates/bonds for exact test assertions).
# ---------------------------------------------------------------------------


def _write_ligand_sdf(raw_dir: Path, pdb_id: str, coords, bonds) -> None:
    """Write a minimal valid V2000 SDF: all-carbon atoms at ``coords``
    (n, 3), single bonds at 1-indexed ``bonds`` pairs."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    nb = len(bonds)
    lines = ["", "  synthtest", ""]
    lines.append(f"{n:3d}{nb:3d}  0  0  0  0  0  0  0  0999 V2000")
    for x, y, z in coords:
        lines.append(
            f"{x:10.4f}{y:10.4f}{z:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for a, b in bonds:
        lines.append(f"{a:3d}{b:3d}  1  0")
    lines.append("M  END")
    lines.append("$$$$")
    molblock = "\n".join(lines) + "\n"

    sub = raw_dir / pdb_id
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{pdb_id}_ligand.sdf").write_text(molblock)


# ===========================================================================
# (a) extract_ligand_bond_edges
# ===========================================================================


def test_extract_ligand_bond_edges_synthetic_chain(tmp_path):
    """3-atom chain (0-1-2, no 0-2 bond); permuted our-indexing to catch
    accidental identity-order assumptions in the nearest-coordinate match."""
    raw_dir = tmp_path / "raw"
    coords = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]
    _write_ligand_sdf(raw_dir, "CHAIN3", coords, bonds=[(1, 2), (2, 3)])

    # Permuted order: our global index 12 <-> rdkit atom 0 (0,0,0), 11 <->
    # rdkit atom 1 (1.5,0,0), 10 <-> rdkit atom 2 (3.0,0,0).
    our_idx = [12, 11, 10]
    our_coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])

    edge_index = extract_ligand_bond_edges("CHAIN3", our_idx, our_coords, str(raw_dir))
    assert edge_index.dtype == torch.long
    assert edge_index.shape[0] == 2

    edges = set(map(tuple, edge_index.t().tolist()))
    assert edges == {(12, 11), (11, 12), (11, 10), (10, 11)}
    # No spurious end-to-end bond.
    assert (12, 10) not in edges and (10, 12) not in edges


def test_extract_ligand_bond_edges_no_bonds_returns_empty(tmp_path):
    """A single isolated heavy atom -> zero heavy-heavy bonds -> (2, 0)."""
    raw_dir = tmp_path / "raw"
    _write_ligand_sdf(raw_dir, "LONE1", [[0.0, 0.0, 0.0]], bonds=[])
    edge_index = extract_ligand_bond_edges("LONE1", [42], [[0.0, 0.0, 0.0]], str(raw_dir))
    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == torch.long


def test_extract_ligand_bond_edges_coord_mismatch_raises(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_ligand_sdf(raw_dir, "BAD1", [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                      bonds=[(1, 2)])
    # our coords are far away (> 0.5 A) from the rdkit ones.
    bad_coords = [[100.0, 100.0, 100.0], [101.5, 100.0, 100.0]]
    try:
        extract_ligand_bond_edges("BAD1", [0, 1], bad_coords, str(raw_dir))
        assert False, "expected ValueError on coordinate mismatch"
    except ValueError:
        pass


def test_extract_ligand_bond_edges_real_complex():
    """Real PDBbind complex: exact coordinate match, sane bond count."""
    if not REAL_CACHE.exists() or not REAL_RAW_DIR.exists():
        import pytest as _pytest
        _pytest.skip("real PDBbind cache/raw_dir not available in this environment")

    data_list = load_pdbbind_cache(str(REAL_CACHE))
    complex_ = data_list[0]
    lig_mask = complex_.segment == 1
    lig_global_idx = torch.nonzero(lig_mask, as_tuple=False).view(-1).tolist()
    lig_coords = complex_.pos[lig_mask].numpy()
    n_lig = len(lig_global_idx)
    assert n_lig > 0

    edge_index = extract_ligand_bond_edges(
        complex_.pdb_id, lig_global_idx, lig_coords, str(REAL_RAW_DIR)
    )
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] % 2 == 0, "edges must be undirected (paired)"
    n_bonds = edge_index.shape[1] // 2
    # Sane bond count for a small-molecule ligand: at least a spanning
    # structure, and not absurdly more than a fully-connected graph.
    assert n_lig - 1 <= n_bonds <= n_lig * (n_lig - 1) // 2
    # All endpoints are valid ligand global indices.
    endpoints = set(edge_index.flatten().tolist())
    assert endpoints <= set(lig_global_idx)
    # Undirected: (i, j) implies (j, i).
    edges = set(map(tuple, edge_index.t().tolist()))
    assert all((j, i) in edges for (i, j) in edges)


# ===========================================================================
# (b) sampler: ligand_graph='bond' walks only bonded ligand atoms
# ===========================================================================


def _make_two_close_not_bonded_complex():
    """1 protein atom (segment 0) + 3 ligand atoms A, B, C (segment 1).

    Distance graph (data.edge_index): protein-A only interface edge, plus
    A<->B (the pair we test: close in space, NOT bonded). C is isolated in
    the distance graph.
    Bond graph (data.ligand_bond_edge_index): A<->C only (non-empty, so
    ligand_graph='bond' does NOT fall back). B is isolated in the bond
    graph; A and B are never bond-connected (directly or indirectly).

    Global indices: protein=0, A=1, B=2, C=3.
    """
    pos = torch.tensor([
        [0.0, 0.0, 0.0],   # 0 protein
        [3.0, 0.0, 0.0],   # 1 ligand A
        [4.5, 0.0, 0.0],   # 2 ligand B  (dist(A,B)=1.5 < 4.5)
        [50.0, 0.0, 0.0],  # 3 ligand C  (far from A/B in the distance graph)
    ])
    segment = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    residue_id = torch.tensor([0, -1, -1, -1], dtype=torch.long)
    x_emb = torch.arange(4, dtype=torch.long)

    # Distance graph: protein(0)<->A(1), A(1)<->B(2). Undirected.
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
    )
    data = Data(edge_index=edge_index, pos=pos, num_nodes=4)
    data.x_emb = x_emb
    data.segment = segment
    data.residue_id = residue_id

    # Bond graph: A(1)<->C(3) only. B(2) isolated.
    data.ligand_bond_edge_index = torch.tensor(
        [[1, 3], [3, 1]], dtype=torch.long
    )
    return data


def test_bond_mode_never_co_occurs_disconnected_pair():
    data = _make_two_close_not_bonded_complex()
    vocab = {"PAD": 99}
    m, s, max_len = 30, 2, 10

    random.seed(123)
    out = sample_protein_ligand(data, m, s, max_len, vocab, ligand_graph="bond")

    saw_a_and_b = False
    for i in range(m):
        L = int(out.lengths[i])
        ids = set(out.walk_ids[0, i, :L].tolist())
        # A and B must never co-occur: they are not bond-connected, directly
        # or indirectly (B is isolated in the bond graph).
        assert not ({1, 2} <= ids), f"walk {i} contains both A and B: {ids}"
        if {1, 2} <= ids:
            saw_a_and_b = True
    assert not saw_a_and_b


def test_distance_mode_can_place_the_pair_adjacent():
    data = _make_two_close_not_bonded_complex()
    vocab = {"PAD": 99}
    m, s, max_len = 30, 2, 10

    random.seed(123)
    out = sample_protein_ligand(data, m, s, max_len, vocab, ligand_graph="distance")

    found_adjacent_ab = False
    for i in range(m):
        L = int(out.lengths[i])
        ids = out.walk_ids[0, i, :L].tolist()
        for pos_ in range(1, L):
            pair = {ids[pos_ - 1], ids[pos_]}
            if pair == {1, 2}:
                found_adjacent_ab = True
    assert found_adjacent_ab, (
        "expected at least one distance-mode walk (over 30 draws) to step "
        "directly between ligand atoms A(1) and B(2); the distance graph "
        "connects them, so this should occur with overwhelming probability"
    )


def test_bond_graph_falls_back_to_distance_when_missing():
    """No ligand_bond_edge_index attribute at all -> silent fallback to
    distance-mode neighbors (must NOT crash; must reproduce distance-mode
    connectivity, so A/B CAN co-occur/be adjacent just like distance mode)."""
    data = _make_two_close_not_bonded_complex()
    del data.ligand_bond_edge_index
    vocab = {"PAD": 99}
    m, s, max_len = 30, 2, 10

    random.seed(123)
    out_fallback = sample_protein_ligand(data, m, s, max_len, vocab, ligand_graph="bond")

    data2 = _make_two_close_not_bonded_complex()
    del data2.ligand_bond_edge_index
    random.seed(123)
    out_distance = sample_protein_ligand(data2, m, s, max_len, vocab, ligand_graph="distance")

    assert torch.equal(out_fallback.walk_ids, out_distance.walk_ids)
    assert torch.equal(out_fallback.lengths, out_distance.lengths)


def test_bond_graph_falls_back_to_distance_when_empty():
    """An explicitly EMPTY ligand_bond_edge_index also falls back."""
    data = _make_two_close_not_bonded_complex()
    data.ligand_bond_edge_index = torch.empty((2, 0), dtype=torch.long)
    vocab = {"PAD": 99}
    m, s, max_len = 30, 2, 10

    random.seed(123)
    out_fallback = sample_protein_ligand(data, m, s, max_len, vocab, ligand_graph="bond")

    data2 = _make_two_close_not_bonded_complex()
    del data2.ligand_bond_edge_index
    random.seed(123)
    out_distance = sample_protein_ligand(data2, m, s, max_len, vocab, ligand_graph="distance")

    assert torch.equal(out_fallback.walk_ids, out_distance.walk_ids)


# ===========================================================================
# (c) ligand_graph='distance' (default) is byte-identical to omitting the
#     new kwarg entirely.
# ===========================================================================


def _regression_complex():
    """9-atom synthetic complex (mirrors tests/test_protein_ligand_sampler.py)."""
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.3, 0.0],
        [4.0, 0.0, 0.5],
        [20.0, 1.0, 0.0],
        [22.0, 0.0, 1.0],
        [7.0, 0.0, 0.0],
        [8.0, 1.0, 0.0],
        [8.0, -1.0, 0.0],
        [7.0, 0.0, 2.0],
    ])
    n = pos.shape[0]
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and torch.norm(pos[i] - pos[j]).item() <= 4.5:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    segment = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    residue_id = torch.tensor([0, 0, 0, 1, 1, -1, -1, -1, -1], dtype=torch.long)
    x_emb = torch.arange(n, dtype=torch.long)
    data = Data(edge_index=edge_index, pos=pos, num_nodes=n)
    data.x_emb = x_emb
    data.segment = segment
    data.residue_id = residue_id
    return data


def test_ligand_graph_distance_default_is_byte_identical_regression():
    vocab = {"PAD": 100}
    m, s, max_len = 6, 2, 20

    random.seed(7)
    out_no_kwarg = sample_protein_ligand(_regression_complex(), m, s, max_len, vocab)

    random.seed(7)
    out_explicit_distance = sample_protein_ligand(
        _regression_complex(), m, s, max_len, vocab, ligand_graph="distance"
    )

    assert torch.equal(out_no_kwarg.walk_emb, out_explicit_distance.walk_emb)
    assert torch.equal(out_no_kwarg.walk_ids, out_explicit_distance.walk_ids)
    assert torch.equal(out_no_kwarg.walk_pe, out_explicit_distance.walk_pe)
    assert torch.equal(out_no_kwarg.lengths, out_explicit_distance.lengths)


# ===========================================================================
# (d) integration: build_data (+attach bonds) -> sample_protein_ligand
#     (ligand_graph='bond') -> RSNN_TRSF_Reg forward, finite (2, 1).
# ===========================================================================


def _synthetic_decoded(pdb_id, seed):
    """Protein (2 residues, 5 atoms) + a 3-atom ligand chain, all coordinates
    exact so the rdkit-authored sdf below can match them exactly."""
    rng = np.random.default_rng(seed)
    prot = np.array([[0, 0, 0], [1.4, 0, 0], [2.6, 0.6, 0],
                     [4.0, 0.4, 0], [5.2, -0.3, 0]], dtype=np.float64)
    lig = np.array([[3.0, 2.0, 0.0], [3.0, 3.5, 0.0], [3.0, 5.0, 0.0]],
                   dtype=np.float64)
    lig += rng.normal(0, 0.01, lig.shape)
    coords = np.vstack([prot, lig])
    elements = ['C', 'N', 'O', 'C', 'N', 'C', 'C', 'C']
    segment = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    residue_id = np.array([0, 0, 0, 1, 1, -1, -1, -1])
    return {
        'coords': coords, 'elements': elements, 'segment': segment,
        'residue_id': residue_id, 'label': float(seed), 'pdb_id': pdb_id,
    }, lig


def test_build_data_attach_bonds_sampler_bond_mode_model_forward(tmp_path):
    import random as _random
    torch.manual_seed(0)
    _random.seed(0)

    raw_dir = tmp_path / "raw"
    decoded_list = []
    lig_coords_list = []
    for i, pdb_id in enumerate(["synthA", "synthB"]):
        decoded, lig_coords = _synthetic_decoded(pdb_id, seed=i)
        decoded_list.append(decoded)
        lig_coords_list.append(lig_coords)
        # ligand chain bonds: 0-1, 1-2 (1-indexed for the molfile).
        _write_ligand_sdf(raw_dir, pdb_id, lig_coords, bonds=[(1, 2), (2, 3)])

    vocab = build_pdbbind_vocab(decoded_list)
    assert vocab['PAD'] == len(vocab) - 1

    s, m, max_len = 8, 4, 20
    data_list = []
    for decoded in decoded_list:
        data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)

        lig_mask = data.segment == 1
        lig_global_idx = torch.nonzero(lig_mask, as_tuple=False).view(-1).tolist()
        lig_coords = data.pos[lig_mask].numpy()
        data.ligand_bond_edge_index = extract_ligand_bond_edges(
            decoded['pdb_id'], lig_global_idx, lig_coords, str(raw_dir)
        )
        assert data.ligand_bond_edge_index.shape == (2, 4)  # 2 bonds, undirected

        data = sample_protein_ligand(
            data, m=m, s=s, max_len=max_len, vocab=vocab,
            angles=True, dihedrals=True, ligand_graph="bond",
        )
        for k in ("x", "pos", "z", "edge_index", "edge_attr", "segment",
                 "residue_id", "num_nodes", "pdb_id", "ligand_bond_edge_index"):
            if hasattr(data, k):
                delattr(data, k)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    pe_in_dim = compute_pe_in_dim("search", w=s, distances=0, mol_edge_feat=0,
                                  angles=1, dihedrals=1, angle_K=8, dihedral_K=4)
    assert batch.walk_pe.shape[-1] == pe_in_dim

    model = RSNN_TRSF_Reg(pe_in_dim, 16, 32, 1, 2, len(vocab), reduce="mean",
                          dropout=0.0, nhead=4, ffn_mult=4, attn_mode="full",
                          pos_enc="sinusoidal")
    out = model(batch)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()
