"""Tests for generation/pdbbind.py (PDBbind atom-level RSNN preprocessing).

All inputs are synthetic -- no dependency on the real PDBbind dataset. These
tests pin down the two things that must be provably correct before any real
data flows through this module: the graph-construction recipe (strict
``d < cutoff``, ``1/d + eps`` weights, one-hot features, SE(3) invariance)
and the global-node-stripping / residue-ordering logic in ``decode_item``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.pdbbind import (
    IDX2ATOM,
    element_to_z,
    decode_item,
    build_pdbbind_vocab,
    build_data,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _edge_weight_map(data):
    """{(i, j): weight} view of edge_index/edge_attr for set-style comparisons."""
    ei = data.edge_index
    ea = data.edge_attr
    out = {}
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        out[(i, j)] = float(ea[k, 0])
    return out


def _make_decoded(coords, elements, segment, residue_id, label=1.0, pdb_id='synth'):
    return {
        'coords': np.asarray(coords, dtype=np.float64),
        'elements': list(elements),
        'segment': np.asarray(segment, dtype=np.int64),
        'residue_id': np.asarray(residue_id, dtype=np.int64),
        'label': float(label),
        'pdb_id': pdb_id,
    }


# ---------------------------------------------------------------------------
# build_data: graph construction recipe
# ---------------------------------------------------------------------------

# 7 colinear points on the x-axis: pairwise distance == |xi - xj|, exact in
# float64. The (0, 4) pair sits at exactly 4.5 -- the strict `d < cutoff`
# boundary case -- and must be EXCLUDED.
_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (4.5, 0.0, 0.0),
    (6.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
]
_ELEMENTS = ['C', 'N', 'O', 'C', 'N', 'O', 'C']
_SEGMENT = [0, 0, 0, 0, 0, 1, 1]
_RESIDUE_ID = [0, 0, 1, 1, 2, -1, -1]
_LABEL = -7.3
_PDB_ID = 'synth_1'

_EXPECTED_EDGES = {
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
    (5, 6),
}


def _decoded_fixture():
    return _make_decoded(_COORDS, _ELEMENTS, _SEGMENT, _RESIDUE_ID, _LABEL, _PDB_ID)


def _vocab_fixture():
    return build_pdbbind_vocab([_decoded_fixture()])


def test_build_data_edge_set_matches_strict_cutoff():
    decoded = _decoded_fixture()
    vocab = _vocab_fixture()
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)

    ei = data.edge_index
    edge_set = {(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])}

    expected_directed = set()
    for (i, j) in _EXPECTED_EDGES:
        expected_directed.add((i, j))
        expected_directed.add((j, i))

    assert edge_set == expected_directed
    # boundary pair (0, 4): d == 4.5 exactly -> must not appear, either direction
    assert (0, 4) not in edge_set and (4, 0) not in edge_set
    # no self loops
    assert all(i != j for (i, j) in edge_set)


def test_build_data_edge_weights_numeric():
    decoded = _decoded_fixture()
    vocab = _vocab_fixture()
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)
    coords = np.asarray(_COORDS)

    wmap = _edge_weight_map(data)
    assert len(wmap) == 2 * len(_EXPECTED_EDGES)
    for (i, j), w in wmap.items():
        d = float(np.linalg.norm(coords[i] - coords[j]))
        assert w == pytest.approx(1.0 / d + 1e-5, abs=1e-6)


def test_build_data_undirected():
    decoded = _decoded_fixture()
    vocab = _vocab_fixture()
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)
    wmap = _edge_weight_map(data)
    for (i, j) in wmap:
        assert (j, i) in wmap
        assert wmap[(i, j)] == pytest.approx(wmap[(j, i)], abs=1e-6)


def test_build_data_one_hot_matches_x_emb():
    decoded = _decoded_fixture()
    vocab = _vocab_fixture()
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)

    assert data.x.shape == (len(_ELEMENTS), len(vocab))
    row_sums = data.x.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
    for i, elem in enumerate(_ELEMENTS):
        assert data.x_emb[i].item() == vocab[elem]
        assert data.x[i, vocab[elem]].item() == 1.0
    # PAD column (last index) is never set by any real atom
    assert torch.all(data.x[:, vocab['PAD']] == 0)


def test_build_data_z_correct():
    decoded = _decoded_fixture()
    vocab = _vocab_fixture()
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)
    for i, elem in enumerate(_ELEMENTS):
        assert data.z[i].item() == element_to_z(elem)
    assert element_to_z('H') == 1
    assert element_to_z('C') == 6
    assert element_to_z('Og') == 118


def test_build_data_segment_residue_label_numnodes():
    decoded = _decoded_fixture()
    vocab = _vocab_fixture()
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)

    assert data.segment.tolist() == _SEGMENT
    assert data.residue_id.tolist() == _RESIDUE_ID
    assert data.residue_id[data.segment == 1].tolist() == [-1, -1]
    assert data.y.shape == (1,)
    assert data.y.item() == pytest.approx(_LABEL)
    assert data.num_nodes == len(_ELEMENTS)
    assert data.pdb_id == _PDB_ID


# ---------------------------------------------------------------------------
# build_data: SE(3) invariance
# ---------------------------------------------------------------------------

def test_build_data_se3_invariance():
    decoded = _decoded_fixture()
    vocab = _vocab_fixture()
    base = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)

    rot = Rotation.from_euler('xyz', [23.0, -41.0, 67.0], degrees=True).as_matrix()
    trans = np.array([2.0, -3.5, 5.25])
    coords = np.asarray(_COORDS)
    transformed = coords @ rot.T + trans

    decoded_t = _make_decoded(transformed, _ELEMENTS, _SEGMENT, _RESIDUE_ID, _LABEL, _PDB_ID)
    moved = build_data(decoded_t, vocab, cutoff=4.5, eps=1e-5)

    base_edges = set(zip(base.edge_index[0].tolist(), base.edge_index[1].tolist()))
    moved_edges = set(zip(moved.edge_index[0].tolist(), moved.edge_index[1].tolist()))
    assert base_edges == moved_edges

    base_w = _edge_weight_map(base)
    moved_w = _edge_weight_map(moved)
    assert set(base_w) == set(moved_w)
    for key in base_w:
        assert moved_w[key] == pytest.approx(base_w[key], abs=1e-5)

    expected_pos = torch.as_tensor(transformed, dtype=torch.float32)
    assert torch.allclose(moved.pos, expected_pos, atol=1e-5)


# ---------------------------------------------------------------------------
# decode_item: global-node dropping + residue ordering
# ---------------------------------------------------------------------------

def _build_raw_item():
    def gid(sym):
        return IDX2ATOM.index(sym)

    # segment 0 (protein): global + residue(3 atoms) + residue(2 atoms)
    # segment 1 (ligand): global + 3 single-atom blocks
    protein_real_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                            (3.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    ligand_real_coords = [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (12.0, 0.0, 0.0)]

    protein_global_coord = tuple(np.mean(protein_real_coords, axis=0))
    ligand_global_coord = tuple(np.mean(ligand_real_coords, axis=0))

    X = ([protein_global_coord] + protein_real_coords
         + [ligand_global_coord] + ligand_real_coords)
    A = ([2, gid('C'), gid('N'), gid('O'), gid('C'), gid('C')]
         + [2, gid('F'), gid('Cl'), gid('Br')])
    atom_positions = [0] * len(A)
    block_lengths = [1, 3, 2, 1, 1, 1, 1]
    segment_ids = [0, 0, 0, 1, 1, 1, 1]

    data = {
        'X': np.asarray(X, dtype=np.float64),
        'B': [0] * len(block_lengths),
        'A': A,
        'atom_positions': atom_positions,
        'block_lengths': block_lengths,
        'segment_ids': segment_ids,
    }
    return {'id': 'synthABCD', 'affinity': {'neglog_aff': 5.42}, 'data': data}


def test_decode_item_drops_globals_and_orders_residues():
    item = _build_raw_item()
    decoded = decode_item(item)

    assert decoded['elements'] == ['C', 'N', 'O', 'C', 'C', 'F', 'Cl', 'Br']
    assert decoded['segment'].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert decoded['residue_id'].tolist() == [0, 0, 0, 1, 1, -1, -1, -1]
    assert decoded['label'] == pytest.approx(5.42)
    assert decoded['pdb_id'] == 'synthABCD'

    expected_coords = np.asarray([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0), (4.0, 0.0, 0.0),
        (10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (12.0, 0.0, 0.0),
    ])
    assert np.allclose(decoded['coords'], expected_coords)


def test_decode_item_handles_3d_X():
    """X.ndim == 3 (an n_channel axis): must take channel 0."""
    item = _build_raw_item()
    X2 = item['data']['X']
    X3 = np.stack([X2, X2 + 100.0], axis=1)  # (Natom, 2, 3); channel 0 == X2
    item['data']['X'] = X3
    decoded = decode_item(item)
    expected_coords = np.asarray([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0), (4.0, 0.0, 0.0),
        (10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (12.0, 0.0, 0.0),
    ])
    assert np.allclose(decoded['coords'], expected_coords)


# ---------------------------------------------------------------------------
# vocab
# ---------------------------------------------------------------------------

def test_build_pdbbind_vocab_reserved_and_deterministic():
    d1 = _make_decoded([(0, 0, 0), (1, 0, 0), (2, 0, 0)], ['C', 'N', 'O'], [0, 0, 0], [0, 0, 0])
    d2 = _make_decoded([(0, 0, 0), (1, 0, 0), (2, 0, 0)], ['C', 'C', 'N'], [0, 0, 0], [0, 0, 0])

    vocab1 = build_pdbbind_vocab([d1, d2])
    vocab2 = build_pdbbind_vocab([d1, d2])

    assert vocab1['UNK'] == 0
    assert vocab1['PAD'] == len(vocab1) - 1  # PAD last -> matches embedding padding_idx
    assert vocab1 == vocab2  # deterministic
    assert set(vocab1) == {'PAD', 'UNK', 'C', 'N', 'O'}
    # alphabetical order among non-reserved tokens (documented policy)
    assert vocab1['C'] < vocab1['N'] < vocab1['O']


def test_build_pdbbind_vocab_min_count_filters_rare_elements():
    d1 = _make_decoded([(0, 0, 0)] * 3, ['C', 'C', 'N'], [0, 0, 0], [0, 0, 0])
    vocab = build_pdbbind_vocab([d1], min_count=2)
    assert 'C' in vocab
    assert 'N' not in vocab  # occurs once, filtered out by min_count=2


def test_build_data_unknown_element_maps_to_unk():
    vocab = build_pdbbind_vocab([_decoded_fixture()])
    assert 'S' not in vocab  # not part of the fixture vocabulary

    decoded = _make_decoded(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], ['C', 'S'], [0, 0], [0, 0], label=0.0,
    )
    data = build_data(decoded, vocab, cutoff=4.5, eps=1e-5)
    assert data.x_emb[1].item() == vocab['UNK']
    assert data.x[1, vocab['UNK']].item() == 1.0
    assert data.x[1].sum().item() == 1.0
