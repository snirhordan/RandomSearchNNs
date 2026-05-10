"""Tests for SMILES->graph preprocessing utilities."""
import sys
from pathlib import Path

import numpy as np
import pytest
import networkx as nx
import regex as re
import torch
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.utils import (
    get_canonical_molecule,
    mol2graph,
    get_ids,
)
from generation.scaffold_split import random_split


# ---------------------------------------------------------------------------
# get_canonical_molecule
# ---------------------------------------------------------------------------
def _to_canon_smi(mol):
    return Chem.MolToSmiles(mol, canonical=True, ignoreAtomMapNumbers=True)


def test_canonical_molecule_returns_mol():
    mol = Chem.MolFromSmiles("CCO")
    canon = get_canonical_molecule(mol)
    assert canon is not None
    assert canon.GetNumAtoms() == mol.GetNumAtoms()


def test_canonical_molecule_idempotent():
    """Canonicalizing twice must yield the same SMILES as canonicalizing once."""
    mol = Chem.MolFromSmiles("OCC")  # ethanol, non-canonical
    once = get_canonical_molecule(mol)
    twice = get_canonical_molecule(once)
    assert _to_canon_smi(once) == _to_canon_smi(twice)


def test_canonical_molecule_invariant_to_input_order():
    """Different starting orders of the same molecule must yield same canonical form."""
    smis = ["CCO", "OCC", "C(O)C"]
    canons = [_to_canon_smi(get_canonical_molecule(Chem.MolFromSmiles(s))) for s in smis]
    assert len(set(canons)) == 1


# ---------------------------------------------------------------------------
# mol2graph
# ---------------------------------------------------------------------------
def test_mol2graph_shapes(tiny_graph):
    g = tiny_graph
    # x: (num_atoms, 9 atom features)
    assert g.x.dim() == 2
    assert g.x.shape[0] == 5            # pentane
    assert g.x.shape[1] == 9
    # edge_index: (2, num_edges)
    assert g.edge_index.dim() == 2
    assert g.edge_index.shape[0] == 2
    # edge_attr: (num_edges, 3)
    assert g.edge_attr.dim() == 2
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]
    assert g.edge_attr.shape[1] == 3
    # node count must match RDKit
    assert g.x.shape[0] == 5


def test_mol2graph_num_nodes_matches_mol(regex_tokenizer, tiny_vocab):
    smi = "c1ccccc1"  # benzene, 6 atoms
    mol = Chem.MolFromSmiles(smi)
    g = mol2graph(mol, tokenizer=regex_tokenizer, vocab=tiny_vocab)
    assert g.x.shape[0] == mol.GetNumAtoms() == 6


def test_mol2graph_undirected_edges(tiny_graph):
    """Every (i,j) edge should also appear as (j,i)."""
    edges = set()
    ei = tiny_graph.edge_index
    for k in range(ei.shape[1]):
        edges.add((ei[0, k].item(), ei[1, k].item()))
    for (a, b) in list(edges):
        assert (b, a) in edges


def test_mol2graph_no_self_loops(tiny_graph):
    ei = tiny_graph.edge_index
    assert (ei[0] == ei[1]).sum().item() == 0


def _graph_hash(data):
    """Produce a Weisfeiler-Lehman hash of the graph using node features as labels."""
    g = nx.Graph()
    n = data.x.shape[0]
    for i in range(n):
        # collapse the 9-dim atom feature vector to a string label
        feat = tuple(int(v) for v in data.x[i].tolist())
        g.add_node(i, label=str(feat))
    ei = data.edge_index
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u != v:
            g.add_edge(u, v)
    return nx.weisfeiler_lehman_graph_hash(g, node_attr='label')


def test_mol2graph_isomorphism_under_smiles_canonicalization(regex_tokenizer, tiny_vocab):
    """The molecular graph must not depend on input SMILES atom ordering.

    Both ``OCC`` and ``CCO`` describe the same ethanol; the resulting
    canonical-graph hash should agree.
    """
    g_a = mol2graph(Chem.MolFromSmiles("CCO"), tokenizer=regex_tokenizer, vocab=tiny_vocab)
    g_b = mol2graph(Chem.MolFromSmiles("OCC"), tokenizer=regex_tokenizer, vocab=tiny_vocab)
    assert _graph_hash(g_a) == _graph_hash(g_b)


def test_mol2graph_x_emb_set_when_vocab_given(tiny_graph):
    assert hasattr(tiny_graph, 'x_emb')
    assert tiny_graph.x_emb.shape[0] == tiny_graph.x.shape[0]
    assert tiny_graph.x_emb.dtype in (torch.int32, torch.long, torch.int)


# ---------------------------------------------------------------------------
# get_ids
# ---------------------------------------------------------------------------
def test_get_ids_returns_correct_subset():
    items = ['a', 'b', 'c', 'd', 'e']
    out = get_ids(items, [0, 2, 4])
    assert out == ['a', 'c', 'e']


def test_get_ids_with_tensor():
    items = [torch.tensor(i) for i in range(5)]
    out = get_ids(items, np.array([1, 3]))
    assert len(out) == 2
    assert int(out[0]) == 1
    assert int(out[1]) == 3


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------
def test_random_split_partitions_disjoint_and_complete():
    n = 100
    splits = random_split(n, test_size=0.2, val_size=0.2, n_splits=3, random_state=42)
    for key in ('train', 'valid', 'test'):
        assert key in splits
        assert len(splits[key]) == 3

    for s in range(3):
        tr = set(splits['train'][s].tolist())
        va = set(splits['valid'][s].tolist())
        te = set(splits['test'][s].tolist())
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)
        union = tr | va | te
        assert union == set(range(n))


def test_random_split_proportions():
    n = 1000
    splits = random_split(n, test_size=0.2, val_size=0.2, n_splits=1, random_state=0)
    assert len(splits['train'][0]) == int(n * 0.6)
    assert len(splits['valid'][0]) == int(n * 0.8) - int(n * 0.6)
    # remainder is test
    assert len(splits['test'][0]) == n - int(n * 0.8)


def test_random_split_deterministic_for_same_seed():
    a = random_split(50, n_splits=1, random_state=7)
    b = random_split(50, n_splits=1, random_state=7)
    assert np.array_equal(a['train'][0], b['train'][0])
    assert np.array_equal(a['valid'][0], b['valid'][0])
    assert np.array_equal(a['test'][0], b['test'][0])
