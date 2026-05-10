"""Tests for the walk/search samplers in ``utils.search``."""
import sys
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.search import (
    sample_bfs,
    sample_dfs,
    sample_walks,
    sample_walks_mdlr,
    sample_walks_rum,
    sample_walks_adaptive,
    sample_walks_mdlr_adaptive,
    sample_walks_rum_adaptive,
    dfs_edges,
    get_neighbor_dict,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _edges_set(data):
    ei = data.edge_index
    out = set()
    for k in range(ei.shape[1]):
        out.add((int(ei[0, k]), int(ei[1, k])))
    return out


def _vocab_for(data):
    """Build a tiny vocab containing PAD; embedding ids on the data are 0."""
    return {'PAD': int(data.x_emb.max().item()) + 1}


def _data_with_vocab_emb(data):
    """Ensure ``x_emb`` accommodates the PAD index used as `vocab['PAD']`."""
    return data, _vocab_for(data)


# ---------------------------------------------------------------------------
# sample_walks
# ---------------------------------------------------------------------------
def test_sample_walks_shape_and_validity(tiny_graph):
    nw, l, s = 4, 6, 2
    data = sample_walks(tiny_graph, nw, l, s, non_backtracking=False)
    assert data.walk_emb.shape == (nw, l)
    assert data.walk_ids.shape == (1, nw, l)
    assert data.walk_pe.shape == (nw, l, 2 * s)
    n = data.x.shape[0]
    ids = data.walk_ids[0]
    assert ((ids >= 0) & (ids < n)).all()


def test_sample_walks_consecutive_vertices_connected(tiny_graph):
    nw, l, s = 8, 8, 2
    data = sample_walks(tiny_graph, nw, l, s, non_backtracking=False)
    edges = _edges_set(tiny_graph)
    ids = data.walk_ids[0]
    n = data.x.shape[0]
    for i in range(nw):
        for j in range(1, l):
            u = int(ids[i, j - 1])
            v = int(ids[i, j])
            # On a connected graph with no isolated nodes, consecutive RW
            # vertices must be edge-connected (or equal in a self-loop case).
            assert u == v or (u, v) in edges or (v, u) in edges


def test_sample_walks_deterministic_given_seed(tiny_graph):
    nw, l, s = 4, 6, 2
    random.seed(123); torch.manual_seed(123); np.random.seed(123)
    a = sample_walks(tiny_graph.clone(), nw, l, s, non_backtracking=False)
    random.seed(123); torch.manual_seed(123); np.random.seed(123)
    b = sample_walks(tiny_graph.clone(), nw, l, s, non_backtracking=False)
    assert torch.equal(a.walk_ids, b.walk_ids)


def test_sample_walks_uniform_starting_vertex(random_graph_50):
    """With many random walks, the starting-vertex distribution should be ~uniform."""
    nw, l, s = 10000, 2, 1
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    data = sample_walks(random_graph_50, nw, l, s, non_backtracking=False)
    starts = data.walk_ids[0, :, 0].numpy()
    n = random_graph_50.x.shape[0]
    counts = np.bincount(starts, minlength=n)
    expected = np.full(n, nw / n)
    chi2, p = stats.chisquare(counts, expected)
    assert p > 0.01, f"starting distribution not uniform (p={p:.4f})"


@pytest.mark.xfail(
    reason=(
        "Upstream bug: in utils.search.sample_walks, the non_backtracking branch "
        "reads `prev_node = walk_ids[i, j-1]` AFTER `current_node` was already set "
        "from that same slot, so `prev_node == current_node` and the filter only "
        "removes self-loops instead of the true predecessor.  See sample_walks "
        "around lines ~265-270."
    ),
    strict=True,
)
def test_sample_walks_non_backtracking(tiny_graph):
    """When non_backtracking=True, walks should avoid immediate backtracks where possible."""
    nw, l, s = 6, 8, 2
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    data = sample_walks(tiny_graph, nw, l, s, non_backtracking=True)
    ids = data.walk_ids[0]
    nbr = get_neighbor_dict(tiny_graph)
    if isinstance(nbr, dict) is False:
        nbr = tiny_graph._neighbor_dict
    for i in range(nw):
        for j in range(2, l):
            cur = int(ids[i, j - 1])
            prev = int(ids[i, j - 2])
            nxt = int(ids[i, j])
            cur_nbrs = nbr[cur] if isinstance(nbr, dict) else tiny_graph._neighbor_dict[cur]
            if len([x for x in cur_nbrs if x != prev]) > 0:
                assert nxt != prev


def test_sample_walks_non_backtracking_actual_behavior(tiny_graph):
    """Document the actual behavior of the (buggy) ``non_backtracking`` flag.

    Because of the bug above, ``non_backtracking=True`` only blocks the random
    walk from emitting a self-loop transition.  This test checks that no walk
    contains ``v -> v`` consecutive duplicates whenever ``v`` has at least one
    neighbor different from itself.  The pentane graph has no self-loops, so
    in practice both modes are equivalent.
    """
    nw, l, s = 6, 8, 2
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    data = sample_walks(tiny_graph, nw, l, s, non_backtracking=True)
    ids = data.walk_ids[0]
    # On a graph with no self-loops, consecutive duplicates should not occur.
    for i in range(nw):
        for j in range(1, l):
            assert int(ids[i, j]) != int(ids[i, j - 1])


# ---------------------------------------------------------------------------
# sample_walks_mdlr / sample_walks_rum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [sample_walks_mdlr, sample_walks_rum])
def test_sample_walks_variants_shape_and_validity(tiny_graph, fn):
    nw, l, s = 4, 6, 2
    data = fn(tiny_graph.clone(), nw, l, s, non_backtracking=False)
    assert data.walk_emb.shape == (nw, l)
    assert data.walk_ids.shape == (1, nw, l)
    assert data.walk_anonym.shape == (nw, l)
    n = data.x.shape[0]
    ids = data.walk_ids[0]
    assert ((ids >= 0) & (ids < n)).all()
    # anonym labels are non-negative and within range [0, l)
    assert (data.walk_anonym >= 0).all()
    assert (data.walk_anonym < l).all()


def test_sample_walks_anonym_consistency(tiny_graph):
    """Anonymized labels must reflect the order of first appearance."""
    nw, l, s = 4, 6, 2
    random.seed(1); torch.manual_seed(1); np.random.seed(1)
    data = sample_walks_mdlr(tiny_graph.clone(), nw, l, s, non_backtracking=False)
    for i in range(nw):
        seen = {}
        for j in range(l):
            v = int(data.walk_ids[0, i, j])
            if v not in seen:
                seen[v] = len(seen)
            assert int(data.walk_anonym[i, j]) == seen[v]


@pytest.mark.parametrize("fn", [sample_walks_mdlr, sample_walks_rum])
def test_sample_walks_variants_deterministic(tiny_graph, fn):
    nw, l, s = 4, 6, 2
    random.seed(7); torch.manual_seed(7); np.random.seed(7)
    a = fn(tiny_graph.clone(), nw, l, s, non_backtracking=False)
    random.seed(7); torch.manual_seed(7); np.random.seed(7)
    b = fn(tiny_graph.clone(), nw, l, s, non_backtracking=False)
    assert torch.equal(a.walk_ids, b.walk_ids)


# ---------------------------------------------------------------------------
# adaptive variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [
    sample_walks_adaptive,
    sample_walks_mdlr_adaptive,
    sample_walks_rum_adaptive,
])
def test_sample_walks_adaptive_variants(tiny_graph, fn):
    nw, l, s = 3, 4, 2
    max_len = 8
    vocab = _vocab_for(tiny_graph)
    # the adaptive samplers index into x_emb but pad with vocab['PAD'];
    # we set max_len > l so the trailing slots stay padded.
    data = fn(tiny_graph.clone(), nw, l, s, False, max_len, vocab)
    assert data.walk_emb.shape == (nw, max_len)
    assert data.walk_ids.shape == (1, nw, max_len)
    assert data.lengths.shape == (nw,)
    # actual walk length is l for all walks
    assert (data.lengths == l).all()
    # padded positions should hold -1 in walk_ids
    if max_len > l:
        assert (data.walk_ids[0, :, l:] == -1).all()


# ---------------------------------------------------------------------------
# BFS / DFS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", [sample_bfs, sample_dfs])
def test_sample_bfs_dfs_shapes(tiny_graph, fn):
    nw, s, max_len = 3, 2, 8
    vocab = _vocab_for(tiny_graph)
    data = fn(tiny_graph.clone(), nw, s, max_len, vocab)
    assert data.walk_emb.shape == (nw, max_len)
    assert data.walk_ids.shape == (1, nw, max_len)
    assert data.walk_pe.shape == (nw, max_len, s)
    assert data.lengths.shape == (nw,)
    n = data.x.shape[0]
    # all visited node ids should be within range
    visited_mask = data.walk_ids[0] != -1
    visited = data.walk_ids[0][visited_mask]
    assert ((visited >= 0) & (visited < n)).all()
    # length matches number of valid (non -1) ids in each row
    for i in range(nw):
        assert int(data.lengths[i]) == int(visited_mask[i].sum())


@pytest.mark.parametrize("fn", [sample_bfs, sample_dfs])
def test_sample_bfs_dfs_deterministic(tiny_graph, fn):
    nw, s, max_len = 3, 2, 8
    vocab = _vocab_for(tiny_graph)
    random.seed(5); torch.manual_seed(5); np.random.seed(5)
    a = fn(tiny_graph.clone(), nw, s, max_len, vocab)
    random.seed(5); torch.manual_seed(5); np.random.seed(5)
    b = fn(tiny_graph.clone(), nw, s, max_len, vocab)
    assert torch.equal(a.walk_ids, b.walk_ids)


@pytest.mark.parametrize("fn", [sample_bfs, sample_dfs])
def test_sample_bfs_dfs_visits_unique_nodes(tiny_graph, fn):
    """BFS/DFS without revisiting must produce only unique node ids per walk."""
    nw, s, max_len = 4, 2, 16
    vocab = _vocab_for(tiny_graph)
    data = fn(tiny_graph.clone(), nw, s, max_len, vocab)
    for i in range(nw):
        ids = [int(v) for v in data.walk_ids[0, i] if int(v) != -1]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# uniform-starting-vertex test for adaptive RW
# ---------------------------------------------------------------------------
def test_adaptive_walk_starting_vertex_uniform(random_graph_50):
    nw, l, s = 10000, 2, 1
    max_len = 4
    vocab = {'PAD': int(random_graph_50.x_emb.max().item()) + 1}
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    data = sample_walks_adaptive(random_graph_50, nw, l, s, False, max_len, vocab)
    starts = data.walk_ids[0, :, 0].numpy()
    n = random_graph_50.x.shape[0]
    counts = np.bincount(starts, minlength=n)
    expected = np.full(n, nw / n)
    chi2, p = stats.chisquare(counts, expected)
    assert p > 0.01, f"adaptive RW start distribution not uniform (p={p:.4f})"


# ---------------------------------------------------------------------------
# dfs_edges
# ---------------------------------------------------------------------------
def test_dfs_edges_returns_tree(tiny_graph):
    """A DFS tree on a connected graph with n nodes has n-1 edges."""
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    edges = dfs_edges(tiny_graph)
    n = tiny_graph.x.shape[0]
    # tiny_graph (pentane) is connected
    assert len(edges) == n - 1
    # all edges are valid graph edges
    g_edges = _edges_set(tiny_graph)
    for u, v in edges:
        assert (u, v) in g_edges or (v, u) in g_edges
