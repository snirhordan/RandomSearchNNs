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


def test_sample_walks_non_backtracking_no_self_loops(tiny_graph):
    """``non_backtracking=True`` is intended to forbid self-loop transitions
    only.  On a graph with no self-loops (e.g. molecular pentane), this means
    the flag is a no-op in distribution; in particular consecutive walk
    positions should never duplicate."""
    nw, l, s = 6, 8, 2
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    data = sample_walks(tiny_graph, nw, l, s, non_backtracking=True)
    ids = data.walk_ids[0]
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


# ---------------------------------------------------------------------------
# Dense distance fix for sample_dfs
#
# Before the fix: when DFS pops a node off the stack that is NOT graph-adjacent
# to the previous-in-order node (a "stack-jump"), the per-step edge feature
# (RBF-expanded Euclidean distance + bond type) was left zero. After the fix,
# the distance slice is always populated; bond-type slice remains zero for
# non-bonded pairs (edge_attr_to_dense yields zeros there, by construction).
# ---------------------------------------------------------------------------


def _build_branched_graph_with_distances():
    """Y-shaped 5-node graph that forces DFS stack-jumps.

    Topology:
        0 -- 1
        0 -- 2
        0 -- 3
        3 -- 4

    DFS from 0 pushes [1, 2, 3] in some order. After visiting 3 (and its
    subtree containing 4), the stack still holds 1 and 2. Popping one of
    them produces a step where `prev_in_order` (deep inside the 3-subtree)
    is NOT bonded to the new node. That's the jump path the fix addresses.
    """
    from torch_geometric.data import Data
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 3, 4],
        [1, 0, 2, 0, 3, 0, 4, 3],
    ], dtype=torch.long)
    x_emb = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
    ])
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    distances = diff.norm(dim=-1)
    # x is a dummy float tensor — matches QM9WalkDataset.__getitem__ usage of d.x
    data = Data(
        x=x_emb.float().unsqueeze(1),
        edge_index=edge_index,
        x_emb=x_emb,
    )
    data.pos = pos
    data.distances = distances
    return data


def test_dfs_dense_distances_synthetic_branched(rng_seed):
    """Synthetic Y-graph: every non-first DFS step has non-zero distance dims.

    Before the fix, DFS stack-jumps left the distance slice as zeros. After
    the fix, the RBF-expanded Euclidean distance is always populated.
    """
    from quickstart.train_qm9 import build_add_edge_feat
    from generation.qm9 import RBFExpansion

    data = _build_branched_graph_with_distances()
    rbf = RBFExpansion(K=16, cutoff=5.0)
    add_ef = build_add_edge_feat(data, distances=1, mol_edge_feat=0, rbf=rbf)
    assert add_ef.shape == (5, 5, 16)

    nw, s, max_len = 8, 2, 5
    vocab = {'PAD': int(data.x_emb.max().item()) + 1}
    random.seed(rng_seed); torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    result = sample_dfs(data.clone(), nw, s, max_len, vocab, add_edge_feat=add_ef)

    # walk_pe shape: (nw, max_len, s + 16) — edge_encoding(s) + RBF_dist(16)
    walk_pe = result.walk_pe
    assert walk_pe.shape == (nw, max_len, s + 16)
    dist_part = walk_pe[..., s:]  # (nw, max_len, 16)

    # Sanity: this DFS layout should yield at least one walk with a jump
    # (some walk of length >= 3 visits a node whose predecessor in DFS order
    # is not graph-adjacent). We don't assert which walk has the jump; we
    # assert the distance signal is populated everywhere it should be.
    for i in range(nw):
        length = int(result.lengths[i])
        for j in range(1, length):
            assert dist_part[i, j].abs().sum() > 0, (
                f"walk {i} step {j}: distance dims are all-zero; "
                f"DFS jump should populate them after the fix"
            )


def test_dfs_dense_distances_qm9_integration(rng_seed):
    """Real QM9 molecule: every non-first DFS step has non-zero distance dims."""
    cache_path = REPO_ROOT / "data/qm9/qm9_d_rwnn_cache/mols_gap.pt"
    if not cache_path.exists():
        pytest.skip(f"QM9 preprocessed cache not found at {cache_path}")
    mols = torch.load(str(cache_path), weights_only=False)
    # Pick a moderately complex molecule (>= 8 atoms) to ensure DFS branching
    chosen = next((m for m in mols if m.x.shape[0] >= 8), None)
    if chosen is None:
        pytest.skip("no QM9 molecule with >=8 atoms found in cache")
    data = chosen.clone()

    from quickstart.train_qm9 import build_add_edge_feat
    from generation.qm9 import RBFExpansion

    rbf = RBFExpansion(K=16, cutoff=5.0)
    add_ef = build_add_edge_feat(data, distances=1, mol_edge_feat=1, rbf=rbf)
    # 16 (RBF) + 3 (bond) = 19
    assert add_ef.shape[-1] == 19

    nw, s, max_len = 8, 2, int(data.x.shape[0]) + 2
    vocab = {'PAD': int(data.x_emb.max().item()) + 1}
    random.seed(rng_seed); torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    result = sample_dfs(data, nw, s, max_len, vocab, add_edge_feat=add_ef)

    walk_pe = result.walk_pe
    assert walk_pe.shape == (nw, max_len, s + 19)
    # First s dims = edge encoding; next 16 = RBF distances; last 3 = bond type
    dist_part = walk_pe[..., s:s + 16]

    for i in range(nw):
        length = int(result.lengths[i])
        for j in range(1, length):
            assert dist_part[i, j].abs().sum() > 0, (
                f"walk {i} step {j} on real QM9 mol has all-zero distance dims; "
                f"every DFS step should carry Euclidean distance signal"
            )


def test_dfs_dense_distances_backward_nonvanishing_grads(rng_seed):
    """Forward + backward + non-vanishing gradient check (||grad|| > 1e-6).

    Uses two synthetic Y-graphs to force DFS jumps, runs RSNN_LSTM_Reg, and
    verifies every learnable parameter receives gradient signal above 1e-6.
    """
    from quickstart.train_qm9 import build_add_edge_feat, RSNN_LSTM_Reg
    from generation.qm9 import RBFExpansion
    from torch_geometric.data import Batch

    random.seed(rng_seed); torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    rbf = RBFExpansion(K=16, cutoff=5.0)
    nw, s, max_len = 4, 2, 5
    vocab = {'PAD': 5}

    samples = []
    for _ in range(2):
        m = _build_branched_graph_with_distances()
        add_ef = build_add_edge_feat(m, distances=1, mol_edge_feat=0, rbf=rbf)
        out = sample_dfs(m.clone(), nw, s, max_len, vocab, add_edge_feat=add_ef)
        for k in ("distances", "pos"):
            if hasattr(out, k):
                delattr(out, k)
        samples.append(out)

    batch = Batch.from_data_list(samples)

    pe_in_dim = s + 16  # edge encoding + RBF distance
    model = RSNN_LSTM_Reg(
        pe_in_dim=pe_in_dim, pe_out_dim=16, hid_dim=32, out_dim=1,
        num_layers=2, n_emb=6, reduce="sum", dropout=0.0,
    )
    pred = model(batch).squeeze(-1)
    target = torch.tensor([1.0, 1.5])
    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()

    vanishing = []
    for name, p in model.named_parameters():
        if p.grad is None:
            vanishing.append((name, "grad=None"))
        elif p.grad.norm().item() < 1e-6:
            vanishing.append((name, f"||grad||={p.grad.norm().item():.2e}"))
    assert not vanishing, (
        f"Parameters with vanishing gradients (<1e-6): {vanishing}"
    )
