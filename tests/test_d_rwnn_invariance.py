"""Invariance and counterexample tests for the d-RWNN (Variant A, PE-append).

Implements the test plan from review/d_rwnn_spec.md sections 4 and 5:

1. ``test_se3_invariance_of_distances``     -- pairwise distances are E(3)-invariant.
2. ``test_se3_invariance_of_d_rwnn_walk_pe`` -- the walk_pe block built from
   RBF-expanded distances + dense bond features is element-equal across random
   3D rigid motions when the walk indices are kept fixed.
3. ``test_permutation_equivariance_in_expectation`` -- the empirical
   distribution of atom-token bags per walk is permutation-invariant in
   expectation (chi^2-tested).
4. ``test_preprocessing_runtime_baseline_vs_d_rwnn`` -- soft assertion that
   ``qm9_to_data(add_distances=True)`` is at most 5x slower than the QM9
   loader (acts as the "baseline" preprocessing here).
5. ``test_memory_baseline_vs_d_rwnn`` -- soft assertion that the per-graph
   memory growth induced by the (N, N) distance matrix stays under 10x.
6. ``test_zian_li_counterexample`` -- the two complementary 6-subsets of a
   regular icosahedron have identical pairwise-distance multisets (Cor. 5.1),
   and d-RWNN's mean output is therefore equal under L^2; a positive control
   verifies that perturbing one cloud breaks the equality.
7. ``test_chirality_failure`` (bonus) -- a chiral molecule and its mirror
   image yield identical d-RWNN distance fingerprints, demonstrating the
   documented chirality blindness of distance-only models.

Run::

    source /home/snirhordan/miniconda3/etc/profile.d/conda.sh && conda activate rwnn
    cd /home/snirhordan/ito/RandomSearchNNs
    python -m pytest tests/test_d_rwnn_invariance.py -v --tb=short
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.qm9 import (  # noqa: E402
    RBFExpansion,
    build_qm9_vocab,
    load_qm9,
    qm9_to_data,
)
from generation.utils import get_neighbor_dict, mol2graph  # noqa: E402
from models.rwnn import RWNN  # noqa: E402
from quickstart.train_qm9 import build_add_edge_feat, edge_attr_to_dense  # noqa: E402
from utils.search import sample_walks  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random 3D rotation via QR decomposition."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1) by absorbing the sign of diag(R) into Q.
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _make_methane_data(vocab: dict | None = None) -> Data:
    """Build a tiny methane-like 5-atom molecule (1 C + 4 H, tetrahedral)."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.089, 1.089, 1.089],
            [-1.089, -1.089, 1.089],
            [-1.089, 1.089, -1.089],
            [1.089, -1.089, -1.089],
        ],
        dtype=torch.float32,
    )
    # Atom types: C, H, H, H, H
    z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    # Edges: C bonded to all four Hs (8 directed)
    src = [0, 0, 0, 0, 1, 2, 3, 4]
    dst = [1, 2, 3, 4, 0, 0, 0, 0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float)
    x = torch.zeros((5, 9), dtype=torch.float)
    if vocab is None:
        vocab = {"C": 0, "H": 1, "PAD": 2}
    sym = {6: "C", 1: "H"}
    x_emb = torch.tensor([vocab[sym[int(zi)]] for zi in z.tolist()], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.pos = pos
    data.z = z
    data.x_emb = x_emb
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    data.distances = diff.norm(dim=-1)
    data = get_neighbor_dict(data)
    return data


def _build_geometric_data(pos: np.ndarray, labels: np.ndarray, vocab: dict) -> Data:
    """Build a fully-connected geometric graph from a 3D point cloud."""
    pos_t = torch.tensor(pos, dtype=torch.float32)
    n = pos_t.size(0)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float)
    x = torch.zeros((n, 9), dtype=torch.float)
    z = torch.tensor([6] * n, dtype=torch.long)  # all carbons
    x_emb = torch.tensor([vocab["C"]] * n, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.pos = pos_t
    data.z = z
    data.x_emb = x_emb
    diff = pos_t.unsqueeze(0) - pos_t.unsqueeze(1)
    data.distances = diff.norm(dim=-1)
    data = get_neighbor_dict(data)
    return data


def _icosahedron_subsets() -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the two complementary 6-subsets of a regular icosahedron.

    Reference: review/d_rwnn_spec.md Section 5.2; Zian Li et al. NeurIPS 2023
    (arXiv:2302.05743), Appendix A.1, Figure 1.
    """
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    icosahedron = np.array(
        [
            [0.0, 1.0, phi],
            [0.0, -1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [-1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    A_plus = icosahedron[[0, 3, 4, 7, 8, 11]]
    A_minus = icosahedron[[1, 2, 5, 6, 9, 10]]
    vocab = {"C": 0, "PAD": 1}
    return A_plus, A_minus, vocab


def _pe_in_dim_for_walk(s: int, K: int, mol_edge_dim: int = 0) -> int:
    """Return the pe_in_dim for ``sample_walks`` walk_pe layout.

    Layout: [encoding_repeat (s) | encoding_edge (s) | RBF (K) | mol_edge (3*mol_edge_dim)].
    """
    return 2 * s + K + 3 * mol_edge_dim


# ---------------------------------------------------------------------------
# Test 1 -- Distances are SE(3)-invariant
# ---------------------------------------------------------------------------


def test_se3_invariance_of_distances():
    """Pairwise Euclidean distances must be invariant under rigid motions."""
    data = _make_methane_data()
    pos = data.pos
    diff0 = pos.unsqueeze(0) - pos.unsqueeze(1)
    D0 = diff0.norm(dim=-1)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for k in range(10):
        R = _random_rotation(rng)
        t = rng.standard_normal(3) * 5.0
        new_pos_np = pos.numpy() @ R.T + t
        new_pos = torch.tensor(new_pos_np, dtype=torch.float32)
        diff1 = new_pos.unsqueeze(0) - new_pos.unsqueeze(1)
        D1 = diff1.norm(dim=-1)
        err = float((D0 - D1).abs().max().item())
        max_err = max(max_err, err)
        assert torch.allclose(D0, D1, atol=1e-5), (
            f"Distances not invariant: max-err={err:.3e}"
        )
    print(f"[se3 distances] max distance error after 10 random rigid motions: "
          f"{max_err:.3e}")


# ---------------------------------------------------------------------------
# Test 2 -- d-RWNN walk_pe is SE(3)-invariant when walk indices are pinned
# ---------------------------------------------------------------------------


def test_se3_invariance_of_d_rwnn_walk_pe():
    """RBF(distance) + bond-feature blocks of walk_pe are unchanged under rigid motions.

    Strategy: seed the RNG identically for both forward calls so that the
    sampled walk indices match exactly. The only feature that depends on the
    geometry is the per-step distance bin, which must be invariant.
    """
    data = _make_methane_data()
    rbf = RBFExpansion(K=16, cutoff=5.0)

    nw, l, s = 8, 6, 2
    rng = np.random.default_rng(1)
    max_err_pe = 0.0
    for k in range(8):
        # 1) sample walk_pe under the original geometry
        _seed_all(2024 + k)
        d0 = data.clone()
        # important: clone wipes the cached neighbor dict; re-build it.
        d0._neighbor_dict = data._neighbor_dict
        ef0 = build_add_edge_feat(d0, distances=1, mol_edge_feat=1, rbf=rbf)
        out0 = sample_walks(d0, nw, l, s, non_backtracking=False, add_edge_feat=ef0)
        pe0 = out0.walk_pe.clone()
        ids0 = out0.walk_ids.clone()

        # 2) apply rigid motion, recompute distances, re-seed identically
        R = _random_rotation(rng)
        t = rng.standard_normal(3) * 3.0
        new_pos = data.pos.numpy() @ R.T + t
        d1 = data.clone()
        d1._neighbor_dict = data._neighbor_dict
        d1.pos = torch.tensor(new_pos, dtype=torch.float32)
        diff = d1.pos.unsqueeze(0) - d1.pos.unsqueeze(1)
        d1.distances = diff.norm(dim=-1)

        _seed_all(2024 + k)  # identical RNG state -> identical walks
        ef1 = build_add_edge_feat(d1, distances=1, mol_edge_feat=1, rbf=rbf)
        out1 = sample_walks(d1, nw, l, s, non_backtracking=False, add_edge_feat=ef1)
        pe1 = out1.walk_pe
        ids1 = out1.walk_ids

        # walks must match exactly
        assert torch.equal(ids0, ids1), (
            f"walk indices mismatched at k={k}: rigid motion broke the seeded sampler"
        )
        err = float((pe0 - pe1).abs().max().item())
        max_err_pe = max(max_err_pe, err)
        assert torch.allclose(pe0, pe1, atol=1e-5), (
            f"walk_pe not SE(3)-invariant at iteration {k}: max-err={err:.3e}"
        )
    print(f"[se3 walk_pe] max walk_pe abs error across 8 random rigid motions: "
          f"{max_err_pe:.3e}")


# ---------------------------------------------------------------------------
# Test 3 -- Permutation equivariance in expectation
# ---------------------------------------------------------------------------


def test_permutation_equivariance_in_expectation():
    """Empirical distribution over per-walk atom-token multisets is
    permutation-invariant in expectation (chi^2 tested).

    The d-RWNN sampler chooses anchors uniformly at random over V and steps
    uniformly to neighbors; both decisions are functions of the unordered
    neighbor structure, so the distribution of (atom-token-bag) per walk
    must be invariant to a relabelling of node indices.

    To make the chi^2 test informative we need a graph with several distinct
    atom types so that the per-walk bag has non-trivial structure. We use a
    4-acetic-acid-like graph: ``[C, O, O, N, S]`` on a 5-cycle, fully
    connected.
    """
    # Build a 5-node fully connected graph with distinct atom labels.
    vocab = {"C": 0, "O": 1, "N": 2, "S": 3, "P": 4, "PAD": 5}
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.5, 1.5, 0.0],
            [-0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    n = pos.shape[0]
    # Atom types: distinct
    z = np.array([6, 8, 7, 16, 15], dtype=np.int64)
    sym_for_z = {6: "C", 8: "O", 7: "N", 16: "S", 15: "P"}
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float)
    x = torch.zeros((n, 9), dtype=torch.float)
    x_emb = torch.tensor([vocab[sym_for_z[int(zi)]] for zi in z], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.pos = torch.tensor(pos, dtype=torch.float32)
    data.z = torch.tensor(z, dtype=torch.long)
    data.x_emb = x_emb
    diff = data.pos.unsqueeze(0) - data.pos.unsqueeze(1)
    data.distances = diff.norm(dim=-1)
    data = get_neighbor_dict(data)

    nw, l, s = 200, 4, 2

    # ordering A: identity (fresh seed)
    _seed_all(11)
    out_a = sample_walks(_clone_for_walks(data), nw, l, s, non_backtracking=False)

    # ordering B: random permutation (fresh, *different* seed -- the spec
    # explicitly requires that we do NOT reuse the same RNG state)
    rng = np.random.default_rng(7)
    perm = rng.permutation(data.x.size(0))
    data_perm = _permute_data(data, perm)
    _seed_all(13)
    out_b = sample_walks(_clone_for_walks(data_perm), nw, l, s, non_backtracking=False)

    # Build empirical multiset distributions over walks: for each walk, take
    # the sorted bag of atom-token ids and count.
    bag_a = Counter()
    for i in range(nw):
        key = tuple(sorted(out_a.walk_emb[i].tolist()))
        bag_a[key] += 1
    bag_b = Counter()
    for i in range(nw):
        key = tuple(sorted(out_b.walk_emb[i].tolist()))
        bag_b[key] += 1

    # Build aligned count vectors over the union of keys
    keys = sorted(set(bag_a.keys()) | set(bag_b.keys()))
    obs_a = np.array([bag_a.get(k, 0) for k in keys], dtype=np.float64)
    obs_b = np.array([bag_b.get(k, 0) for k in keys], dtype=np.float64)
    # smooth by 0.5 to avoid zero expected counts ruining chi^2
    smooth = 0.5
    obs_a += smooth
    obs_b += smooth
    expected_pool = (obs_a + obs_b) / 2.0
    chi2_a = float(((obs_a - expected_pool) ** 2 / expected_pool).sum())
    chi2_b = float(((obs_b - expected_pool) ** 2 / expected_pool).sum())
    chi2 = chi2_a + chi2_b
    df = max(len(keys) - 1, 1)
    p = float(stats.chi2.sf(chi2, df))
    print(f"[perm-invariance] chi2={chi2:.3f}, df={df}, p={p:.4f}, "
          f"|keys|={len(keys)}")
    assert p > 0.01, (
        f"Empirical walk-bag distributions differ between orderings "
        f"(chi2={chi2:.3f}, df={df}, p={p:.4f})"
    )


def _permute_data(data: Data, perm) -> Data:
    """Apply a node-index permutation ``perm`` (new[i] = old[perm[i]]) to a
    Data object: rows of x/x_emb/pos/distances are reindexed and the
    edge_index entries are remapped through the inverse permutation.
    """
    perm = np.asarray(perm, dtype=np.int64)
    inv = np.argsort(perm)
    new_data = Data()
    n = data.x.size(0)
    new_data.x = data.x[perm].clone()
    new_data.x_emb = data.x_emb[perm].clone()
    new_data.pos = data.pos[perm].clone()
    if hasattr(data, "z") and data.z is not None:
        new_data.z = data.z[perm].clone()
    if hasattr(data, "distances") and data.distances is not None:
        new_data.distances = data.distances[perm][:, perm].clone()
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr.clone()
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    new_src = inv[src]
    new_dst = inv[dst]
    new_data.edge_index = torch.tensor(np.stack([new_src, new_dst], axis=0),
                                       dtype=torch.long)
    new_data = get_neighbor_dict(new_data)
    return new_data


def _clone_for_walks(data: Data) -> Data:
    """Shallow-copy the fields needed by ``sample_walks`` while preserving
    the cached neighbor dictionary (so we don't pay for re-building it)."""
    new_data = Data()
    new_data.x = data.x
    new_data.x_emb = data.x_emb
    new_data.edge_index = data.edge_index
    if hasattr(data, "pos"):
        new_data.pos = data.pos
    if hasattr(data, "distances"):
        new_data.distances = data.distances
    if hasattr(data, "edge_attr"):
        new_data.edge_attr = data.edge_attr
    if hasattr(data, "_neighbor_dict"):
        new_data._neighbor_dict = data._neighbor_dict
    else:
        new_data = get_neighbor_dict(new_data)
    return new_data


# ---------------------------------------------------------------------------
# Test 4 -- preprocessing runtime, baseline vs d-RWNN
# ---------------------------------------------------------------------------


def test_preprocessing_runtime_baseline_vs_d_rwnn():
    """Measure mean wall-clock for ``qm9_to_data`` with vs. without
    distance + edge-attr augmentation across 100 QM9 molecules.

    Soft assertion: d-RWNN preprocessing must be at most 5x slower than the
    minimum-feature baseline path. This is a sanity check that the (N, N)
    distance computation does not introduce pathological overhead.
    """
    ds = load_qm9(root="./data/qm9")
    vocab = build_qm9_vocab(ds[:200], tokenizer=None)
    samples = [ds[i] for i in range(100)]

    # Warm-up
    for s in samples[:5]:
        qm9_to_data(s, vocab=vocab, add_distances=False, add_edge_attr=False)
        qm9_to_data(s, vocab=vocab, add_distances=True, add_edge_attr=True)

    t0 = time.perf_counter()
    for s in samples:
        qm9_to_data(s, vocab=vocab, add_distances=False, add_edge_attr=False)
    t_baseline = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in samples:
        qm9_to_data(s, vocab=vocab, add_distances=True, add_edge_attr=True)
    t_drwnn = time.perf_counter() - t0

    ratio = t_drwnn / max(t_baseline, 1e-9)
    print(f"[runtime] baseline (no dist/no edge_attr): {t_baseline*1000:.2f} ms "
          f"({t_baseline*1000/100:.3f} ms/molecule)")
    print(f"[runtime] d-RWNN  (dist+edge_attr)       : {t_drwnn*1000:.2f} ms "
          f"({t_drwnn*1000/100:.3f} ms/molecule)")
    print(f"[runtime] ratio d-RWNN / baseline         : {ratio:.3f}x")
    assert ratio <= 5.0, (
        f"d-RWNN preprocessing is {ratio:.3f}x slower than baseline "
        f"(expected <= 5x)"
    )


# ---------------------------------------------------------------------------
# Test 5 -- per-graph memory, baseline vs d-RWNN
# ---------------------------------------------------------------------------


def _data_byte_size(data: Data) -> int:
    """Sum element-bytes across all tensor fields on a ``Data`` object."""
    total = 0
    for key, value in data:
        if isinstance(value, torch.Tensor):
            total += value.element_size() * value.numel()
    return total


def test_memory_baseline_vs_d_rwnn():
    """Per-graph memory ratio (d-RWNN / baseline) must stay under 10x."""
    ds = load_qm9(root="./data/qm9")
    vocab = build_qm9_vocab(ds[:200], tokenizer=None)
    samples = [ds[i] for i in range(100)]

    base_bytes = []
    drwnn_bytes = []
    for s in samples:
        d_base = qm9_to_data(s, vocab=vocab, add_distances=False, add_edge_attr=False)
        d_full = qm9_to_data(s, vocab=vocab, add_distances=True, add_edge_attr=True)
        base_bytes.append(_data_byte_size(d_base))
        drwnn_bytes.append(_data_byte_size(d_full))

    mean_base = float(np.mean(base_bytes))
    mean_drwnn = float(np.mean(drwnn_bytes))
    ratio = mean_drwnn / max(mean_base, 1)
    print(f"[memory] mean base bytes/graph : {mean_base:.1f}")
    print(f"[memory] mean d-RWNN bytes/g.  : {mean_drwnn:.1f}")
    print(f"[memory] ratio d-RWNN / base   : {ratio:.3f}x")
    assert ratio <= 10.0, (
        f"d-RWNN per-graph memory is {ratio:.3f}x baseline (expected <= 10x)"
    )


# ---------------------------------------------------------------------------
# Test 6 -- Zian Li icosahedron 6/6 counterexample
# ---------------------------------------------------------------------------


def _build_d_rwnn_data_with_walks(
    pos: np.ndarray,
    vocab: dict,
    nw: int,
    l: int,
    s: int,
    rbf: RBFExpansion,
    seed: int,
) -> Data:
    """Construct a fully-connected geometric graph and sample walks on it."""
    data = _build_geometric_data(pos, np.zeros(pos.shape[0], dtype=np.int64), vocab)
    ef = build_add_edge_feat(data, distances=1, mol_edge_feat=0, rbf=rbf)
    _seed_all(seed)
    data = sample_walks(data, nw, l, s, non_backtracking=False, add_edge_feat=ef)
    return data


def _wrap_for_rwnn(data: Data) -> Data:
    """Bundle walk_emb / walk_ids / walk_pe into a Data with the layout
    expected by ``RWNN.forward``.

    The model expects ``walk_ids`` of shape ``(b, nw, l)`` (the sampler
    already produces ``(1, nw, l)``) and ``walk_emb``/``walk_pe`` for a
    *single* graph (shapes ``(nw, l)`` and ``(nw, l, pe_in_dim)``); see
    ``models/rwnn.py:RWNN.forward`` lines 36-57.
    """
    batched = Data()
    batched.walk_emb = data.walk_emb           # (nw, l)
    batched.walk_ids = data.walk_ids           # (1, nw, l)
    batched.walk_pe = data.walk_pe             # (nw, l, pe_in_dim)
    return batched


def test_zian_li_counterexample():
    """The two complementary 6-subsets of a regular icosahedron have identical
    pairwise-distance multisets (Cor. 5.1). d-RWNN with distance-only PE must
    therefore produce the same expected output on both clouds.

    Includes a positive control: perturbing one cloud breaks distance
    equality and the assertion correctly fails.
    """
    A_plus, A_minus, vocab = _icosahedron_subsets()

    # 1) Verify the precondition: same sorted distance multiset.
    def _sorted_dist(p):
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        return np.sort(d[np.triu_indices_from(d, k=1)])

    dA = _sorted_dist(A_plus)
    dB = _sorted_dist(A_minus)
    multiset_diff = float(np.max(np.abs(dA - dB)))
    print(f"[zian-li] sorted pairwise-distance L_inf diff: {multiset_diff:.3e}")
    assert multiset_diff < 1e-9, (
        "Precondition failed: sorted distance multisets differ"
    )

    # 2) Build a small d-RWNN model and average the output over many seeds.
    K = 16
    s = 2
    nw = 1024
    l = 4
    M = 32  # number of seed sweeps
    pe_in_dim = _pe_in_dim_for_walk(s=s, K=K, mol_edge_dim=0)
    n_emb = max(vocab.values()) + 1
    rbf = RBFExpansion(K=K, cutoff=5.0)

    # Construct one model and freeze its weights -- both forward calls share weights.
    torch.manual_seed(0)
    model = RWNN(
        pe_in_dim=pe_in_dim,
        pe_out_dim=8,
        hid_dim=16,
        out_dim=4,
        num_layers=1,
        n_emb=n_emb,
        reduce="mean",
    )
    model.eval()

    outs_A, outs_B = [], []
    with torch.no_grad():
        for m in range(M):
            seed = 4242 + m
            data_A = _build_d_rwnn_data_with_walks(A_plus, vocab, nw, l, s, rbf, seed)
            data_B = _build_d_rwnn_data_with_walks(A_minus, vocab, nw, l, s, rbf, seed)
            outs_A.append(model(_wrap_for_rwnn(data_A)).cpu())
            outs_B.append(model(_wrap_for_rwnn(data_B)).cpu())

    mean_A = torch.stack(outs_A).mean(dim=0)
    mean_B = torch.stack(outs_B).mean(dim=0)
    l2 = float((mean_A - mean_B).norm().item())
    linf = float((mean_A - mean_B).abs().max().item())
    print(f"[zian-li] mean output A: {mean_A.flatten().tolist()}")
    print(f"[zian-li] mean output B: {mean_B.flatten().tolist()}")
    print(f"[zian-li] L_inf diff means : {linf:.3e}")
    print(f"[zian-li] L_2  diff means  : {l2:.3e}")
    # Since the per-seed walks see literally identical token streams as
    # multisets (because both clouds have the same node-by-node sorted
    # distance multiset and a single atom label), the empirical means must be
    # very close. We allow a modest atol to accommodate residual differences
    # in walk-step ordering that cancel on average over M seeds.
    assert torch.allclose(mean_A, mean_B, atol=2e-2), (
        f"d-RWNN distinguishes the Zian Li 6/6 split: "
        f"L_inf={linf:.3e}, L_2={l2:.3e} -- this contradicts Cor. 5.1 and "
        f"is a real bug in the implementation."
    )

    # 3) Positive control: perturb cloud B and assert the means now differ.
    # We use a *large* per-coordinate perturbation (scale=2 A) so the model
    # has substantial geometric signal to pick up; the equality test in (2)
    # uses atol=2e-2, so the positive control needs to clear that floor by a
    # comfortable margin to demonstrate non-vacuity.
    rng = np.random.default_rng(0)
    A_perturbed = A_minus + rng.normal(scale=2.0, size=A_minus.shape)
    outs_pert = []
    with torch.no_grad():
        for m in range(M):
            seed = 4242 + m
            data_p = _build_d_rwnn_data_with_walks(A_perturbed, vocab,
                                                   nw, l, s, rbf, seed)
            outs_pert.append(model(_wrap_for_rwnn(data_p)).cpu())
    mean_pert = torch.stack(outs_pert).mean(dim=0)
    pert_diff = float((mean_A - mean_pert).abs().max().item())
    null_diff = float((mean_A - mean_B).abs().max().item())
    print(f"[zian-li] positive control: L_inf(mean_A - mean_perturbed) = "
          f"{pert_diff:.3e}")
    print(f"[zian-li] signal/noise ratio (pert / null) = "
          f"{pert_diff / max(null_diff, 1e-12):.1f}")
    # Positive control: the perturbed-distance signal must dominate the
    # Monte-Carlo noise floor of the equality test by at least 100x.
    assert pert_diff > 100.0 * null_diff, (
        f"Positive control failed: perturbed cloud differs from A by only "
        f"{pert_diff:.3e}, which is not large compared to the null-case noise "
        f"floor of {null_diff:.3e}. The equality test in (2) might be "
        f"vacuous (model output is essentially constant)."
    )


# ---------------------------------------------------------------------------
# Test 7 (bonus) -- chirality blindness
# ---------------------------------------------------------------------------


def test_chirality_failure():
    """A chiral molecule (R-CHFClBr) and its mirror image (S-CHFClBr) have
    identical complete distance matrices, hence d-RWNN cannot tell them apart.
    """
    # Tetrahedral CHFClBr coordinates: C at origin, H/F/Cl/Br on tetrahedral arms.
    pos_R = np.array(
        [
            [0.0, 0.0, 0.0],     # C
            [1.0, 1.0, 1.0],     # H
            [-1.0, -1.0, 1.0],   # F
            [-1.0, 1.0, -1.0],   # Cl
            [1.0, -1.0, -1.0],   # Br
        ],
        dtype=np.float64,
    )
    # Mirror image: reflect across the z=0 plane.
    pos_S = pos_R.copy()
    pos_S[:, 2] = -pos_S[:, 2]

    # Verify: the *labelled* distance matrices are identical (the atom labels
    # ride along under reflection because reflection commutes with permutation
    # of the position rows when applied to all atoms uniformly).
    def _D(p):
        return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    D_R = _D(pos_R)
    D_S = _D(pos_S)
    diff = float(np.max(np.abs(D_R - D_S)))
    print(f"[chirality] |D(R) - D(S)|_inf = {diff:.3e}")
    assert diff < 1e-12, "Chiral pair must share the labelled distance matrix"

    # Now run d-RWNN on both and check the mean outputs match.
    vocab = {"C": 0, "H": 1, "F": 2, "Cl": 3, "Br": 4, "PAD": 5}
    K = 8
    s = 2
    nw = 256
    l = 4
    M = 16
    rbf = RBFExpansion(K=K, cutoff=5.0)

    def _make_chiral_data(pos: np.ndarray) -> Data:
        labels = np.array([vocab[x] for x in ["C", "H", "F", "Cl", "Br"]],
                          dtype=np.int64)
        n = pos.shape[0]
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float)
        x = torch.zeros((n, 9), dtype=torch.float)
        z = torch.tensor([6, 1, 9, 17, 35], dtype=torch.long)
        x_emb = torch.tensor(labels, dtype=torch.long)
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        d.pos = torch.tensor(pos, dtype=torch.float32)
        d.z = z
        d.x_emb = x_emb
        diff_t = d.pos.unsqueeze(0) - d.pos.unsqueeze(1)
        d.distances = diff_t.norm(dim=-1)
        return get_neighbor_dict(d)

    pe_in_dim = _pe_in_dim_for_walk(s=s, K=K, mol_edge_dim=0)
    n_emb = max(vocab.values()) + 1
    torch.manual_seed(0)
    model = RWNN(
        pe_in_dim=pe_in_dim,
        pe_out_dim=8,
        hid_dim=16,
        out_dim=4,
        num_layers=1,
        n_emb=n_emb,
        reduce="mean",
    )
    model.eval()

    outs_R, outs_S = [], []
    with torch.no_grad():
        for m in range(M):
            seed = 7777 + m
            dR = _make_chiral_data(pos_R)
            dS = _make_chiral_data(pos_S)
            efR = build_add_edge_feat(dR, distances=1, mol_edge_feat=0, rbf=rbf)
            efS = build_add_edge_feat(dS, distances=1, mol_edge_feat=0, rbf=rbf)
            _seed_all(seed)
            dR = sample_walks(dR, nw, l, s, non_backtracking=False, add_edge_feat=efR)
            _seed_all(seed)
            dS = sample_walks(dS, nw, l, s, non_backtracking=False, add_edge_feat=efS)
            outs_R.append(model(_wrap_for_rwnn(dR)).cpu())
            outs_S.append(model(_wrap_for_rwnn(dS)).cpu())

    mean_R = torch.stack(outs_R).mean(dim=0)
    mean_S = torch.stack(outs_S).mean(dim=0)
    err = float((mean_R - mean_S).abs().max().item())
    print(f"[chirality] L_inf(mean_R - mean_S) = {err:.3e}")
    # Same RNG, identical distance matrix => identical token streams =>
    # outputs match to the floating-point precision of the model forward.
    assert torch.allclose(mean_R, mean_S, atol=1e-5), (
        f"Chirality blindness failed: d-RWNN distinguished R / S "
        f"(diff={err:.3e}); a distance-only model should not."
    )
