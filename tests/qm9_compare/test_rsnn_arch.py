"""Architectural + permutation tests for ``RSNN_LSTM_Reg``.

Covers three properties of the regression head defined in
``quickstart/train_qm9.py``:

A. Multi-layer instantiation: ``num_layers`` actually allocates that many
   *separate* ``nn.LSTM`` modules with distinct parameter tensors (i.e. the
   custom loop is not silently sharing weights).
B. Forward-pass coverage: each LSTM in ``model.rnn_layers`` is invoked
   exactly once per forward call (verified via forward hooks).
C. Permutation behaviour: with sum/mean graph readout the output is
   invariant to a permutation of the *walks* (rows of ``walk_emb`` etc.),
   but is sensitive to a permutation of *timesteps within a walk* — the
   LSTM is an explicit sequence model, not a set encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Make the upstream repo importable so ``from quickstart.train_qm9 import ...``
# resolves regardless of how pytest is invoked.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quickstart.train_qm9 import RSNN_LSTM_Reg  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed model / batch hyperparameters used across the file.
# ---------------------------------------------------------------------------
PE_IN_DIM = 8
PE_OUT_DIM = 16
HID_DIM = 64
OUT_DIM = 1
N_EMB = 6           # vocab size (incl. PAD)
NW = 3              # walks per graph
MAX_LEN = 5         # walk length (incl. start node)


def _make_model(num_layers: int, reduce: str = "sum") -> RSNN_LSTM_Reg:
    return RSNN_LSTM_Reg(
        pe_in_dim=PE_IN_DIM,
        pe_out_dim=PE_OUT_DIM,
        hid_dim=HID_DIM,
        out_dim=OUT_DIM,
        num_layers=num_layers,
        n_emb=N_EMB,
        reduce=reduce,
    )


def _make_graph(n_nodes: int, seed: int) -> Data:
    """Build one synthetic graph's worth of walk tensors.

    Matches the shapes produced by ``sample_walks_adaptive``:
      walk_emb: (nw, max_len) long           — token ids in [0, N_EMB-1)
      walk_ids: (1, nw, max_len) long        — raw node ids in [0, n_nodes)
      walk_pe : (nw, max_len, PE_IN_DIM) float
      lengths : (nw,) long                   — actual walk length (<= max_len)

    Padding convention: ``walk_ids`` uses -1 for unused positions; we keep
    every walk fully populated (lengths == MAX_LEN) so the test is unaffected
    by ``pack_padded_sequence`` semantics.
    """
    g = torch.Generator().manual_seed(seed)
    walk_emb = torch.randint(0, N_EMB - 1, (NW, MAX_LEN), generator=g)
    # Cover all n_nodes ids so graph_ns = max(walk_ids[i]) is well-defined.
    walk_ids = torch.randint(0, n_nodes, (NW, MAX_LEN), generator=g)
    walk_pe = torch.randn(NW, MAX_LEN, PE_IN_DIM, generator=g)
    lengths = torch.full((NW,), MAX_LEN, dtype=torch.long)

    d = Data()
    d.walk_emb = walk_emb
    d.walk_ids = walk_ids.unsqueeze(0)            # (1, nw, max_len)
    d.walk_pe = walk_pe
    d.lengths = lengths
    d.x = torch.zeros(n_nodes, 1)
    d.num_nodes = n_nodes
    return d


def _make_batch(n_per_graph=(5, 6), seed: int = 0) -> Batch:
    return Batch.from_data_list(
        [_make_graph(n, seed + i) for i, n in enumerate(n_per_graph)]
    )


# ---------------------------------------------------------------------------
# A. num_layers is structural, not just an iteration count.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 2, 3, 4, 6])
def test_num_layers_instantiates_separate_lstms(L):
    model = _make_model(num_layers=L)

    # (A1) Container length matches num_layers exactly.
    assert len(model.rnn_layers) == L, (
        f"expected {L} LSTM modules, got {len(model.rnn_layers)}"
    )

    # (A2) Every entry is a bidirectional ``nn.LSTM``.
    for i, layer in enumerate(model.rnn_layers):
        assert isinstance(layer, nn.LSTM), (
            f"rnn_layers[{i}] is {type(layer).__name__}, not nn.LSTM"
        )
        assert layer.bidirectional, (
            f"rnn_layers[{i}].bidirectional must be True"
        )
        # Each entry must be a single-layer LSTM — the multi-layer stack is
        # an explicit Python loop with scatter-mean in between, not the
        # ``num_layers=`` argument of ``nn.LSTM`` itself.
        assert layer.num_layers == 1, (
            f"rnn_layers[{i}].num_layers must be 1 (got {layer.num_layers})"
        )

    # (A3) Layers do not share parameter tensors. We check the recurrent
    # weight on layer 0 (``weight_hh_l0``) across consecutive layers — if the
    # ModuleList accidentally pointed at one shared LSTM these would alias.
    for i in range(1, L):
        assert (
            id(model.rnn_layers[0].weight_hh_l0)
            != id(model.rnn_layers[i].weight_hh_l0)
        ), f"weight_hh_l0 is shared between layers 0 and {i}"
        assert (
            id(model.rnn_layers[i - 1].weight_hh_l0)
            != id(model.rnn_layers[i].weight_hh_l0)
        ), f"weight_hh_l0 is shared between layers {i - 1} and {i}"
        # Also check the input weight, which has a different shape on
        # layer 0 vs subsequent layers (input_size differs):
        assert (
            id(model.rnn_layers[i - 1].weight_ih_l0)
            != id(model.rnn_layers[i].weight_ih_l0)
        ), f"weight_ih_l0 is shared between layers {i - 1} and {i}"


def test_lstm_block_param_count_scales_with_num_layers():
    """Total LSTM-block parameter count should grow ~linearly with L.

    Layer 0 has input size ``hid_dim + pe_out_dim``; layers >= 1 have input
    size ``2 * hid_dim`` (bidirectional output of the previous layer). The
    ratio P(L) / P(1) should sit between L * (lower-bound layer) / P(1) and
    L * (upper-bound layer) / P(1), which is well below "L exactly" but still
    monotone in L.
    """
    counts = {}
    for L in (1, 2, 3, 4, 6):
        m = _make_model(num_layers=L)
        counts[L] = sum(p.numel() for p in m.rnn_layers.parameters())

    # Strictly increasing.
    keys = sorted(counts)
    for a, b in zip(keys, keys[1:]):
        assert counts[b] > counts[a], (
            f"LSTM-block params did not grow when num_layers went "
            f"{a} -> {b}: {counts[a]} vs {counts[b]}"
        )

    # Roughly linear: the slope per added layer is the cost of one
    # (2*hid_dim -> hid_dim, bidir) LSTM block, which is constant. So
    # (counts[6] - counts[1]) / 5 should be close to (counts[3] - counts[1]) / 2.
    per_layer_5 = (counts[6] - counts[1]) / 5
    per_layer_2 = (counts[3] - counts[1]) / 2
    rel_err = abs(per_layer_5 - per_layer_2) / max(per_layer_2, 1.0)
    assert rel_err < 0.05, (
        f"per-added-layer param cost is not linear: "
        f"slope_5={per_layer_5:.1f}, slope_2={per_layer_2:.1f}, "
        f"rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# B. forward() actually uses every layer (hook-based verification).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_forward_invokes_every_lstm_layer_exactly_once(L):
    model = _make_model(num_layers=L).eval()
    batch = _make_batch(seed=L)

    counts = [0] * L
    handles = []
    for i, layer in enumerate(model.rnn_layers):
        def _hook(_module, _inp, _out, idx=i):
            counts[idx] += 1
        handles.append(layer.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            out = model(batch)
    finally:
        for h in handles:
            h.remove()

    assert out.shape == (2, OUT_DIM), f"unexpected output shape {out.shape}"
    assert all(c == 1 for c in counts), (
        f"each LSTM layer must fire exactly once per forward; got {counts}"
    )


# ---------------------------------------------------------------------------
# C. Permutation properties.
# ---------------------------------------------------------------------------
def _permute_walks(batch: Batch, perm_per_graph) -> Batch:
    """Return a NEW batch where, for each graph i, walks are reordered by
    ``perm_per_graph[i]`` (a permutation of range(NW)).

    walk_emb is (B*NW, max_len) flat over graphs, walk_pe is
    (B*NW, max_len, pe), lengths is (B*NW,) — we permute the rows belonging
    to each graph independently. walk_ids is (B, NW, max_len) — we permute
    along dim=1.
    """
    B = batch.walk_ids.shape[0]
    new_walk_emb = batch.walk_emb.clone()
    new_walk_ids = batch.walk_ids.clone()
    new_walk_pe = batch.walk_pe.clone()
    new_lengths = batch.lengths.clone()

    for i in range(B):
        perm = torch.as_tensor(perm_per_graph[i], dtype=torch.long)
        sl = slice(i * NW, (i + 1) * NW)
        new_walk_emb[sl] = batch.walk_emb[sl][perm]
        new_walk_pe[sl] = batch.walk_pe[sl][perm]
        new_lengths[sl] = batch.lengths[sl][perm]
        new_walk_ids[i] = batch.walk_ids[i][perm]

    new = batch.clone()
    new.walk_emb = new_walk_emb
    new.walk_ids = new_walk_ids
    new.walk_pe = new_walk_pe
    new.lengths = new_lengths
    return new


def _reverse_timesteps(batch: Batch) -> Batch:
    """Return a new batch with each walk's timesteps reversed.

    Only meaningful when lengths == max_len for every walk (no padding to
    worry about). Our synthetic batches satisfy this by construction.
    """
    assert torch.all(batch.lengths == MAX_LEN), (
        "reverse_timesteps assumes lengths == MAX_LEN for every walk"
    )
    new = batch.clone()
    new.walk_emb = torch.flip(batch.walk_emb, dims=[1])
    new.walk_pe = torch.flip(batch.walk_pe, dims=[1])
    new.walk_ids = torch.flip(batch.walk_ids, dims=[2])
    return new


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_output_is_invariant_to_walk_permutation(reduce):
    model = _make_model(num_layers=2, reduce=reduce).eval()
    batch = _make_batch(seed=42)

    with torch.no_grad():
        y1 = model(batch)

    # Non-trivial permutations per graph (NW=3, so (2,0,1) and (1,2,0)).
    perms = [[2, 0, 1], [1, 2, 0]]
    batch2 = _permute_walks(batch, perms)
    with torch.no_grad():
        y2 = model(batch2)

    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4), (
        f"output not invariant to walk permutation under reduce={reduce}: "
        f"y1={y1}, y2={y2}, max_abs_diff={(y1 - y2).abs().max().item():.6f}"
    )


def test_output_is_sensitive_to_timestep_permutation_within_walk():
    """Reversing every walk should change the LSTM output.

    This documents (rather than tests-for-a-bug) the fact that ``RSNN_LSTM``
    is a sequence model: per-walk timestep order matters even though across
    walks the model is permutation-invariant.
    """
    model = _make_model(num_layers=2, reduce="sum").eval()
    batch = _make_batch(seed=7)

    with torch.no_grad():
        y1 = model(batch)

    batch_rev = _reverse_timesteps(batch)
    with torch.no_grad():
        y3 = model(batch_rev)

    # Outputs must differ noticeably — assert a non-trivial gap so the test
    # isn't a tautology if both happened to be near zero.
    diff = (y1 - y3).abs().max().item()
    assert diff > 1e-4, (
        f"LSTM was expected to be sensitive to within-walk timestep order, "
        f"but y1 ~= y3 (max abs diff = {diff:.2e}). y1={y1}, y3={y3}"
    )
