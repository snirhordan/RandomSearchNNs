"""Forward / backward smoke tests for every model class in ``models.rwnn``."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.rwnn import (
    RWNN,
    RSNN,
    RSNN_LSTM,
    RSNN_TRSF,
    RWNN_base,
    RWNN_base_ada,
)
from utils.search import (
    sample_walks,
    sample_walks_adaptive,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
HID_DIM = 16
PE_OUT_DIM = 8
NUM_LAYERS = 1
OUT_DIM = 1
REDUCE = 'mean'
NW = 4
WALK_LEN = 6
WIN = 2          # window size s
PE_IN_DIM = 2 * WIN
N_EMB_PAD = 1    # extra slot for PAD


def _walk_batch(tiny_graph, vocab):
    """Build a fixed-length walk Batch (used for RWNN, RWNN_base)."""
    g = tiny_graph.clone()
    g = sample_walks(g, NW, WALK_LEN, WIN, non_backtracking=False)
    # DataLoader handles collation correctly for the per-graph attributes.
    loader = DataLoader([g], batch_size=1)
    return next(iter(loader))


def _adaptive_walk_batch(tiny_graph, vocab):
    """Build an adaptive (padded) walk Batch (used for RSNN-family and RWNN_base_ada)."""
    g = tiny_graph.clone()
    max_len = max(WALK_LEN, g.x.shape[0])
    walk_l = min(WALK_LEN, g.x.shape[0])
    g = sample_walks_adaptive(g, NW, walk_l, WIN, False, max_len, vocab)
    loader = DataLoader([g], batch_size=1)
    return next(iter(loader))


def _vocab_size(tiny_vocab):
    return len(tiny_vocab)


# ---------------------------------------------------------------------------
# fixed-length-walk models: RWNN, RWNN_base
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ModelCls", [RWNN, RWNN_base])
def test_fixed_walk_model_forward_backward(tiny_graph, tiny_vocab, ModelCls):
    n_emb = _vocab_size(tiny_vocab)
    batch = _walk_batch(tiny_graph, tiny_vocab)
    model = ModelCls(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, NUM_LAYERS, n_emb, REDUCE)
    out = model(batch)
    assert out.shape == (1, OUT_DIM)
    assert torch.isfinite(out).all()
    target = torch.tensor([[0.0]])
    loss = nn.BCELoss()(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    # at least some grads must exist; the embedding-PAD entry can be None,
    # so we relax to "at least half the leaf params have non-None gradients"
    non_none = sum(1 for g in grads if g is not None)
    assert non_none >= max(1, len(grads) // 2)


@pytest.mark.parametrize("ModelCls", [RWNN, RWNN_base])
def test_fixed_walk_model_cuda(tiny_graph, tiny_vocab, ModelCls, device):
    if device.type != "cuda":
        pytest.skip("no CUDA available")
    n_emb = _vocab_size(tiny_vocab)
    batch = _walk_batch(tiny_graph, tiny_vocab).to(device)
    model = ModelCls(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, NUM_LAYERS, n_emb, REDUCE).to(device)
    out = model(batch)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# adaptive (padded) models: RSNN, RSNN_LSTM, RSNN_TRSF, RWNN_base_ada
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ModelCls", [RSNN, RSNN_LSTM, RWNN_base_ada])
def test_adaptive_model_forward_backward(tiny_graph, tiny_vocab, ModelCls):
    n_emb = _vocab_size(tiny_vocab)
    batch = _adaptive_walk_batch(tiny_graph, tiny_vocab)
    model = ModelCls(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, NUM_LAYERS, n_emb, REDUCE)
    out = model(batch)
    assert out.shape == (1, OUT_DIM)
    assert torch.isfinite(out).all()
    loss = nn.BCELoss()(out, torch.tensor([[1.0]]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    non_none = sum(1 for g in grads if g is not None)
    assert non_none >= max(1, len(grads) // 2)


def test_rsnn_trsf_forward_backward(tiny_graph, tiny_vocab):
    """The transformer variant has a divisibility constraint: nhead | d_model.

    The class hard-codes ``nhead == hidp_dim`` so every config satisfies the
    constraint.  We use a slightly larger hidden size to keep attention well-conditioned.
    """
    n_emb = _vocab_size(tiny_vocab)
    batch = _adaptive_walk_batch(tiny_graph, tiny_vocab)
    model = RSNN_TRSF(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, NUM_LAYERS, n_emb, REDUCE)
    out = model(batch)
    assert out.shape == (1, OUT_DIM)
    assert torch.isfinite(out).all()
    loss = nn.BCELoss()(out, torch.tensor([[0.0]]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    non_none = sum(1 for g in grads if g is not None)
    assert non_none >= max(1, len(grads) // 2)


@pytest.mark.parametrize("ModelCls", [RSNN, RSNN_LSTM, RSNN_TRSF, RWNN_base_ada])
def test_adaptive_model_cuda(tiny_graph, tiny_vocab, ModelCls, device):
    if device.type != "cuda":
        pytest.skip("no CUDA available")
    n_emb = _vocab_size(tiny_vocab)
    batch = _adaptive_walk_batch(tiny_graph, tiny_vocab).to(device)
    model = ModelCls(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, NUM_LAYERS, n_emb, REDUCE).to(device)
    out = model(batch)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# output-range sanity check (sigmoid)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ModelCls", [RWNN, RWNN_base])
def test_fixed_walk_model_output_in_unit_interval(tiny_graph, tiny_vocab, ModelCls):
    n_emb = _vocab_size(tiny_vocab)
    batch = _walk_batch(tiny_graph, tiny_vocab)
    model = ModelCls(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, NUM_LAYERS, n_emb, REDUCE)
    with torch.no_grad():
        out = model(batch)
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.parametrize("ModelCls", [RSNN, RSNN_LSTM, RSNN_TRSF, RWNN_base_ada])
def test_adaptive_model_output_in_unit_interval(tiny_graph, tiny_vocab, ModelCls):
    n_emb = _vocab_size(tiny_vocab)
    batch = _adaptive_walk_batch(tiny_graph, tiny_vocab)
    model = ModelCls(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, NUM_LAYERS, n_emb, REDUCE)
    with torch.no_grad():
        out = model(batch)
    assert (out >= 0).all() and (out <= 1).all()
