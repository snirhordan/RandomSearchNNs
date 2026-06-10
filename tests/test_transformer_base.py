"""Tests for the transformer sequence base (models/seq_encoder.py) and the
QM9 regression head ``RSNN_TRSF_Reg`` (quickstart/train_qm9.py).

Covers: forward/backward, causal-mask future independence, full-vs-causal
divergence, padding invariance, RoPE norm preservation + relative-position
property, and finite outputs at quadruplet (angle+dihedral) feature widths.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.seq_encoder import (
    TransformerSeqLayer,
    apply_rope,
    rope_cos_sin,
    sinusoidal_positional_encoding,
)
from quickstart.train_qm9 import RSNN_TRSF_Reg
from utils.search import sample_walks_adaptive
from torch_geometric.loader import DataLoader


HID_DIM = 16
PE_OUT_DIM = 8
D_MODEL = HID_DIM + PE_OUT_DIM   # 24
NHEAD = 4                        # head_dim = 6 (even -> RoPE-compatible)
OUT_DIM = 1
REDUCE = "mean"
NW = 4
WALK_LEN = 6
WIN = 2
PE_IN_DIM = 2 * WIN


def _adaptive_walk_batch(tiny_graph, vocab):
    g = tiny_graph.clone()
    max_len = max(WALK_LEN, g.x.shape[0])
    walk_l = min(WALK_LEN, g.x.shape[0])
    g = sample_walks_adaptive(g, NW, walk_l, WIN, False, max_len, vocab)
    loader = DataLoader([g], batch_size=1)
    return next(iter(loader))


def _synthetic_batch(pe_in_dim=PE_IN_DIM, seed=0):
    """Two graphs x 3 walks, padded length 10, vocab size 6 (PAD = 5)."""

    class B:
        pass

    torch.manual_seed(seed)
    b = B()
    b.lengths = torch.tensor([10, 7, 5, 10, 3, 8])
    b.walk_emb = torch.randint(0, 5, (6, 10))
    b.walk_pe = torch.randn(6, 10, pe_in_dim)
    wi = torch.full((2, 3, 10), -1, dtype=torch.long)
    for g in range(2):
        for s in range(3):
            L = int(b.lengths[g * 3 + s])
            wi[g, s, :L] = torch.randint(0, 4, (L,))
    b.walk_ids = wi
    for i in range(6):
        b.walk_emb[i, int(b.lengths[i]):] = 5
    return b


# ---------------------------------------------------------------------------
# layer-level: masking semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pos_enc", ["sinusoidal", "rope", "none"])
def test_causal_layer_future_independence(pos_enc):
    """Perturbing token t must not change outputs at positions < t."""
    torch.manual_seed(0)
    layer = TransformerSeqLayer(D_MODEL, NHEAD, attn_mode="causal",
                                pos_enc=pos_enc).eval()
    x = torch.randn(2, 8, D_MODEL)
    t = 5
    x2 = x.clone()
    x2[:, t:, :] += torch.randn_like(x2[:, t:, :])
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x2)
    assert torch.allclose(y1[:, :t, :], y2[:, :t, :], atol=1e-6), \
        "causal layer leaked future information"
    assert not torch.allclose(y1[:, t:, :], y2[:, t:, :], atol=1e-4)


def test_full_layer_uses_future():
    """Full attention must propagate a future perturbation backwards."""
    torch.manual_seed(0)
    layer = TransformerSeqLayer(D_MODEL, NHEAD, attn_mode="full",
                                pos_enc="none").eval()
    x = torch.randn(2, 8, D_MODEL)
    x2 = x.clone()
    x2[:, 5:, :] += torch.randn_like(x2[:, 5:, :])
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x2)
    assert not torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-4), \
        "full attention did not see future tokens"


@pytest.mark.parametrize("attn_mode", ["full", "causal"])
@pytest.mark.parametrize("pos_enc", ["sinusoidal", "rope", "none"])
def test_layer_padding_invariance(attn_mode, pos_enc):
    """Appending PAD columns must not change real-position outputs."""
    torch.manual_seed(0)
    layer = TransformerSeqLayer(D_MODEL, NHEAD, attn_mode=attn_mode,
                                pos_enc=pos_enc).eval()
    B, L, extra = 3, 6, 4
    x = torch.randn(B, L, D_MODEL)
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    x_ext = torch.cat([x, torch.randn(B, extra, D_MODEL)], dim=1)
    pad_ext = torch.cat(
        [pad_mask, torch.ones(B, extra, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        y = layer(x, pad_mask=pad_mask)
        y_ext = layer(x_ext, pad_mask=pad_ext)
    assert torch.allclose(y, y_ext[:, :L, :], atol=1e-5), \
        "padding columns contaminated real positions"


def test_padded_rows_no_nan():
    """A fully-padded query row must not produce NaN (self-attend guard)."""
    torch.manual_seed(0)
    layer = TransformerSeqLayer(D_MODEL, NHEAD, attn_mode="causal",
                                pos_enc="rope").eval()
    x = torch.randn(2, 6, D_MODEL)
    pad_mask = torch.zeros(2, 6, dtype=torch.bool)
    pad_mask[1, 3:] = True
    with torch.no_grad():
        y = layer(x, pad_mask=pad_mask)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# RoPE math
# ---------------------------------------------------------------------------
def test_rope_norm_preservation():
    torch.manual_seed(0)
    head_dim = 6
    x = torch.randn(2, 3, 10, head_dim)
    cos, sin = rope_cos_sin(10, head_dim)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_relative_position_property():
    """q_i . k_j after RoPE depends only on the offset j - i."""
    torch.manual_seed(0)
    head_dim = 8
    L = 12
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)
    cos, sin = rope_cos_sin(L, head_dim)

    def scored(i, j):
        qi = apply_rope(q[None, :], cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k[None, :], cos[j:j + 1], sin[j:j + 1])
        return float((qi * kj).sum())

    # same offset (2), different absolute positions
    assert abs(scored(1, 3) - scored(5, 7)) < 1e-4
    assert abs(scored(0, 2) - scored(8, 10)) < 1e-4
    # different offsets disagree
    assert abs(scored(1, 3) - scored(1, 6)) > 1e-3


def test_constructor_validation():
    with pytest.raises(ValueError):
        TransformerSeqLayer(D_MODEL, 5)                 # 5 does not divide 24
    with pytest.raises(ValueError):
        TransformerSeqLayer(D_MODEL, 8, pos_enc="rope")  # head_dim 3 odd
    with pytest.raises(ValueError):
        TransformerSeqLayer(D_MODEL, NHEAD, attn_mode="banana")
    with pytest.raises(ValueError):
        TransformerSeqLayer(D_MODEL, NHEAD, pos_enc="banana")


# ---------------------------------------------------------------------------
# model-level: RSNN_TRSF_Reg
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attn_mode", ["full", "causal"])
@pytest.mark.parametrize("pos_enc", ["sinusoidal", "rope", "none"])
def test_model_forward_backward(tiny_graph, tiny_vocab, attn_mode, pos_enc):
    n_emb = len(tiny_vocab)
    batch = _adaptive_walk_batch(tiny_graph, tiny_vocab)
    model = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, n_emb,
                          REDUCE, nhead=NHEAD, attn_mode=attn_mode,
                          pos_enc=pos_enc)
    out = model(batch)
    assert out.shape == (1, OUT_DIM)
    assert torch.isfinite(out).all()
    loss = nn.L1Loss()(out, torch.tensor([[0.5]]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    non_none = sum(1 for g in grads if g is not None)
    assert non_none >= max(1, len(grads) // 2)


def test_model_full_vs_causal_differ():
    """Identical weights, different attention mode => different outputs."""
    batch = _synthetic_batch()
    torch.manual_seed(1)
    m_full = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6,
                           "sum", nhead=NHEAD, attn_mode="full",
                           pos_enc="rope").eval()
    m_causal = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6,
                             "sum", nhead=NHEAD, attn_mode="causal",
                             pos_enc="rope").eval()
    m_causal.load_state_dict(m_full.state_dict())
    with torch.no_grad():
        assert not torch.allclose(m_full(batch), m_causal(batch), atol=1e-5)


def test_model_quadruplet_feature_width():
    """pe_in_dim = 16 window + 8 angle + 8 dihedral = 32 (Phase-2 config)."""
    batch = _synthetic_batch(pe_in_dim=32)
    model = RSNN_TRSF_Reg(32, 16, 128, OUT_DIM, 2, 6, "sum",
                          nhead=8, attn_mode="full", pos_enc="rope")
    out = model(batch)
    assert out.shape == (2, OUT_DIM)
    assert torch.isfinite(out).all()


def test_model_extra_padding_invariance():
    """Extending dataset-level padding must not change the output."""
    batch = _synthetic_batch()
    torch.manual_seed(2)
    model = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6,
                          "sum", nhead=NHEAD, attn_mode="full",
                          pos_enc="rope").eval()
    with torch.no_grad():
        out = model(batch)

    ext = _synthetic_batch()
    pad_cols = 5
    ext.walk_emb = torch.cat(
        [ext.walk_emb, torch.full((6, pad_cols), 5, dtype=torch.long)], dim=1)
    ext.walk_pe = torch.cat(
        [ext.walk_pe, torch.zeros(6, pad_cols, PE_IN_DIM)], dim=1)
    ext.walk_ids = torch.cat(
        [ext.walk_ids, torch.full((2, 3, pad_cols), -1, dtype=torch.long)],
        dim=2)
    with torch.no_grad():
        out_ext = model(ext)
    assert torch.allclose(out, out_ext, atol=1e-5)


def test_sinusoidal_pe_odd_d_model():
    """Odd d_model must not crash (cosine half has floor(d/2) columns)."""
    pe = sinusoidal_positional_encoding(10, 143)
    assert pe.shape == (10, 143)
    assert torch.isfinite(pe).all()


def _permute_walk_steps(batch, walk_idx, perm):
    """Permute the step order of one walk consistently across all attrs."""
    L = len(perm)
    batch.walk_emb[walk_idx, :L] = batch.walk_emb[walk_idx, perm]
    batch.walk_pe[walk_idx, :L] = batch.walk_pe[walk_idx, perm]
    g, s = divmod(walk_idx, batch.walk_ids.shape[1])
    batch.walk_ids[g, s, :L] = batch.walk_ids[g, s, perm]
    return batch


def test_positional_signal_reaches_upper_layers():
    """The node-state write-back between layers erases additive positional
    info, so sinusoidal PE must be re-injected before every layer: with
    pos_enc='none' a 2-layer full-attention model is invariant to permuting
    a walk's step order, while 'sinusoidal' and 'rope' must NOT be."""
    L = 10
    perm = torch.randperm(L)
    while torch.equal(perm, torch.arange(L)):
        perm = torch.randperm(L)

    outs = {}
    for pe in ["none", "sinusoidal", "rope"]:
        torch.manual_seed(3)
        model = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6,
                              "sum", nhead=NHEAD, attn_mode="full",
                              pos_enc=pe).eval()
        with torch.no_grad():
            base = model(_synthetic_batch(seed=7))
            permuted = model(
                _permute_walk_steps(_synthetic_batch(seed=7), 0, perm))
        outs[pe] = (base, permuted)

    assert torch.allclose(*outs["none"], atol=1e-5), \
        "pos_enc='none' should be step-order invariant (sanity check)"
    assert not torch.allclose(*outs["sinusoidal"], atol=1e-6), \
        "sinusoidal PE not reaching upper layers (write-back erased it)"
    assert not torch.allclose(*outs["rope"], atol=1e-6)


def test_model_cuda(tiny_graph, tiny_vocab, device):
    if device.type != "cuda":
        pytest.skip("no CUDA available")
    n_emb = len(tiny_vocab)
    batch = _adaptive_walk_batch(tiny_graph, tiny_vocab).to(device)
    model = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, n_emb,
                          REDUCE, nhead=NHEAD, attn_mode="causal",
                          pos_enc="rope").to(device)
    out = model(batch)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
