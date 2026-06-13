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


# ---------------------------------------------------------------------------
# Cross-path attention (attn_mode="full_xpath")
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pos_enc", ["rope", "sinusoidal", "none"])
def test_xpath_forward_backward(pos_enc):
    batch = _synthetic_batch()
    model = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6,
                          "sum", nhead=NHEAD, attn_mode="full_xpath",
                          pos_enc=pos_enc)
    out = model(batch)
    assert out.shape == (2, OUT_DIM)
    assert torch.isfinite(out).all()
    nn.L1Loss()(out, torch.zeros_like(out)).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert sum(1 for g in grads if g is not None) >= max(1, len(grads) // 2)


def test_xpath_param_count_identical_to_full():
    """Cross-path must be parameter-count-neutral vs full (same weights)."""
    kw = dict(nhead=NHEAD, pos_enc="rope")
    pf = sum(p.numel() for p in RSNN_TRSF_Reg(
        PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 3, 6, "sum",
        attn_mode="full", **kw).parameters())
    px = sum(p.numel() for p in RSNN_TRSF_Reg(
        PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 3, 6, "sum",
        attn_mode="full_xpath", **kw).parameters())
    assert pf == px


def test_xpath_cross_molecule_isolation():
    """A molecule's output must not depend on another molecule's tokens."""
    torch.manual_seed(2)
    model = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6,
                          "sum", nhead=NHEAD, attn_mode="full_xpath",
                          pos_enc="rope").eval()
    b = _synthetic_batch(seed=7)
    with torch.no_grad():
        o1 = model(b)
    b2 = _synthetic_batch(seed=7)
    # Perturb only molecule 1's walks (rows 3..5), keep PADs consistent.
    b2.walk_emb[3:] = torch.randint(0, 5, (3, 10))
    b2.walk_pe[3:] = torch.randn(3, 10, PE_IN_DIM)
    for i in range(3, 6):
        b2.walk_emb[i, int(b2.lengths[i]):] = 5
    with torch.no_grad():
        o2 = model(b2)
    assert torch.allclose(o1[0], o2[0], atol=1e-6)        # mol 0 unchanged
    assert not torch.allclose(o1[1], o2[1], atol=1e-5)    # mol 1 changed


def test_xpath_walk_permutation_invariance():
    """Cross-path treats a molecule's m walks as an unordered set: permuting
    the walks (with per-walk RoPE reset + permutation-invariant node mean)
    must not change the molecule's output."""
    torch.manual_seed(3)
    model = RSNN_TRSF_Reg(PE_IN_DIM, PE_OUT_DIM, HID_DIM, OUT_DIM, 2, 6,
                          "sum", nhead=NHEAD, attn_mode="full_xpath",
                          pos_enc="rope").eval()
    b = _synthetic_batch(seed=4)
    with torch.no_grad():
        o1 = model(b)
    p = _synthetic_batch(seed=4)
    perm = [2, 0, 1]                       # reorder molecule 0's 3 walks
    p.walk_emb[0:3] = p.walk_emb[0:3][perm]
    p.walk_pe[0:3] = p.walk_pe[0:3][perm]
    p.lengths[0:3] = p.lengths[0:3][perm]
    p.walk_ids[0] = p.walk_ids[0][perm]
    with torch.no_grad():
        o2 = model(p)
    assert torch.allclose(o1[0], o2[0], atol=1e-5)


# ---------------------------------------------------------------------------
# bonded_angles_only gating in sample_dfs
# ---------------------------------------------------------------------------
def _star_graph():
    """Center node 0 bonded to leaves 1,2,3 — DFS must stack-jump between
    leaves, producing non-bonded consecutive walk-order atoms."""
    from torch_geometric.data import Data
    ei = torch.tensor([[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]])
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       dtype=torch.float)
    d = Data(edge_index=ei, x_emb=torch.zeros(4, dtype=torch.long),
             pos=pos)
    d.num_nodes = 4
    return d


def test_bonded_only_zeros_stackjump_angles():
    import random
    from utils.search import sample_dfs
    vocab = {"PAD": 1}
    K = 8
    # Identical DFS via identical seeds; only the gating flag differs.
    random.seed(5); torch.manual_seed(5)
    u = sample_dfs(_star_graph(), 1, 4, 4, vocab, angles=True, dihedrals=True,
                   angle_K=K, dihedral_K=4, bonded_only=False)
    random.seed(5); torch.manual_seed(5)
    b = sample_dfs(_star_graph(), 1, 4, 4, vocab, angles=True, dihedrals=True,
                   angle_K=K, dihedral_K=4, bonded_only=True)
    assert torch.equal(u.walk_ids, b.walk_ids)            # same DFS order
    s = 4                                                  # edge-encoding width
    ua, ba = u.walk_pe[0, :, s:s + K], b.walk_pe[0, :, s:s + K]
    # Every angle the gated path keeps must equal the ungated value...
    for i in range(ba.shape[0]):
        if ba[i].abs().sum() > 0:
            assert torch.allclose(ba[i], ua[i])
    # ...and gating must zero at least one stack-jump angle the ungated kept.
    assert any((ua[i].abs().sum() > 0) and (ba[i].abs().sum() == 0)
               for i in range(ba.shape[0]))


def test_bonded_only_noop_on_path_graph():
    """On a linear path (pentane-like), DFS from an end never stack-jumps, so
    bonded_only must be a no-op (every consecutive pair is bonded)."""
    import random
    from torch_geometric.data import Data
    from utils.search import sample_dfs
    # 5-node path 0-1-2-3-4
    src = [0, 1, 1, 2, 2, 3, 3, 4]
    dst = [1, 0, 2, 1, 3, 2, 4, 3]
    ei = torch.tensor([src, dst])
    pos = torch.tensor([[i, 0, 0] for i in range(5)], dtype=torch.float)
    vocab = {"PAD": 1}

    def mk():
        d = Data(edge_index=ei, x_emb=torch.zeros(5, dtype=torch.long),
                 pos=pos.clone())
        d.num_nodes = 5
        return d

    # Force start at an end node so DFS is the straight path (no backtrack).
    for seed in (0, 1, 2, 3, 4):
        random.seed(seed); torch.manual_seed(seed)
        u = sample_dfs(mk(), 1, 5, 5, vocab, angles=True, dihedrals=True,
                       bonded_only=False)
        order = u.walk_ids[0, 0].tolist()
        if order[0] not in (0, 4):
            continue
        random.seed(seed); torch.manual_seed(seed)
        b = sample_dfs(mk(), 1, 5, 5, vocab, angles=True, dihedrals=True,
                       bonded_only=True)
        assert torch.allclose(u.walk_pe, b.walk_pe)
        return
    pytest.skip("no end-start DFS sampled in tried seeds")
