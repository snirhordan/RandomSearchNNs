"""Sanity-check tests for the LSTM internals of ``RSNN_LSTM_Reg``.

These tests pin the *current* (PyTorch-default) initialization and gradient
behaviour of the bidirectional LSTM stack inside
``quickstart.train_qm9.RSNN_LSTM_Reg`` so that upcoming training-optimisation
flags (``--grad_clip``, ``--lstm_init=orthogonal``, ``--dropout``, ...) have
a documented baseline to flip away from.

The checks correspond to standard RNN training concerns at the level of the
CS230 RNN cheatsheet:

1. Bidirectionality is enabled by default (and the backward-direction weights
   are actually allocated).
2. The total parameter count matches the sum over named sub-modules
   (rnn_layers, embedding, pe_encoding, readout).
3. The default ``W_hh`` initialisation is *not* orthogonal -- so a future
   ``--lstm_init=orthogonal`` flag has real work to do.
4. The default forget-gate bias slice is approximately zero -- so the
   classic "+1 forget bias" trick from Jozefowicz et al. ("An Empirical
   Exploration of Recurrent Network Architectures", 2015) is currently
   absent.
5. Gradients flow through both LSTM layers in finite, non-zero magnitudes
   (vanishing/exploding-gradient detector).
6. ``torch.nn.utils.clip_grad_norm_`` interoperates with the model and
   actually reduces grad norms when they exceed ``max_norm``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# --- make the repo (and its quickstart/) importable -------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
QUICKSTART = REPO_ROOT / "quickstart"
if str(QUICKSTART) not in sys.path:
    sys.path.insert(0, str(QUICKSTART))

from quickstart.train_qm9 import RSNN_LSTM_Reg  # noqa: E402
from utils.search import sample_walks_adaptive  # noqa: E402


# ---------------------------------------------------------------------------
# Model hyper-params (match the spec in the task description).
# ---------------------------------------------------------------------------
PE_IN_DIM = 8
PE_OUT_DIM = 16
HID_DIM = 64
OUT_DIM = 1
NUM_LAYERS = 2
N_EMB = 6
REDUCE = "sum"


def _make_model() -> RSNN_LSTM_Reg:
    torch.manual_seed(0)
    return RSNN_LSTM_Reg(
        pe_in_dim=PE_IN_DIM,
        pe_out_dim=PE_OUT_DIM,
        hid_dim=HID_DIM,
        out_dim=OUT_DIM,
        num_layers=NUM_LAYERS,
        n_emb=N_EMB,
        reduce=REDUCE,
    )


def _make_batch() -> "torch_geometric.data.Batch":
    """Build a tiny synthetic adaptive-walk batch compatible with the model.

    A 5-node path graph (pentane-like) with:
      - x_emb in {0..N_EMB-1=5}; PAD token = N_EMB-1=5 (model's padding_idx)
      - sample_walks_adaptive(nw=4, walk_l=5, s=PE_IN_DIM=8, max_len=5)
    """
    from torch_geometric.data import Data
    from generation.utils import get_neighbor_dict

    n = 5
    # path graph: 0-1-2-3-4
    src = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4], dtype=torch.long)
    dst = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    x = torch.zeros((n, 1), dtype=torch.float)
    # leave the PAD index (N_EMB - 1) for padding; assign real tokens 0..3.
    x_emb = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, x_emb=x_emb)
    data = get_neighbor_dict(data)

    # PAD index must match the model's embedding padding_idx (n_emb - 1).
    vocab = {"PAD": N_EMB - 1}
    # sample_walks_adaptive emits walk_pe of width 2*s (repeat + edge encodings)
    # so PE_IN_DIM=8 requires s=4.
    assert PE_IN_DIM % 2 == 0
    s = PE_IN_DIM // 2
    g = sample_walks_adaptive(
        data, 4, 5, s, False, 5, vocab, add_edge_feat=None
    )
    loader = DataLoader([g], batch_size=1)
    return next(iter(loader))


# ===========================================================================
# 1. Bidirectional verification
# ===========================================================================
def test_lstm_layers_are_bidirectional():
    """Each ``nn.LSTM`` is created with ``bidirectional=True`` and PyTorch
    therefore allocates the ``*_reverse`` parameter tensors for the backward
    direction. This pins the current default before we add flags."""
    model = _make_model()
    assert len(model.rnn_layers) == NUM_LAYERS
    for i, lstm in enumerate(model.rnn_layers):
        assert isinstance(lstm, nn.LSTM), f"layer {i} is not nn.LSTM"
        assert lstm.bidirectional is True, (
            f"layer {i} bidirectional={lstm.bidirectional!r}, expected True"
        )
        names = {n for n, _ in lstm.named_parameters()}
        for required in (
            "weight_ih_l0",
            "weight_hh_l0",
            "weight_ih_l0_reverse",
            "weight_hh_l0_reverse",
            "bias_ih_l0",
            "bias_hh_l0",
            "bias_ih_l0_reverse",
            "bias_hh_l0_reverse",
        ):
            assert required in names, (
                f"layer {i} missing parameter {required!r}; "
                f"available: {sorted(names)}"
            )


# ===========================================================================
# 2. Param count per component
# ===========================================================================
def test_param_count_breakdown_sums_to_total():
    """Sum of params across (rnn_layers, embedding, pe_encoding, readout)
    must equal ``sum(p.numel() for p in model.parameters())``."""
    model = _make_model()
    breakdown = {
        "rnn_layers": sum(p.numel() for p in model.rnn_layers.parameters()),
        "embedding": sum(p.numel() for p in model.embedding.parameters()),
        "pe_encoding": sum(p.numel() for p in model.pe_encoding.parameters()),
        "readout": sum(p.numel() for p in model.readout.parameters()),
    }
    total = sum(p.numel() for p in model.parameters())
    summed = sum(breakdown.values())
    print("\n[param breakdown]")
    for k, v in breakdown.items():
        print(f"  {k:<12s}: {v:>9d}")
    print(f"  {'TOTAL':<12s}: {total:>9d}")
    assert summed == total, (
        f"submodule sum {summed} != total {total}; "
        f"breakdown={breakdown}"
    )


# ===========================================================================
# 3. Default init is NOT orthogonal
# ===========================================================================
def test_default_w_hh_is_not_orthogonal():
    """PyTorch initialises ``weight_hh_l0`` with uniform in
    ``+- 1/sqrt(hidden_size)`` -- so ``W W^T`` is far from identity.
    This confirms an ``--lstm_init=orthogonal`` flag would actually change
    initialisation."""
    model = _make_model()
    for i, lstm in enumerate(model.rnn_layers):
        # weight_hh_l0 has shape (4 * hidden_size, hidden_size)
        W = lstm.weight_hh_l0.detach()
        hs = lstm.hidden_size
        assert W.shape == (4 * hs, hs)
        # Compare W @ W^T against identity in (4*hs, 4*hs).
        gram = W @ W.t()
        identity = torch.eye(4 * hs, dtype=gram.dtype)
        # If init were orthogonal (rows orthonormal), gram would be close to I.
        diff_norm = (gram - identity).norm().item()
        print(
            f"[layer {i}] ||W_hh W_hh^T - I||_F = {diff_norm:.3f}  "
            f"(would be ~0 for orthogonal init)"
        )
        assert not torch.allclose(gram, identity, atol=1e-2), (
            f"layer {i} weight_hh_l0 looks orthogonal under default init; "
            f"diff_norm={diff_norm}"
        )


# ===========================================================================
# 4. Default forget-gate bias is ~0
# ===========================================================================
def test_default_forget_gate_bias_is_zero():
    """PyTorch's default LSTM bias init is uniform in ``+- 1/sqrt(hidden_size)``
    *summed* across ``bias_ih`` and ``bias_hh`` (which both contribute), so
    the **mean** forget-gate bias is statistically ~0. We assert the absolute
    mean is small relative to ``1/sqrt(hidden_size)``.

    Classic LSTM trick (Jozefowicz et al., 2015): set the forget-gate bias
    slice -- ``bias_ih_l0[hs:2*hs] + bias_hh_l0[hs:2*hs]`` -- to ``+1.0`` to
    bias the cell toward remembering at the start of training. The future
    ``--lstm_init=orthogonal`` flag should also apply this trick."""
    model = _make_model()
    for i, lstm in enumerate(model.rnn_layers):
        hs = lstm.hidden_size
        # Gate slices in PyTorch LSTM bias: [i_gate, f_gate, g_gate, o_gate].
        bias_ih = lstm.bias_ih_l0.detach()
        bias_hh = lstm.bias_hh_l0.detach()
        assert bias_ih.shape == (4 * hs,)
        assert bias_hh.shape == (4 * hs,)
        f_ih = bias_ih[hs : 2 * hs]
        f_hh = bias_hh[hs : 2 * hs]
        f_total_mean = (f_ih + f_hh).mean().item()
        # Default-init range bound: each entry is U(-1/sqrt(hs), 1/sqrt(hs)),
        # so the mean of (f_ih + f_hh) has std ~ sqrt(2/hs) / sqrt(hs).
        # We just assert it's well below 1 (which is the post-trick value).
        bound = 1.0 / (hs ** 0.5)
        print(
            f"[layer {i}] mean(f_ih + f_hh) = {f_total_mean:+.4f}  "
            f"(default bound ~ +-{bound:.3f}; post-trick value would be ~+1.0)"
        )
        assert abs(f_total_mean) < 0.5, (
            f"layer {i} forget-gate bias mean {f_total_mean} is suspiciously "
            "large for default init"
        )


# ===========================================================================
# 5. Gradient flow check
# ===========================================================================
def test_gradient_flow_through_lstm_stack():
    """Run one forward + backward on a synthetic batch and inspect
    ``weight_hh_l0.grad.norm()`` per layer. All norms must be finite,
    > 1e-8 (no zero-gradient) and < 100 (no exploding gradient).

    A monotonically decreasing pattern across layers (deeper layer has
    smaller norm) is an early sign of vanishing gradients and is logged
    as a warning -- not a hard failure -- since at init the magnitude is
    expected to be small but non-zero."""
    torch.manual_seed(0)
    model = _make_model()
    batch = _make_batch()
    out = model(batch)
    assert out.shape == (1, OUT_DIM)
    # Regression head: use MSE against a unit target.
    target = torch.ones_like(out)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    norms = []
    for i, lstm in enumerate(model.rnn_layers):
        g = lstm.weight_hh_l0.grad
        assert g is not None, f"layer {i} weight_hh_l0 has no grad"
        n = g.norm().item()
        norms.append(n)
        print(f"[layer {i}] ||grad weight_hh_l0|| = {n:.4e}")
        assert torch.isfinite(g).all(), f"layer {i} grad has NaN/Inf"
        assert 1e-8 < n < 100.0, (
            f"layer {i} grad norm {n:.4e} outside (1e-8, 100); "
            "possible vanishing/exploding gradient"
        )

    # Soft diagnostic: monotonic decrease across depth.
    monotone_decreasing = all(
        norms[k] >= norms[k + 1] for k in range(len(norms) - 1)
    )
    if monotone_decreasing and len(norms) > 1:
        print(
            "[diag] grad norms are monotonically decreasing across layers "
            f"({norms}); watch for vanishing gradients as depth grows."
        )


# ===========================================================================
# 6. grad_clip would work
# ===========================================================================
def test_grad_clip_norm_runs_and_reduces_large_grads():
    """``torch.nn.utils.clip_grad_norm_`` must:
      (a) run cleanly after ``loss.backward()`` on this model;
      (b) actually reduce the grad norm when the unclipped norm exceeds
          ``max_norm``.

    This pins the contract a future ``--grad_clip`` CLI flag will rely on."""
    torch.manual_seed(0)
    model = _make_model()
    batch = _make_batch()
    out = model(batch)
    # Multiply loss by a large factor to push grads above max_norm.
    loss = 1e3 * (out ** 2).sum()
    loss.backward()

    # Snapshot the pre-clip total grad norm.
    params = [p for p in model.parameters() if p.grad is not None]
    pre_norm = torch.norm(
        torch.stack([p.grad.detach().norm() for p in params])
    ).item()
    print(f"[clip] pre-clip total grad norm = {pre_norm:.4e}")
    assert pre_norm > 1.0, (
        f"expected pre-clip norm > 1.0 to exercise clipping; got {pre_norm}"
    )

    max_norm = 1.0
    returned = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    # ``returned`` is the total norm *before* clipping (torch's contract).
    print(f"[clip] clip_grad_norm_ returned (pre-clip norm) = {float(returned):.4e}")
    assert abs(float(returned) - pre_norm) / max(pre_norm, 1e-12) < 1e-4

    post_norm = torch.norm(
        torch.stack([p.grad.detach().norm() for p in params])
    ).item()
    print(f"[clip] post-clip total grad norm = {post_norm:.4e}")
    # Allow a tiny float-rounding slack.
    assert post_norm <= max_norm + 1e-4, (
        f"post-clip norm {post_norm} exceeds max_norm {max_norm}"
    )
    assert post_norm < pre_norm, (
        f"clipping did not reduce norm (pre={pre_norm}, post={post_norm})"
    )
