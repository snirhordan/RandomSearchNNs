"""Modular transformer sequence encoder for RSNN/RWNN walk models.

Provides ``TransformerSeqLayer``, a pre-LN transformer encoder layer built on
``F.scaled_dot_product_attention`` that supports:

- ``attn_mode``: "full" (every token attends to every token) or "causal"
  (each token attends to itself and preceding tokens, LM-style).
- ``pos_enc``: "rope" applies rotary position embeddings to queries/keys
  inside attention (RoFormer, Su et al. 2021); "sinusoidal"/"none" expect the
  caller to handle (or skip) additive encodings outside the layer.
- key-padding masks for variable-length (padded) walk batches.

The layer is drop-in for both the adaptive RSNN models (padded DFS search
sequences with ``lengths``) and fixed-length RWNN walks (no padding mask).
Consumers stack layers in a ``ModuleList`` so per-layer node aggregation
(the RSNN scatter step) can run between layers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positional_encoding(max_len: int, d_model: int,
                                   device=None) -> torch.Tensor:
    """Standard additive sinusoidal encoding, shape ``(max_len, d_model)``."""
    positions = torch.arange(max_len, dtype=torch.float32,
                             device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


def rope_cos_sin(seq_len: int, head_dim: int, device=None,
                 base: float = 10000.0):
    """Rotary embedding tables ``cos, sin`` of shape ``(seq_len, head_dim//2)``.

    Position index is the walk-step number 0..seq_len-1, so attention scores
    depend on the relative offset between sequence positions (RoFormer eq. 16).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    inv_freq = base ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        / head_dim)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)              # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (..., seq_len, head_dim) by the RoPE tables.

    Pairs (x_{2i}, x_{2i+1}) are rotated by angle pos * theta_i; norms are
    preserved so only relative phase enters the q.k inner product.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class TransformerSeqLayer(nn.Module):
    """Pre-LN transformer encoder layer over walk-step token sequences.

    Parameters
    ----------
    d_model : int
        Token dimension (= hid_dim + pe_out_dim in the RSNN pipeline).
    nhead : int
        Attention heads; must divide ``d_model`` (and give even head_dim
        when ``pos_enc="rope"``).
    ffn_mult : int
        Feed-forward width multiplier (dim_feedforward = ffn_mult * d_model).
    attn_mode : str
        "full" or "causal".
    pos_enc : str
        "rope" rotates q/k in attention; "sinusoidal"/"none" are no-ops here
        (additive encodings are the caller's responsibility).
    dropout : float
        Dropout on attention output and FFN hidden activation.
    """

    def __init__(self, d_model: int, nhead: int, ffn_mult: int = 4,
                 attn_mode: str = "full", pos_enc: str = "sinusoidal",
                 dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"nhead={nhead} must divide d_model={d_model}")
        if attn_mode not in ("full", "causal"):
            raise ValueError(f"unknown attn_mode {attn_mode!r}")
        if pos_enc not in ("rope", "sinusoidal", "none"):
            raise ValueError(f"unknown pos_enc {pos_enc!r}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if pos_enc == "rope" and self.head_dim % 2 != 0:
            raise ValueError(
                f"pos_enc='rope' needs even head_dim; got {self.head_dim}")
        self.attn_mode = attn_mode
        self.pos_enc = pos_enc

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """``x``: (B, L, d_model); ``pad_mask``: (B, L) bool, True = PAD."""
        B, L, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)      # each (B, nhead, L, hd)

        if self.pos_enc == "rope":
            cos, sin = rope_cos_sin(L, self.head_dim, device=x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Compose causal + key-padding into one boolean mask (True = keep).
        attn_mask = None
        if pad_mask is not None:
            attn_mask = ~pad_mask[:, None, None, :]    # (B, 1, 1, L)
        if self.attn_mode == "causal":
            causal = torch.ones(L, L, dtype=torch.bool,
                                device=x.device).tril()
            attn_mask = causal if attn_mask is None else attn_mask & causal
        if attn_mask is not None and attn_mask.dim() == 4:
            # Guard fully-masked rows (padded query positions would softmax
            # over -inf only and yield NaN): let them attend to themselves.
            eye = torch.eye(L, dtype=torch.bool, device=x.device)
            attn_mask = attn_mask | eye[None, None, :, :]

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, L, self.d_model)
        x = x + self.dropout(self.out_proj(attn))
        x = x + self.ffn(self.norm2(x))
        return x


__all__ = [
    "TransformerSeqLayer",
    "sinusoidal_positional_encoding",
    "rope_cos_sin",
    "apply_rope",
]
