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
    # odd d_model: cosine half has floor(d_model/2) columns
    pe[:, 1::2] = torch.cos(positions * div_term[:d_model // 2])
    return pe


def rope_cos_sin(seq_len: int, head_dim: int, device=None,
                 base: float = 10000.0):
    """Rotary embedding tables ``cos, sin`` of shape ``(seq_len, head_dim//2)``.

    The position index is the step number 0..seq_len-1, so attention scores
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


class GeometricAttentionBias(nn.Module):
    """E(3)-invariant pairwise geometric attention bias over walk positions.

    For every ordered pair of walk positions ``(i, j)`` this builds a feature
    vector from the rigid-motion-invariant geometry of the four positions
    ``i-1, i, j, j+1`` and maps it through a small MLP to a per-head additive
    bias added to the attention logits (T5/Graphormer-style relative bias).

    Per-pair features (all `SE(3)`-invariant; the dihedral sin channels flip
    sign under reflection, exactly like ``utils.search._dihedral_basis``):

    - pairwise distance ``|x_i - x_j|`` -> Gaussian RBF (``rbf_K`` channels);
    - bond angle at vertex i = ``angle(x_{i-1}, x_i, x_j)`` -> cos basis
      (``angle_K`` channels);
    - bond angle at vertex j = ``angle(x_i, x_j, x_{j+1})`` -> cos basis
      (``angle_K`` channels);
    - dihedral ``torsion(x_{i-1}, x_i, x_j, x_{j+1})`` -> sin/cos basis
      (``2*dihedral_K`` channels).

    For an adjacent pair ``j = i+1`` these reduce exactly (same arg order, same
    eps/clamp) to the consecutive bond-angle / dihedral features emitted by
    ``sample_dfs``; the math mirrors the batched helpers in ``utils.search``.

    Boundary positions (i=0 has no ``i-1``; the last real position has no
    ``j+1``) zero the channels that need the missing neighbor; padded positions
    are masked too. Coordinates are never read past the real walk (``walk_xyz``
    pads with zeros and every norm is eps-guarded, so no NaN can arise).

    Parameters
    ----------
    nhead : int
        Attention heads (bias is produced per head).
    rbf_K : int
        Gaussian RBF channels for the pairwise distance.
    rbf_cutoff : float
        RBF center range ``[0, rbf_cutoff]`` (mirrors ``RBFExpansion``).
    angle_K : int
        Bond-angle cos-basis size (default 8, per DimeNet).
    dihedral_K : int
        Dihedral basis size (default 4); doubled (sin+cos) -> ``2*dihedral_K``.
    hidden : int
        Hidden width of the bias MLP.
    eps : float
        Numerical guard (matches the scalar/batched helpers in utils.search).
    """

    def __init__(self, nhead: int, rbf_K: int = 16, rbf_cutoff: float = 5.0,
                 angle_K: int = 8, dihedral_K: int = 4, hidden: int = 32,
                 eps: float = 1e-8):
        super().__init__()
        self.nhead = nhead
        self.angle_K = angle_K
        self.dihedral_K = dihedral_K
        self.rbf_K = rbf_K
        self.eps = eps
        # acos'(+/-1) = -inf, so the backward w.r.t. coordinates is taken from a
        # clamp strictly inside (-1, 1) while the forward value uses the [-1, 1]
        # clamp (see _pair_angle).
        self._acos_eps = 1e-6
        # Gaussian RBF centers/width (identical math to generation.qm9.RBFExpansion,
        # replicated as buffers to avoid a quickstart->model->generation import cycle).
        self.register_buffer("rbf_centers",
                             torch.linspace(0.0, rbf_cutoff, rbf_K))
        self.rbf_width = rbf_cutoff / rbf_K

        in_dim = rbf_K + 2 * angle_K + 2 * dihedral_K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, nhead),
        )

    def _basis(self, walk_xyz: torch.Tensor,
               pad_mask: torch.Tensor) -> torch.Tensor:
        """Pre-MLP per-pair basis features, shape ``(BN, L, L, in_dim)``.

        Exposed (separate from the MLP) so the adjacent-reduction and chirality
        properties are testable without depending on the random MLP weights.
        ``walk_xyz``: (BN, L, 3); ``pad_mask``: (BN, L) bool, True = PAD.
        """
        BN, L, _ = walk_xyz.shape
        eps = self.eps
        valid = ~pad_mask                                   # (BN, L) real step
        zcol = walk_xyz.new_zeros(BN, 1, 3)
        # Flanking neighbors via slice+pad (roll would wrap; never wrap).
        x_im1 = torch.cat([zcol, walk_xyz[:, :-1]], dim=1)  # x at i-1 (0 -> zeros)
        x_jp1 = torch.cat([walk_xyz[:, 1:], zcol], dim=1)   # x at j+1 (last -> zeros)
        fcol = valid.new_zeros(BN, 1)
        valid_im1 = torch.cat([fcol, valid[:, :-1]], dim=1) & valid  # i>0 and real
        valid_jp1 = torch.cat([valid[:, 1:], fcol], dim=1) & valid   # j<last and real

        # Pair grids: index i over dim 1, j over dim 2.
        xi = walk_xyz[:, :, None, :]                        # (BN, L, 1, 3)
        xj = walk_xyz[:, None, :, :]                        # (BN, 1, L, 3)
        xim1 = x_im1[:, :, None, :]                          # (BN, L, 1, 3)
        xjp1 = x_jp1[:, None, :, :]                          # (BN, 1, L, 3)

        # Distance + RBF (always defined for any two real positions).
        diff = xi - xj                                      # (BN, L, L, 3)
        dist = diff.norm(dim=-1)                            # (BN, L, L)
        rbf = torch.exp(-((dist[..., None] - self.rbf_centers) ** 2)
                        / (2 * self.rbf_width ** 2))         # (BN, L, L, rbf_K)

        # Angle at vertex i = angle(x_{i-1}, x_i, x_j): vertex x_i.
        theta_i = self._pair_angle(xim1, xi, xj)            # (BN, L, L)
        angle_i = self._angle_basis(theta_i)                # (BN, L, L, angle_K)
        # Angle at vertex j = angle(x_i, x_j, x_{j+1}): vertex x_j.
        theta_j = self._pair_angle(xi, xj, xjp1)
        angle_j = self._angle_basis(theta_j)
        # Dihedral over (x_{i-1}, x_i, x_j, x_{j+1}).
        phi = self._pair_dihedral(xim1, xi, xj, xjp1)       # (BN, L, L)
        dih = self._dihedral_basis(phi)                     # (BN, L, L, 2*dK)

        # Boundary/padding masks (zero channels that read a missing or padded
        # neighbor). angle_i = angle(x_{i-1}, x_i, x_j) needs i-1, i AND j real;
        # angle_j = angle(x_i, x_j, x_{j+1}) needs i, j AND j+1 real; the
        # dihedral needs all four. Self-pairs (i == j) are geometrically
        # degenerate, so all angle/dihedral channels are masked off the diagonal
        # (the distance/RBF channel stays: distance to self is well defined).
        dtype = walk_xyz.dtype
        m_i = valid_im1[:, :, None, None].to(dtype)             # i-1 & i real
        m_j = valid_jp1[:, None, :, None].to(dtype)             # j & j+1 real
        vi = valid[:, :, None, None].to(dtype)                  # i real
        vj = valid[:, None, :, None].to(dtype)                  # j real
        m_pair = vi * vj                                        # both real
        offdiag = (~torch.eye(L, dtype=torch.bool, device=walk_xyz.device))[
            None, :, :, None].to(dtype)
        rbf = rbf * m_pair
        angle_i = angle_i * m_i * vj * offdiag
        angle_j = angle_j * m_j * vi * offdiag
        dih = dih * m_i * m_j * offdiag

        return torch.cat([rbf, angle_i, angle_j, dih], dim=-1)

    def forward(self, walk_xyz: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """Pairwise bias, shape ``(BN, nhead, L, L)``.

        ``walk_xyz``: (BN, L, 3); ``pad_mask``: (BN, L) bool, True = PAD.
        Computed once per forward (geometry is identical across layers).
        """
        feats = self._basis(walk_xyz, pad_mask)             # (BN, L, L, in_dim)
        bias = self.mlp(feats)                              # (BN, L, L, nhead)
        return bias.permute(0, 3, 1, 2).contiguous()        # (BN, nhead, L, L)

    def _pair_angle(self, pa: torch.Tensor, pb: torch.Tensor,
                    pc: torch.Tensor) -> torch.Tensor:
        """Bond angle at vertex pb between rays to pa and pc (broadcast grid).

        Same forward value as ``utils.search._batch_bond_angle`` (same eps,
        clamp). The backward is made finite at the degenerate cells where
        ``cos = +/-1`` (``acos`` has an infinite slope there): the value comes
        from the ``[-1, 1]`` clamp, the gradient from a strictly-interior clamp.
        """
        a = pa - pb
        b = pc - pb
        cos_t = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + self.eps)
        cos_v = cos_t.clamp(-1.0, 1.0)
        cos_g = cos_t.clamp(-1.0 + self._acos_eps, 1.0 - self._acos_eps)
        return torch.acos(cos_g) + (torch.acos(cos_v) - torch.acos(cos_g)).detach()

    def _angle_basis(self, theta: torch.Tensor) -> torch.Tensor:
        """cos(l*theta), l=1..angle_K (mirrors _batch_angle_basis)."""
        ls = torch.arange(1, self.angle_K + 1, dtype=theta.dtype,
                          device=theta.device)
        return torch.cos(theta[..., None] * ls)

    def _pair_dihedral(self, p0: torch.Tensor, p1: torch.Tensor,
                       p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """Dihedral over (p0, p1, p2, p3) (broadcast grid).

        Identical atan2 formula to ``utils.search._batch_dihedral`` (same
        clamp on |b2|); sign-aware (chirality) via the scalar triple product.
        """
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        b2_norm = b2.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        b2_hat = b2 / b2_norm
        m1 = torch.cross(n1, b2_hat, dim=-1)
        x = (n1 * n2).sum(-1)
        y = (m1 * n2).sum(-1)
        # atan2(0, 0) has a NaN gradient; it occurs on degenerate cells
        # (collinear triple / coincident central bond, e.g. the grid diagonal).
        # Feed atan2 a safe (1, 0) there -> phi = 0 (the value the masked output
        # discards anyway) while keeping the backward finite.
        deg = (x * x + y * y) < self.eps ** 2
        x = torch.where(deg, torch.ones_like(x), x)
        y = torch.where(deg, torch.zeros_like(y), y)
        return torch.atan2(y, x)

    def _dihedral_basis(self, phi: torch.Tensor) -> torch.Tensor:
        """[sin(l*phi), cos(l*phi)], l=1..dihedral_K (mirrors _batch_dihedral_basis)."""
        ls = torch.arange(1, self.dihedral_K + 1, dtype=phi.dtype,
                          device=phi.device)
        lp = phi[..., None] * ls
        return torch.cat([torch.sin(lp), torch.cos(lp)], dim=-1)


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
                pad_mask: torch.Tensor = None,
                attn_bias: torch.Tensor = None) -> torch.Tensor:
        """``x``: (B, L, d_model); ``pad_mask``: (B, L) bool, True = PAD;
        ``attn_bias``: (B, nhead, L, L) float added to logits, or None."""
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

        if attn_bias is not None:
            # SDPA adds a FLOAT mask to the logits but masks a BOOL mask, so to
            # inject an additive bias while keeping causal/padding/eye-guard we
            # build a float mask: bias where kept, -inf where masked. The
            # eye-guard keeps every query row with >=1 finite entry (no NaN).
            keep = attn_mask if attn_mask is not None \
                else torch.ones(B, 1, L, L, dtype=torch.bool, device=x.device)
            neg = torch.finfo(x.dtype).min
            float_mask = torch.where(
                keep, attn_bias.to(x.dtype),
                torch.full((), neg, dtype=x.dtype, device=x.device))
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, L, self.d_model)
        x = x + self.dropout(self.out_proj(attn))
        x = x + self.ffn(self.norm2(x))
        return x


__all__ = [
    "TransformerSeqLayer",
    "GeometricAttentionBias",
    "sinusoidal_positional_encoding",
    "rope_cos_sin",
    "apply_rope",
]
