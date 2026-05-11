"""QM9 preprocessing for d-RWNN (distance-augmented random-walk neural networks).

This module is purely additive: it does NOT alter ``generation/utils.py`` or any
existing pipeline component. It exposes:

- ``load_qm9``           -- thin wrapper around ``torch_geometric.datasets.QM9``
                           that always uses the pre-processed PyG release
                           (``qm9_v3.pt``) so we sidestep occasional RDKit SDF
                           parsing failures with newer rdkit versions.
- ``qm9_to_data``        -- convert one QM9 sample to a ``Data`` object whose
                           layout is compatible with the existing ClinTox /
                           ``mol2graph`` pipeline (``x``, ``x_emb``,
                           ``edge_index``, ``edge_attr``, plus optional
                           ``pos``/``distances`` for d-RWNN).
- ``build_qm9_vocab``    -- build an atom-symbol-keyed vocab over the dataset.
- ``RBFExpansion``       -- ``nn.Module`` implementing a Gaussian RBF basis
                           expansion of scalar Euclidean distances into a
                           ``K``-dimensional feature vector.

The integration steps that touch the sampler (``utils/search.py``) and the
model (``models/rwnn.py``) live on a different branch -- this file only sets
up the data side.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.datasets import QM9 as _PyGQM9


# ---------------------------------------------------------------------------
# QM9 element vocabulary helpers
# ---------------------------------------------------------------------------

# QM9 contains only H, C, N, O, F (atomic numbers 1, 6, 7, 8, 9).
QM9_Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# OGB allowable atom feature lists -- mirrors the indices used by
# ``generation/utils.py::atom_to_feature_vector`` so that downstream code that
# inspects ``data.x`` channels keeps the same semantics.
_POSSIBLE_ATOMIC_NUM_LIST = list(range(1, 119)) + ["misc"]
_POSSIBLE_CHIRALITY_IDX_UNSPEC = 0  # 'CHI_UNSPECIFIED'
_POSSIBLE_DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"]
_POSSIBLE_FORMAL_CHARGE_IDX_ZERO = 5  # neutral charge slot
_POSSIBLE_NUMH_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"]
_POSSIBLE_RAD_E_IDX_ZERO = 0
_POSSIBLE_HYB_LIST = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"]
_POSSIBLE_IS_AROMATIC = [False, True]
_POSSIBLE_IS_IN_RING = [False, True]


def _safe_index(lst, e):
    try:
        return lst.index(e)
    except ValueError:
        return len(lst) - 1


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


class _QM9Preprocessed(_PyGQM9):
    """``torch_geometric.datasets.QM9`` forced onto the no-rdkit code path.

    The newer rdkit versions sanitize-fail on a handful of QM9 SDF entries,
    which makes the standard ``QM9`` constructor crash mid-processing. The
    PyG-hosted ``qm9_v3.pt`` snapshot is already cleaned and contains the
    same 130,831 molecules with ``x``, ``pos``, ``z``, ``edge_index``,
    ``edge_attr``, ``y``, ``name``, ``idx`` per entry, so we always use it.
    """

    @property
    def raw_file_names(self):  # type: ignore[override]
        return ["qm9_v3.pt"]

    def download(self):  # type: ignore[override]
        path = download_url(self.processed_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):  # type: ignore[override]
        from torch_geometric.io import fs

        data_list = fs.torch_load(self.raw_paths[0])
        data_list = [Data(**d) for d in data_list]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])


def load_qm9(root: str = "./data/qm9"):
    """Load QM9 (130,831 molecules, 19 regression targets)."""
    return _QM9Preprocessed(root=root)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def build_qm9_vocab(dataset, tokenizer=None) -> dict:
    """Build an atom-symbol vocabulary for QM9.

    The QM9 pre-processed bundle does not ship per-molecule SMILES strings,
    so we fall back to indexing by element symbol (the spec explicitly
    permits this alternative). The returned dict has the form::

        {'C': 0, 'F': 1, 'H': 2, 'N': 3, 'O': 4, 'PAD': 5}

    The ``tokenizer`` argument is accepted for API symmetry with the ClinTox
    pipeline but is unused for QM9 -- we keep it in the signature so callers
    can swap loaders without changing call sites.
    """
    del tokenizer  # unused for QM9
    seen = set()
    for d in dataset:
        for z in d.z.tolist():
            if z in QM9_Z_TO_SYMBOL:
                seen.add(QM9_Z_TO_SYMBOL[z])
            else:
                seen.add(f"Z{z}")
    vocab_keys = sorted(seen)
    vocab_keys.append("PAD")
    return {k: i for i, k in enumerate(vocab_keys)}


# ---------------------------------------------------------------------------
# Per-sample conversion
# ---------------------------------------------------------------------------


def _build_x_ogb(z: torch.Tensor, x_qm9: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Synthesize a 9-dim OGB-style atom-feature tensor from QM9 fields.

    QM9's pre-processed ``x`` is ``[N, 11]`` = ``[5-class type one-hot]`` ++
    ``[atomic_number, aromatic, sp, sp2, sp3, num_hs]`` (floats). We map this
    back to OGB indices where possible. Unknown fields (chirality, formal
    charge, radical electrons, in-ring) default to neutral / unspecified
    slots.
    """
    n = z.size(0)
    # Degrees from edge_index
    if edge_index.numel() > 0:
        deg = torch.bincount(edge_index[0], minlength=n).tolist()
    else:
        deg = [0] * n

    aromatic_col = x_qm9[:, 6].long().tolist()  # 0/1 aromatic flag
    sp_col = x_qm9[:, 7].long().tolist()
    sp2_col = x_qm9[:, 8].long().tolist()
    sp3_col = x_qm9[:, 9].long().tolist()
    num_hs_col = x_qm9[:, 10].long().tolist()

    feats = []
    for i in range(n):
        atomic_num = int(z[i].item())
        if sp_col[i]:
            hyb_idx = _POSSIBLE_HYB_LIST.index("SP")
        elif sp2_col[i]:
            hyb_idx = _POSSIBLE_HYB_LIST.index("SP2")
        elif sp3_col[i]:
            hyb_idx = _POSSIBLE_HYB_LIST.index("SP3")
        else:
            hyb_idx = _POSSIBLE_HYB_LIST.index("misc")
        feats.append([
            _safe_index(_POSSIBLE_ATOMIC_NUM_LIST, atomic_num),
            _POSSIBLE_CHIRALITY_IDX_UNSPEC,
            _safe_index(_POSSIBLE_DEGREE_LIST, deg[i]),
            _POSSIBLE_FORMAL_CHARGE_IDX_ZERO,
            _safe_index(_POSSIBLE_NUMH_LIST, num_hs_col[i]),
            _POSSIBLE_RAD_E_IDX_ZERO,
            hyb_idx,
            _POSSIBLE_IS_AROMATIC.index(bool(aromatic_col[i])),
            _POSSIBLE_IS_IN_RING.index(False),
        ])
    return torch.tensor(feats, dtype=torch.float)


# Mapping between QM9 19-dim target tensor and the named targets we expose.
# Includes case-insensitive aliases for the standard 12 properties so that
# downstream callers can pass either ``R2``/``r2``, ``Cv``/``cv``, etc.
QM9_TARGET_INDEX = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "R2": 5,
    "zpve": 6,
    "U0": 7,
    "U": 8,
    "H": 9,
    "G": 10,
    "Cv": 11,
}


def qm9_to_data(
    pyg_data: Data,
    tokenizer=None,
    vocab: Optional[dict] = None,
    add_distances: bool = False,
    add_edge_attr: bool = False,
    rbf_K: int = 16,
    rbf_cutoff: float = 5.0,
    target: str = "U0",
) -> Data:
    """Convert one QM9 ``Data`` sample into the d-RWNN-compatible layout.

    Parameters
    ----------
    pyg_data : torch_geometric.data.Data
        A single sample from ``load_qm9()``; expected attrs: ``x`` (N x 11),
        ``pos`` (N x 3), ``z`` (N,), ``edge_index`` (2 x E), ``edge_attr``
        (E x 4 one-hot), ``y`` (1 x 19).
    tokenizer : Any, optional
        Unused for QM9 (kept for API symmetry); see ``build_qm9_vocab``.
    vocab : dict, optional
        Atom-symbol-keyed vocabulary; required if ``x_emb`` is needed by the
        downstream sampler.
    add_distances : bool, default False
        If True, attach ``data.distances`` (N x N) pairwise Euclidean
        distance matrix (in Angstroms).
    add_edge_attr : bool, default False
        If True, keep the bond ``edge_attr`` as a ``(E, 3)`` int64 tensor in
        OGB layout (bond_type, bond_stereo=NONE, is_conjugated=False). If
        False, ``edge_attr`` is filled with zeros so batching stays uniform.
    rbf_K, rbf_cutoff
        Forwarded for symmetry; the actual expansion is performed by
        ``RBFExpansion`` inside the model, not here.
    target : str
        One of the keys of ``QM9_TARGET_INDEX``; selects which scalar target
        is exposed as ``data.y`` (shape ``[1]``). The full ``[1, 19]`` target
        vector is preserved as ``data.y_full``.
    """
    del tokenizer, rbf_K, rbf_cutoff  # callers pass for API symmetry

    if target not in QM9_TARGET_INDEX:
        raise ValueError(f"Unknown QM9 target {target!r}; expected one of "
                         f"{list(QM9_TARGET_INDEX)}")

    z = pyg_data.z
    pos = pyg_data.pos.float()
    n = z.size(0)

    # Bond graph, undirected.
    edge_index = pyg_data.edge_index.long()
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    # OGB-style 9-dim atom feature tensor.
    x = _build_x_ogb(z, pyg_data.x, edge_index)

    # Bond features: pre-processed QM9 ships a 4-class one-hot
    # (SINGLE, DOUBLE, TRIPLE, AROMATIC). We compress into the 3-channel
    # OGB layout (bond_type_idx, stereo=STEREONONE=0, is_conjugated=0).
    qm9_edge_attr = pyg_data.edge_attr  # [E, 4]
    if qm9_edge_attr is not None and qm9_edge_attr.numel() > 0:
        # Re-order edge_attr to follow the (possibly re-oriented) edge_index.
        # ``to_undirected`` may reorder edges, so re-derive bond types from
        # the raw pair (start, end) -> bond_type_idx mapping.
        raw_ei = pyg_data.edge_index.long()
        raw_attr = qm9_edge_attr
        bond_type_idx = raw_attr.argmax(dim=-1)  # [E_raw]
        bond_lookup = {}
        for k in range(raw_ei.size(1)):
            i, j = int(raw_ei[0, k].item()), int(raw_ei[1, k].item())
            bond_lookup[(i, j)] = int(bond_type_idx[k].item())
        e = edge_index.size(1)
        bt = torch.zeros(e, dtype=torch.int64)
        for k in range(e):
            i, j = int(edge_index[0, k].item()), int(edge_index[1, k].item())
            bt[k] = bond_lookup.get((i, j), bond_lookup.get((j, i), 0))
        ogb_edge_attr = torch.stack(
            [bt, torch.zeros(e, dtype=torch.int64), torch.zeros(e, dtype=torch.int64)],
            dim=-1,
        ).float()
    else:
        ogb_edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float)

    if not add_edge_attr:
        # Fill with zeros (preserves shape so batched downstream code keeps a
        # consistent edge_attr signature even when bond features are unused).
        ogb_edge_attr = torch.zeros_like(ogb_edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=ogb_edge_attr)
    data.pos = pos
    data.z = z

    if vocab is not None:
        x_emb = torch.empty((n,), dtype=torch.long)
        for i, zi in enumerate(z.tolist()):
            sym = QM9_Z_TO_SYMBOL.get(int(zi), f"Z{int(zi)}")
            x_emb[i] = vocab[sym]
        data.x_emb = x_emb

    if add_distances:
        # Pairwise Euclidean distances (N x N), float, symmetric, zero diag.
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = diff.norm(dim=-1)
        data.distances = dist

    # Targets: extract the chosen scalar plus keep the full 19-dim tensor for
    # ablations / multi-target training later.
    y_full = pyg_data.y.float()  # [1, 19]
    data.y_full = y_full
    data.y = y_full[0, QM9_TARGET_INDEX[target]].view(1)

    # Mirror the neighbor-dict caching used by mol2graph so the samplers see
    # the same interface.
    from generation.utils import get_neighbor_dict
    data = get_neighbor_dict(data)
    return data


# ---------------------------------------------------------------------------
# Gaussian RBF expansion of distances
# ---------------------------------------------------------------------------


class RBFExpansion(nn.Module):
    """Gaussian radial-basis expansion of scalar distances.

    Parameters
    ----------
    K : int, default 16
        Number of basis functions (centers).
    cutoff : float, default 5.0
        Distance (in Angstroms) at which the last center is placed. Centers
        are evenly spaced in ``[0, cutoff]``.
    width : float, optional
        Gaussian width. Defaults to ``cutoff / K``.
    trainable : bool, default False
        If True, centers and width become learnable parameters.

    Forward signature
    -----------------
    Input : tensor of any shape ``(..., )`` of distances (last dim treated
    elementwise; no trailing 1-dim required).
    Output : tensor of shape ``(..., K)`` -- the input shape with a new
    trailing basis dimension.
    """

    def __init__(self, K: int = 16, cutoff: float = 5.0,
                 width: Optional[float] = None, trainable: bool = False):
        super().__init__()
        self.K = K
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, steps=K)
        if width is None:
            width = cutoff / K
        self.width = width
        if trainable:
            self.centers = nn.Parameter(centers)
            self.log_width = nn.Parameter(torch.log(torch.tensor(float(width))))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("log_width",
                                 torch.log(torch.tensor(float(width))))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # Broadcast: distances (..., 1) - centers (K,) -> (..., K)
        d = distances.unsqueeze(-1)
        w = torch.exp(self.log_width)
        return torch.exp(-((d - self.centers) ** 2) / (2.0 * w ** 2))


__all__ = [
    "load_qm9",
    "qm9_to_data",
    "build_qm9_vocab",
    "RBFExpansion",
    "QM9_Z_TO_SYMBOL",
    "QM9_TARGET_INDEX",
]
