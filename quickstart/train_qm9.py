#!/usr/bin/env python3
"""QM9 training for d-RWNN (variant A: PE-append).

Trains an RSNN_LSTM on QM9 (single-target regression) with the four-flag
matrix:

    --distances {0,1}  --mol_edge_feat {0,1}

Per-step add_edge_feat is built from RBF-expanded pairwise Euclidean
distances (when ``distances=1``) and/or 3-channel bond features (when
``mol_edge_feat=1``); both are appended into ``walk_pe`` by
``utils.search.sample_walks_adaptive``.

The training loop minimises MSE on the chosen QM9 target and reports MAE
in the target's natural units (eV for U0/gap, Debye for mu).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

# Allow ``from generation.qm9 import ...`` when invoked from quickstart/.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generation.qm9 import (  # noqa: E402
    RBFExpansion,
    build_qm9_vocab,
    load_qm9,
    qm9_to_data,
)
from generation.scaffold_split import random_split  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.utils import scatter  # noqa: E402
from utils.search import sample_walks_adaptive, sample_dfs  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers wiring distances + bond features through into add_edge_feat
# ---------------------------------------------------------------------------


def edge_attr_to_dense(data) -> torch.Tensor:
    """Scatter ``data.edge_attr`` (E x 3) into a dense ``(N, N, 3)`` tensor."""
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        n = int(data.num_nodes)
    elif hasattr(data, "x") and data.x is not None:
        n = int(data.x.size(0))
    else:
        n = int(data.edge_index.max()) + 1
    e_attr = data.edge_attr
    if e_attr is None or e_attr.numel() == 0:
        return torch.zeros((n, n, 3), dtype=torch.float)
    if e_attr.dim() != 2 or e_attr.size(-1) != 3:
        raise ValueError(
            f"edge_attr_to_dense expects (E, 3); got shape {tuple(e_attr.shape)}"
        )
    out = torch.zeros((n, n, 3), dtype=torch.float)
    src = data.edge_index[0].long()
    dst = data.edge_index[1].long()
    out[src, dst] = e_attr.float()
    out[dst, src] = e_attr.float()
    return out


def build_add_edge_feat(
    data,
    distances: int,
    mol_edge_feat: int,
    rbf: Optional[RBFExpansion] = None,
) -> Optional[torch.Tensor]:
    """Build the ``add_edge_feat`` tensor for one ``Data``."""
    parts = []
    if distances:
        if rbf is None:
            raise ValueError("distances=1 requires rbf=RBFExpansion(...)")
        if not hasattr(data, "distances") or data.distances is None:
            raise ValueError("distances=1 requires data.distances")
        parts.append(rbf(data.distances))
    if mol_edge_feat:
        parts.append(edge_attr_to_dense(data))
    if not parts:
        return None
    return torch.cat(parts, dim=-1).float()


def compute_pe_in_dim(walk_type: str, w: int, distances: int,
                      mol_edge_feat: int, rbf_K: int = 16) -> int:
    """Return the model's ``pe_in_dim`` for the given config."""
    bonus = rbf_K * distances + 3 * mol_edge_feat
    if walk_type in ("walk", "walk_ada"):
        return 2 * w + bonus
    # RSNN DFS-based search (sample_dfs): produces only an edge encoding of
    # width s (=w here).
    if walk_type in ("dfs", "bfs", "search"):
        return w + bonus
    if walk_type in ("walk_mdlr", "walk_rum",
                     "walk_mdlr_ada", "walk_rum_ada"):
        return bonus
    raise ValueError(f"unknown walk_type {walk_type!r}")


# ---------------------------------------------------------------------------
# Regression head: RSNN_LSTM clone WITHOUT the final sigmoid.
# ---------------------------------------------------------------------------


class RSNN_LSTM_Reg(nn.Module):
    """Bidirectional LSTM-based RSNN with a linear regression readout."""

    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers,
                 n_emb, reduce, dropout=0.0):
        super().__init__()
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(
            nn.LSTM(hid_dim + pe_out_dim, hid_dim, 1,
                    batch_first=True, bidirectional=True))
        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                nn.LSTM(2 * hid_dim, hid_dim, 1,
                        batch_first=True, bidirectional=True))
        self.readout = nn.ModuleList()
        self.readout.append(nn.Linear(2 * hid_dim, 2 * hid_dim))
        self.readout.append(nn.Linear(2 * hid_dim, out_dim))

        self.pe_encoding = nn.Linear(pe_in_dim, pe_out_dim)
        self.embedding = nn.Embedding(n_emb, hid_dim, n_emb - 1)

        self.reduce = reduce
        self.num_layers = num_layers
        # Dropout applied to graph-level features before readout MLP. Param-
        # count-neutral training-time regularization (nn.Dropout has no params).
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, batch):
        walk_emb = batch.walk_emb
        walk_ids = batch.walk_ids
        encoding = batch.walk_pe
        lengths = batch.lengths.cpu()

        graph_ns = [torch.max(walk_ids[i, :, :])
                    for i in range(walk_ids.shape[0])]
        walk_ids = walk_ids[:, :, :torch.max(lengths)]
        walk_ids_proc = []
        for i in range(walk_ids.shape[0]):
            if i == 0:
                walk_ids_proc.append(
                    torch.zeros((1, walk_ids.shape[1], walk_ids.shape[2]),
                                dtype=int).to(walk_emb.device))
            else:
                mult = sum(graph_ns[:i]) + i
                walk_ids_proc.append(
                    torch.ones((1, walk_ids.shape[1], walk_ids.shape[2]),
                               dtype=int).to(walk_emb.device) * mult)

        walk_ids_flat = torch.flatten(walk_ids, start_dim=0, end_dim=2)
        walk_ids_proc = torch.flatten(
            walk_ids + torch.cat(walk_ids_proc, dim=0),
            start_dim=0, end_dim=1)
        walk_ids_proc_flat = torch.flatten(walk_ids_proc,
                                            start_dim=0, end_dim=1)
        walk_ids_proc_flat_masked = walk_ids_proc_flat[walk_ids_flat != -1]

        x = torch.cat([self.embedding(walk_emb),
                       self.pe_encoding(encoding)], dim=-1)

        for l in range(self.num_layers):
            x = pack_padded_sequence(x, lengths,
                                     batch_first=True,
                                     enforce_sorted=False)
            if l == 0:
                x, h = self.rnn_layers[l](x)
            else:
                x, h = self.rnn_layers[l](x, h)
            x, _ = pad_packed_sequence(x, batch_first=True)

            node_agg = torch.flatten(x, start_dim=0, end_dim=1)
            node_agg = node_agg[walk_ids_flat != -1, :]
            node_agg = scatter(node_agg, walk_ids_proc_flat_masked,
                               dim=0, reduce='mean')

            x_flat = torch.flatten(x, start_dim=0, end_dim=1)
            if l != self.num_layers - 1:
                x_flat[walk_ids_flat != -1, :] = node_agg[
                    walk_ids_proc_flat_masked, :]
                x = x_flat.reshape(x.shape)

        graph_ids = torch.cat([
            torch.ones((graph_ns[i] + 1, ), dtype=int) * i
            for i in range(walk_ids.shape[0])
        ]).to(x.device)
        x = scatter(node_agg, graph_ids, dim=0, reduce=self.reduce)

        x = self.dropout(x)
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        # NO sigmoid -- regression head.
        return x


# ---------------------------------------------------------------------------
# Dataset wrapper: preprocessed Data + per-epoch walk sampling.
# ---------------------------------------------------------------------------


class QM9WalkDataset(Dataset):
    """Wraps a list of preprocessed ``Data`` objects.

    Each ``__getitem__`` clones the underlying graph, re-runs adaptive walk
    sampling, and returns a ``Data`` with ``walk_emb``/``walk_ids``/
    ``walk_pe``/``lengths`` attached.

    Collation-incompatible attrs (``distances``, ``pos``, ``y_full``, ``z``,
    ``_neighbor_dict``) are stripped before return so PyG ``Batch`` can
    collate cleanly.
    """

    _STRIP_KEYS = ("distances", "pos", "y_full", "z", "_neighbor_dict",
                   "idx", "name", "y_orig")

    def __init__(self, mols, vocab, rbf, distances, mol_edge_feat,
                 m, w, max_len, walk_type="walk_ada"):
        self.mols = mols
        self.vocab = vocab
        self.rbf = rbf
        self.distances = int(distances)
        self.mol_edge_feat = int(mol_edge_feat)
        self.m = int(m)
        self.w = int(w)
        self.max_len = int(max_len)
        self.walk_type = str(walk_type)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, i):
        d = self.mols[i].clone()
        if hasattr(d, "_neighbor_dict"):
            del d._neighbor_dict
        add_ef = build_add_edge_feat(d, self.distances, self.mol_edge_feat,
                                     rbf=self.rbf)
        n = d.x.shape[0]
        if self.walk_type == "search":
            # RSNN DFS-based search: pad to ``max_len`` and record ``lengths``.
            d = sample_dfs(d, self.m, self.w, self.max_len,
                           self.vocab, add_edge_feat=add_ef)
        else:
            d = sample_walks_adaptive(d, self.m, n, self.w, False,
                                      self.max_len, self.vocab,
                                      add_edge_feat=add_ef)
        for k in self._STRIP_KEYS:
            if hasattr(d, k):
                delattr(d, k)
        return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QM9 d-RWNN trainer (variant A).")
    p.add_argument(
        "--target",
        choices=[
            "mu", "alpha", "homo", "lumo", "gap", "R2", "zpve",
            "U0", "U", "H", "G", "Cv",
        ],
        default="U0",
    )
    p.add_argument("--distances", type=int, choices=[0, 1], default=0)
    p.add_argument("--mol_edge_feat", type=int, choices=[0, 1], default=0)
    p.add_argument("--rbf_K", type=int, default=16)
    p.add_argument("--rbf_cutoff", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--early_stopping", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--h_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--w", type=int, default=8)
    p.add_argument("--reduce", choices=["mean", "sum", "max"], default="mean")
    p.add_argument("--walk_type", default="walk_ada")
    p.add_argument("--n_splits", type=int, default=1)
    p.add_argument("--device_idx", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_root", default="./data/qm9")
    p.add_argument("--out_root", default="./runs/qm9")
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, limit dataset to first N molecules (debug).")
    p.add_argument(
        "--run_subdir",
        default="",
        help=(
            "If non-empty, place outputs under <out_root>/<run_subdir>/<target> "
            "instead of the default <out_root>/d<d>_m<m>/<target> layout. "
            "Useful for the RSNN m-sweep where the cfg_tag does not encode m."
        ),
    )
    p.add_argument(
        "--split",
        choices=["random", "cormorant"],
        default="random",
        help="Split protocol. 'random' = sklearn-style 60/20/20 (legacy). "
             "'cormorant' = fixed 100k/17748/13083 from EGNN's Anderson tarball.",
    )
    p.add_argument(
        "--cormorant_data_dir",
        default="./external/egnn/qm9/temp/qm9",
        help="Directory holding Cormorant's train.npz/valid.npz/test.npz "
             "(used only when --split cormorant).",
    )
    p.add_argument(
        "--lr_scheduler",
        choices=["none", "cosine"],
        default="none",
        help="LR schedule. 'cosine' = CosineAnnealingLR(T_max=epochs), "
             "stepped once per epoch.",
    )
    p.add_argument(
        "--use_egnn_normalization",
        type=int, choices=[0, 1], default=0,
        help="If 1, load meann/MAD from --norm_constants_json and use EGNN's "
             "(label - meann) / MAD normalization with L1Loss. If 0, fall back "
             "to legacy z-score on combined targets with MSELoss.",
    )
    p.add_argument(
        "--norm_constants_json",
        default="",
        help="JSON file with normalization.<target>.{meann,MAD} keys. "
             "Required when --use_egnn_normalization 1.",
    )
    # --- Training-optimization flags (CS230 RNN cheatsheet) ---
    p.add_argument(
        "--grad_clip",
        type=float, default=0.0,
        help="L2-norm gradient clipping max value (0=off). Mitigates exploding "
             "gradients in deep/recurrent stacks.",
    )
    p.add_argument(
        "--weight_decay",
        type=float, default=0.0,
        help="L2 regularization (passed to optimizer as weight_decay).",
    )
    p.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "rmsprop"],
        default="adam",
        help="Optimizer choice. 'adamw' decouples weight-decay from the gradient "
             "update; 'rmsprop' is the classic RNN optimizer.",
    )
    p.add_argument(
        "--lstm_init",
        choices=["default", "orthogonal"],
        default="default",
        help="LSTM weight init. 'orthogonal' initializes all weight_hh_* with "
             "orthogonal matrices and sets forget-gate biases to 1.0 "
             "(Jozefowicz et al. trick) — mitigates vanishing gradients.",
    )
    p.add_argument(
        "--dropout",
        type=float, default=0.0,
        help="LSTM hidden-state dropout (only effective when num_layers >= 2 "
             "via PyTorch's native LSTM dropout between stacked blocks).",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int, default=0,
        help="Linear LR warmup for this many epochs before --lr_scheduler kicks in.",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()

    # --- seeds ---
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")

    cfg_tag = f"d{args.distances}_m{args.mol_edge_feat}"
    if args.run_subdir:
        run_dir = Path(args.out_root) / args.run_subdir / args.target
    else:
        run_dir = Path(args.out_root) / cfg_tag / args.target
    run_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{cfg_tag}/{args.target}] {msg}", flush=True)

    log(f"device={device} seed={SEED} cwd={Path.cwd()}")

    # --- data ---
    # Cache preprocessed mols list to disk so the 4-config matrix can share
    # the (target-specific) preprocessed dataset and skip the ~5min preproc
    # on every config.
    cache_dir = Path(args.data_root) / "qm9_d_rwnn_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = f"_lim{args.limit}" if args.limit > 0 else ""
    cache_path = cache_dir / f"mols_{args.target}{limit_tag}.pt"
    vocab_path = cache_dir / "vocab.pt"

    t0 = time.time()
    if cache_path.exists() and vocab_path.exists():
        log(f"loading preprocessed cache from {cache_path}")
        mols = torch.load(cache_path)
        vocab = torch.load(vocab_path)
        log(f"loaded cache ({len(mols)} mols, vocab={vocab}), "
            f"dt={time.time()-t0:.1f}s")
        n_total = len(mols)
    else:
        ds = load_qm9(root=args.data_root)
        vocab = build_qm9_vocab(ds, tokenizer=None)
        log(f"loaded raw QM9 ({len(ds)} mols), vocab={vocab}, "
            f"dt={time.time()-t0:.1f}s")

        n_total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
        log(f"preprocessing {n_total} mols (target={args.target}, "
            f"add_distances=True, add_edge_attr=True)...")
        t0 = time.time()
        mols = []
        for i in range(n_total):
            d = qm9_to_data(ds[i], vocab=vocab,
                            add_distances=True, add_edge_attr=True,
                            target=args.target)
            mols.append(d)
            if (i + 1) % 20000 == 0:
                log(f"  preprocessed {i+1}/{n_total} "
                    f"({(i+1)/n_total*100:.1f}%), dt={time.time()-t0:.1f}s")
        log(f"preprocess done: {time.time()-t0:.1f}s for {n_total} mols")
        del ds

        # Save cache for sibling configs.
        try:
            log(f"saving preprocessed cache to {cache_path}")
            torch.save(mols, cache_path)
            torch.save(vocab, vocab_path)
        except Exception as e:
            log(f"  cache save failed: {e}")

    # Compute max_len across the preprocessed set.
    max_len = int(max(d.x.shape[0] for d in mols))
    log(f"max_len={max_len}")

    # --- splits ---
    if args.split == "cormorant":
        # Fixed 100k / 17748 / 13083 Anderson split from EGNN's npz tarball.
        # PyG Data.idx is 0-indexed gdb_idx; Cormorant npz['index'] is 1-indexed.
        cdir = Path(args.cormorant_data_dir)
        ctrain = {int(i) - 1 for i in np.load(cdir / "train.npz")["index"]}
        cval   = {int(i) - 1 for i in np.load(cdir / "valid.npz")["index"]}
        ctest  = {int(i) - 1 for i in np.load(cdir / "test.npz")["index"]}
        # If cache pre-dates the .idx-preservation patch in qm9_to_data,
        # recover idx positionally from a fresh PyG QM9 load. qm9_to_data
        # iterates PyG in order so cache mols[i] corresponds to PyG QM9[i].
        if not hasattr(mols[0], "idx") or mols[0].idx is None:
            log("cache mols lack .idx; recovering positionally from PyG QM9")
            from torch_geometric.datasets import QM9 as PyGQM9
            pyg = PyGQM9(root=args.data_root)
            assert len(pyg) == len(mols), (
                f"PyG QM9 len={len(pyg)} != cache len={len(mols)}; "
                "cache and PyG dataset are out of sync, please rebuild cache.")
            for k, d in enumerate(mols):
                d.idx = pyg[k].idx.clone()
            del pyg
        idx_lookup = {int(d.idx.item()): k for k, d in enumerate(mols)}
        train_idxs = [idx_lookup[g] for g in ctrain if g in idx_lookup]
        valid_idxs = [idx_lookup[g] for g in cval   if g in idx_lookup]
        test_idxs  = [idx_lookup[g] for g in ctest  if g in idx_lookup]
        log(f"cormorant split: train={len(train_idxs)} "
            f"valid={len(valid_idxs)} test={len(test_idxs)}")
        assert len(train_idxs) == 100000, f"train != 100000 ({len(train_idxs)})"
        assert len(valid_idxs) == 17748,  f"valid != 17748 ({len(valid_idxs)})"
        assert len(test_idxs)  == 13083,  f"test != 13083 ({len(test_idxs)})"
        # Pack into the same structure produced by random_split (n_splits=1).
        splits = {"train": [train_idxs], "valid": [valid_idxs],
                  "test":  [test_idxs]}
    else:
        # --- splits (random, 60/20/20) ---
        splits = random_split(len(mols), test_size=0.2, val_size=0.2,
                              n_splits=max(1, args.n_splits), random_state=0)

    # --- target standardisation: stabilise loss; report MAE in original units.
    if args.use_egnn_normalization:
        if not args.norm_constants_json:
            raise SystemExit(
                "--use_egnn_normalization 1 requires --norm_constants_json")
        with open(args.norm_constants_json) as f:
            norm = json.load(f)["normalization"][args.target]
        y_mean = float(norm["meann"])
        y_std  = float(norm["MAD"])
        log(f"loaded EGNN normalization: meann={y_mean:.6f} MAD={y_std:.6f}")
    else:
        # Legacy: z-score over the combined target set.
        ys = torch.stack([d.y for d in mols]).view(-1).float()
        y_mean = float(ys.mean().item())
        y_std  = float(ys.std().item())
        log(f"legacy target stats: mean={y_mean:.4f} std={y_std:.4f}")
    for d in mols:
        d.y = ((d.y - y_mean) / max(y_std, 1e-8)).float()

    # --- RBF (CPU-side; pickled by DataLoader workers) ---
    rbf = RBFExpansion(K=args.rbf_K, cutoff=args.rbf_cutoff)

    pe_in_dim = compute_pe_in_dim(args.walk_type, args.w,
                                  args.distances, args.mol_edge_feat,
                                  rbf_K=args.rbf_K)
    pe_out_dim = 16
    log(f"pe_in_dim={pe_in_dim} pe_out_dim={pe_out_dim} "
        f"vocab_size={len(vocab)}")

    run_metrics = {
        "config": {
            "target": args.target,
            "distances": args.distances,
            "mol_edge_feat": args.mol_edge_feat,
            "rbf_K": args.rbf_K,
            "rbf_cutoff": args.rbf_cutoff,
            "seed": SEED,
            "epochs": args.epochs,
            "early_stopping": args.early_stopping,
            "batch_size": args.batch_size,
            "h_dim": args.h_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "m": args.m,
            "w": args.w,
            "reduce": args.reduce,
            "walk_type": args.walk_type,
            "n_splits": args.n_splits,
            "max_len": max_len,
            "n_total": n_total,
            "pe_in_dim": pe_in_dim,
            "pe_out_dim": pe_out_dim,
            "y_mean": y_mean,
            "y_std": y_std,
            "vocab": vocab,
            # Apples-to-apples protocol recap (added for qm9_compare sweep):
            "split": args.split,
            "cormorant_data_dir": (
                args.cormorant_data_dir if args.split == "cormorant" else None),
            "lr_scheduler": args.lr_scheduler,
            "use_egnn_normalization": bool(args.use_egnn_normalization),
            "norm_constants_json": (
                args.norm_constants_json if args.use_egnn_normalization else None),
            # Loss is L1 when EGNN normalization is on (matches EGNN's main_qm9.py),
            # otherwise legacy MSE.
            "loss": ("L1" if args.use_egnn_normalization else "MSE"),
            # Normalization formula records *which* recipe produced y_mean/y_std:
            #   - "egnn_meann_mad": train-fold meann + Mean Absolute Deviation,
            #     loaded from norm_constants_json (single source of truth).
            #   - "combined_z_score": legacy z-score over combined train+val+test
            #     targets (in-distribution scaling only, not strict apples-to-apples
            #     with EGNN).
            "normalization_recipe": (
                "egnn_meann_mad" if args.use_egnn_normalization
                else "combined_z_score"),
            # Training-optimization knobs (CS230 RNN cheatsheet).
            "grad_clip": args.grad_clip,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "lstm_init": args.lstm_init,
            "dropout": args.dropout,
            "warmup_epochs": args.warmup_epochs,
        },
        "splits": [],
    }

    t_global = time.time()
    peak_mem_mb = 0.0
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            log(f"reset_peak_memory_stats failed: {e}")

    test_maes, valid_maes = [], []
    for s in range(max(1, args.n_splits)):
        split_t0 = time.time()
        train_idx = splits["train"][s]
        valid_idx = splits["valid"][s]
        test_idx = splits["test"][s]
        log(f"split {s}: train={len(train_idx)} valid={len(valid_idx)} "
            f"test={len(test_idx)}")

        train_mols = [mols[i] for i in train_idx]
        valid_mols = [mols[i] for i in valid_idx]
        test_mols = [mols[i] for i in test_idx]

        train_ds = QM9WalkDataset(train_mols, vocab, rbf,
                                  args.distances, args.mol_edge_feat,
                                  args.m, args.w, max_len,
                                  walk_type=args.walk_type)
        valid_ds = QM9WalkDataset(valid_mols, vocab, rbf,
                                  args.distances, args.mol_edge_feat,
                                  args.m, args.w, max_len,
                                  walk_type=args.walk_type)
        test_ds = QM9WalkDataset(test_mols, vocab, rbf,
                                 args.distances, args.mol_edge_feat,
                                 args.m, args.w, max_len,
                                 walk_type=args.walk_type)

        common = dict(batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      persistent_workers=(args.num_workers > 0),
                      pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True, **common)
        valid_loader = DataLoader(valid_ds, shuffle=False, **common)
        test_loader = DataLoader(test_ds, shuffle=False, **common)

        model = RSNN_LSTM_Reg(pe_in_dim, pe_out_dim, args.h_dim, 1,
                              args.num_layers, len(vocab),
                              args.reduce, dropout=args.dropout).to(device)
        # Optional orthogonal LSTM init + forget-gate bias = 1.0 (Jozefowicz trick).
        if args.lstm_init == "orthogonal":
            for lstm in model.rnn_layers:
                for name, p in lstm.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias_ih" in name or "bias_hh" in name:
                        n = p.numel() // 4
                        p.data[n:2*n].fill_(1.0)  # forget-gate chunk
            log(f"applied orthogonal init + forget-gate bias=1 to "
                f"{len(model.rnn_layers)} LSTM layers")

        # Optimizer factory.
        opt_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), **opt_kwargs)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        elif args.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), **opt_kwargs)
        # L1 (MAE) loss matches EGNN when training in z-scored target space.
        criterion = nn.L1Loss() if args.use_egnn_normalization else nn.MSELoss()
        # LR schedule: optional linear warmup, then optional cosine decay.
        sub_schedulers = []
        if args.warmup_epochs > 0:
            sub_schedulers.append(torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0,
                total_iters=args.warmup_epochs))
        if args.lr_scheduler == "cosine":
            t_cos = max(1, args.epochs - args.warmup_epochs)
            sub_schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_cos))
        if len(sub_schedulers) == 0:
            scheduler = None
        elif len(sub_schedulers) == 1:
            scheduler = sub_schedulers[0]
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=sub_schedulers,
                milestones=[args.warmup_epochs])

        best_valid_mae = float("inf")
        best_state = None
        stop_counter = 0
        epoch_log = []

        for epoch in range(args.epochs):
            if stop_counter >= args.early_stopping:
                log(f"split {s} early stop at epoch {epoch}")
                break
            t0 = time.time()
            model.train()
            train_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch).squeeze(-1)
                loss = criterion(out, batch.y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                train_losses.append(float(loss.item()))
            train_mse = float(np.mean(train_losses))
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            # Compute valid MAE in ORIGINAL units (un-standardise).
            model.eval()
            abs_errs = []
            with torch.no_grad():
                for batch in valid_loader:
                    batch = batch.to(device)
                    out_norm = model(batch).squeeze(-1)
                    out = out_norm * y_std + y_mean
                    y_real = batch.y.view(-1) * y_std + y_mean
                    abs_errs.append((out - y_real).abs().detach().cpu())
            valid_mae = float(torch.cat(abs_errs).mean().item()) \
                if abs_errs else float("nan")

            dt = time.time() - t0
            improved = valid_mae < best_valid_mae - 1e-6
            if improved:
                best_valid_mae = valid_mae
                best_state = copy.deepcopy(model.state_dict())
                stop_counter = 0
            else:
                stop_counter += 1
            log(f"split {s} epoch {epoch:3d} lr={current_lr:.2e} "
                f"train_mse={train_mse:.4f} valid_mae={valid_mae:.4f} "
                f"dt={dt:.1f}s {'*' if improved else ' '}")
            epoch_log.append({
                "epoch": epoch,
                "lr": current_lr,
                "train_mse": train_mse,
                "valid_mae": valid_mae,
                "dt_sec": dt,
                "improved": improved,
            })

        # Final test eval with best weights.
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        abs_errs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out_norm = model(batch).squeeze(-1)
                out = out_norm * y_std + y_mean
                y_real = batch.y.view(-1) * y_std + y_mean
                abs_errs.append((out - y_real).abs().detach().cpu())
        test_mae = float(torch.cat(abs_errs).mean().item()) \
            if abs_errs else float("nan")
        valid_maes.append(best_valid_mae)
        test_maes.append(test_mae)

        # Save best checkpoint.
        ckpt_path = run_dir / f"split{s}_best.pt"
        torch.save({
            "state_dict": (best_state if best_state is not None
                           else model.state_dict()),
            "best_valid_mae": best_valid_mae,
            "test_mae": test_mae,
            "split": s,
            "config": run_metrics["config"],
        }, ckpt_path)

        if device.type == "cuda":
            try:
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                peak_mem_mb = max(peak_mem_mb, float(peak))
            except Exception:
                pass

        split_dt = time.time() - split_t0
        run_metrics["splits"].append({
            "split": s,
            "epochs_run": len(epoch_log),
            "best_valid_mae": best_valid_mae,
            "test_mae": test_mae,
            "split_dt_sec": split_dt,
            "epochs": epoch_log,
        })
        log(f"split {s} DONE best_valid_mae={best_valid_mae:.4f} "
            f"test_mae={test_mae:.4f} split_dt={split_dt:.1f}s")

    total_dt = time.time() - t_global
    run_metrics["summary"] = {
        "mean_test_mae": float(np.mean(test_maes)) if test_maes else float("nan"),
        "std_test_mae": float(np.std(test_maes)) if test_maes else 0.0,
        "mean_valid_mae": float(np.mean(valid_maes)) if valid_maes else float("nan"),
        "total_wall_sec": total_dt,
        "peak_gpu_mem_mb": peak_mem_mb,
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(run_metrics, f, indent=2)
    log(f"DONE mean_test_mae={run_metrics['summary']['mean_test_mae']:.4f} "
        f"total={total_dt:.1f}s peak_gpu_mem={peak_mem_mb:.1f}MB")
    log(f"metrics saved -> {metrics_path}")
    return 0


__all__ = [
    "build_add_edge_feat",
    "compute_pe_in_dim",
    "edge_attr_to_dense",
    "QM9WalkDataset",
    "RSNN_LSTM_Reg",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
