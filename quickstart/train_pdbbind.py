#!/usr/bin/env python3
"""PDBbind training for atom-level RSNN (protein-ligand binding affinity regression).

Trains ``RSNN_TRSF_Reg`` (or ``RSNN_LSTM_Reg``) on a preprocessed PDBbind
split (see ``generation/pdbbind.py``) using the protein-ligand walk sampler
(``utils.search.sample_protein_ligand``): every walk is a deterministic
protein-pocket prefix followed by a random ligand DFS walk, so within-walk
attention mixes protein and ligand context around the binding interface.

Label is ``neglog_aff`` (pK units). Targets are optionally standardized with
TRAIN-split mean/std (``--standardize``); training minimises MSE on the
standardized target, and all reported metrics (Pearson r, Spearman rho,
RMSE, MAE) are computed after inverting back to pK units. Model selection is
early-stopping on validation RMSE; the test split is scored once, with the
best-validation checkpoint.

This module deliberately reuses the shared backbone from
``quickstart.train_qm9`` (``RSNN_TRSF_Reg``, ``RSNN_LSTM_Reg``,
``compute_pe_in_dim``, ``build_add_edge_feat``) rather than redefining it.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import Dataset

# Allow ``from generation.pdbbind import ...`` when invoked from quickstart/.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generation.pdbbind import load_pdbbind_cache, load_vocab  # noqa: E402
from generation.qm9 import RBFExpansion  # noqa: E402
from quickstart.train_qm9 import (  # noqa: E402
    RSNN_LSTM_Reg,
    RSNN_TRSF_Reg,
    build_add_edge_feat,
    compute_pe_in_dim,
)
from torch_geometric.loader import DataLoader  # noqa: E402
from utils.search import sample_protein_ligand  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset wrapper: preprocessed atom-level Data + per-epoch walk sampling.
# ---------------------------------------------------------------------------


class PDBbindWalkDataset(Dataset):
    """Wraps a list of preprocessed PDBbind ``Data`` objects.

    Each ``__getitem__`` clones the underlying atom-level graph, re-runs
    ``sample_protein_ligand`` (deterministic protein prefix + fresh random
    ligand DFS walk), and returns a ``Data`` with ``walk_emb``/``walk_ids``/
    ``walk_pe``/``lengths``/``y`` attached. Resampling happens on every call,
    so the ligand walks vary epoch to epoch while the protein prefix -- being
    a deterministic function of ``residue_id``/interface distance -- is
    stable by construction.

    Collation-incompatible per-atom attrs are stripped before return so PyG
    ``Batch`` can collate cleanly (mirrors ``tests/test_pdbbind_integration.py``).
    """

    _STRIP_KEYS = ("x", "pos", "z", "edge_index", "edge_attr", "segment",
                   "residue_id", "num_nodes", "pdb_id", "ligand_bond_edge_index",
                   "torsion_order", "torsion_seg")

    def __init__(self, data_list, vocab, m, s, max_len, ligand_max_len,
                 angles=0, dihedrals=0, angle_K=8, dihedral_K=4,
                 distances=0, mol_edge_feat=0, rbf=None,
                 ligand_graph='distance', protein_order='allatom',
                 emit_xyz=0):
        self.data_list = data_list
        self.vocab = vocab
        self.m = int(m)
        self.s = int(s)
        self.max_len = int(max_len)
        self.ligand_max_len = int(ligand_max_len)
        self.angles = int(angles)
        self.dihedrals = int(dihedrals)
        self.angle_K = int(angle_K)
        self.dihedral_K = int(dihedral_K)
        self.distances = int(distances)
        self.mol_edge_feat = int(mol_edge_feat)
        self.rbf = rbf
        self.ligand_graph = ligand_graph
        self.protein_order = protein_order
        # When set, sample_protein_ligand attaches data.walk_xyz (m, max_len, 3)
        # for the geometric attention bias. Intentionally NOT in _STRIP_KEYS so
        # it survives PyG collation (like walk_pe/walk_emb).
        self.emit_xyz = int(emit_xyz)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        d = self.data_list[i].clone()

        add_ef = None
        if self.distances or self.mol_edge_feat:
            if self.distances:
                # Pairwise Euclidean distances (N x N), symmetric, zero diag
                # -- same convention as generation.qm9.qm9_to_data's
                # ``data.distances``, computed on the fly here since PDBbind
                # Data objects don't carry it by default.
                d.distances = torch.cdist(d.pos, d.pos)
            add_ef = build_add_edge_feat(d, self.distances, self.mol_edge_feat,
                                         rbf=self.rbf)

        d = sample_protein_ligand(
            d, m=self.m, s=self.s, max_len=self.max_len, vocab=self.vocab,
            add_edge_feat=add_ef, ligand_max_len=self.ligand_max_len,
            angles=bool(self.angles), dihedrals=bool(self.dihedrals),
            angle_K=self.angle_K, dihedral_K=self.dihedral_K, vectorize=True,
            ligand_graph=self.ligand_graph, protein_order=self.protein_order,
            emit_xyz=bool(self.emit_xyz),
        )
        for k in self._STRIP_KEYS:
            if hasattr(d, k):
                delattr(d, k)
        return d


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    """Pearson r, Spearman rho, RMSE, MAE, all in the caller's (pK) units.

    Correlations are defined 0.0 on degenerate inputs (fewer than 2 samples,
    or zero variance in either array) instead of NaN, so metrics.json always
    contains finite numbers even on tiny eval splits.
    """
    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2))) if preds.size else float("nan")
    mae = float(np.mean(np.abs(preds - trues))) if preds.size else float("nan")
    if preds.size >= 2 and np.std(preds) > 1e-12 and np.std(trues) > 1e-12:
        pearson_r = float(pearsonr(preds, trues)[0])
        spearman_rho = float(spearmanr(preds, trues)[0])
    else:
        pearson_r, spearman_rho = 0.0, 0.0
    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "rmse": rmse,
        "mae": mae,
    }


def _evaluate(model, loader, device, y_mean: float, y_std: float) -> dict:
    """Run ``model`` over ``loader``, un-standardize, return pK-unit metrics."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1)
            pred = (out * y_std + y_mean).detach().cpu().numpy()
            true = (batch.y.view(-1) * y_std + y_mean).detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    return _compute_metrics(preds, trues)


# ---------------------------------------------------------------------------
# Model construction (shared between --print_params_only and real training).
# ---------------------------------------------------------------------------


def _build_model(args, pe_in_dim: int, pe_out_dim: int, vocab_size: int):
    if args.base == "transformer":
        return RSNN_TRSF_Reg(
            pe_in_dim, pe_out_dim, args.h_dim, 1, args.num_layers, vocab_size,
            args.reduce, dropout=args.dropout, nhead=args.nhead,
            ffn_mult=args.ffn_mult, attn_mode=args.attn_mode,
            pos_enc=args.pos_enc,
            geom_bias=bool(args.geom_bias), geom_rbf_K=args.geom_rbf_K,
            geom_rbf_cutoff=args.geom_rbf_cutoff, geom_angle_K=args.geom_angle_K,
            geom_dihedral_K=args.geom_dihedral_K, geom_hidden=args.geom_hidden,
        )
    return RSNN_LSTM_Reg(
        pe_in_dim, pe_out_dim, args.h_dim, 1, args.num_layers, vocab_size,
        args.reduce, dropout=args.dropout,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDBbind atom-level RSNN trainer.")
    p.add_argument("--data_dir", default=None,
                   help="Directory containing train.pt/valid.pt/test.pt/vocab.json "
                        "(required unless --print_params_only).")
    p.add_argument("--out_dir", default="./runs/pdbbind/default")
    p.add_argument("--base", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--h_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--nhead", type=int, default=8,
                   help="Attention heads (transformer base only). Must divide "
                        "h_dim + pe_out_dim (=h_dim+16).")
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--pos_enc", choices=["sinusoidal", "rope", "none"], default="rope")
    p.add_argument("--attn_mode", choices=["full", "causal"], default="full")
    p.add_argument("--reduce", choices=["mean", "sum", "max"], default="sum")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                   help="'cosine' anneals lr over the epoch budget (per-batch steps). "
                        "'constant' (default) reproduces earlier runs byte-identically.")
    p.add_argument("--lr_min_frac", type=float, default=0.01,
                   help="Cosine floor as a fraction of --lr (eta_min). Ignored when constant.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=50,
                   help="Early stop after this many epochs without a val-RMSE "
                        "improvement.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=int, default=8,
                   help="Number of ligand DFS walks per complex (protein prefix "
                        "is shared/repeated across all m).")
    p.add_argument("--w", type=int, default=8,
                   help="Edge-encoding window size (sample_protein_ligand's 's').")
    p.add_argument("--max_len", type=int, default=256,
                   help="Total padded sequence length (protein prefix + ligand walk).")
    p.add_argument("--ligand_max_len", type=int, default=32,
                   help="Cap on the ligand DFS walk length.")
    p.add_argument("--ligand_graph", choices=["distance", "bond"], default="distance",
                   help="Ligand DFS traversal connectivity (ABLATION, additive, "
                        "default-off): 'distance' (default) = unchanged behavior, "
                        "ligand neighbors from the 4.5A distance graph. 'bond' = "
                        "ligand neighbors from data.ligand_bond_edge_index (see "
                        "generation.pdbbind.augment_cache_with_bonds); complexes "
                        "missing that attribute silently fall back to 'distance' "
                        "for that complex only.")
    p.add_argument("--protein_order", choices=["allatom", "torsion"], default="allatom",
                   help="Protein-prefix ordering (ABLATION, additive, default-off): "
                        "'allatom' (default) = unchanged behavior, flat "
                        "(residue_id, atom_index) prefix, no segment gating. "
                        "'torsion' = torsion-aware prefix from "
                        "data.torsion_order/torsion_seg (see "
                        "generation.pdbbind.build_protein_torsion_order / "
                        "augment_cache_with_torsion), with hard-reset segment "
                        "gating on the edge-window/angle/dihedral features; "
                        "complexes missing torsion_order/torsion_seg silently "
                        "fall back to 'allatom' for that complex only.")
    p.add_argument("--angles", type=int, choices=[0, 1], default=1)
    p.add_argument("--dihedrals", type=int, choices=[0, 1], default=1)
    p.add_argument("--angle_K", type=int, default=8)
    p.add_argument("--dihedral_K", type=int, default=4)
    p.add_argument("--distances", type=int, choices=[0, 1], default=0,
                   help="If 1, append an RBF expansion of on-the-fly pairwise "
                        "Euclidean distances (data.pos) into walk_pe, mirroring "
                        "QM9's --distances=1 path.")
    p.add_argument("--mol_edge_feat", type=int, choices=[0, 1], default=0,
                   help="QM9-only feature (3-channel bond one-hot edge_attr); "
                        "PDBbind's edge_attr is a 1-channel inverse-distance "
                        "weight, so this is unsupported here and must stay 0.")
    # --- Geometric pairwise attention bias (transformer base only) ---
    # E(3)-invariant bias injected into attention logits, computed once over
    # walk positions from walk_xyz (distance RBF + flanking bond angle +
    # dihedral). Distinct from the --angles/--dihedrals walk_pe CONCATENATION
    # path above; keep the geom_ prefix so the two mechanisms stay separate.
    p.add_argument("--geom_bias", type=int, choices=[0, 1], default=0,
                   help="Transformer base only. If 1, add an E(3)-invariant "
                        "pairwise geometric attention bias (distance RBF + "
                        "flanking bond angles + dihedral over walk positions) "
                        "to every attention layer. Auto-enables walk_xyz "
                        "emission from sample_protein_ligand. Adds ~1.4k params. "
                        "Default 0 keeps prior behavior.")
    p.add_argument("--geom_rbf_K", type=int, default=16,
                   help="Geometric-bias distance RBF channels (default 16).")
    p.add_argument("--geom_rbf_cutoff", type=float, default=5.0,
                   help="Geometric-bias RBF cutoff in Angstrom (default 5.0). "
                        "Protein pockets span larger than QM9 molecules, so "
                        "this is worth sweeping, e.g. {5,10,15}.")
    p.add_argument("--geom_angle_K", type=int, default=8,
                   help="Geometric-bias bond-angle cos-basis size (default 8).")
    p.add_argument("--geom_dihedral_K", type=int, default=4,
                   help="Geometric-bias dihedral basis size (default 4; "
                        "doubled sin+cos).")
    p.add_argument("--geom_hidden", type=int, default=32,
                   help="Geometric-bias MLP hidden width (default 32).")
    p.add_argument("--emit_xyz", type=int, choices=[0, 1], default=0,
                   help="If 1, sample_protein_ligand emits walk_xyz "
                        "(m, max_len, 3): xyz of the atom at each walk position "
                        "(zeros at padding) for the geometric attention bias. "
                        "Auto-enabled by --geom_bias 1. Default 0.")
    p.add_argument("--standardize", type=int, choices=[0, 1], default=1,
                   help="If 1, standardize the target with TRAIN-split mean/std "
                        "before training (MSE on standardized targets); metrics "
                        "are always reported in pK units.")
    p.add_argument("--print_params_only", action="store_true",
                   help="Construct the model for the given config, print "
                        "n_params, and exit 0 without touching any data. Reads "
                        "--data_dir/vocab.json for the real vocab size if given, "
                        "else assumes a vocab size of 40.")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.mol_edge_feat:
        raise SystemExit(
            "--mol_edge_feat 1 is not supported for PDBbind: edge_attr is a "
            "1-channel inverse-distance weight (generation/pdbbind.py), not "
            "the 3-channel bond one-hot build_add_edge_feat's mol_edge_feat "
            "path expects. Leave --mol_edge_feat 0."
        )

    # The geometric attention bias is a transformer-only mechanism that
    # consumes per-position walk_xyz (auto-emitted by sample_protein_ligand).
    # Fail fast on the LSTM base (mirrors train_qm9.py's --geom_bias guard).
    if args.geom_bias and args.base != "transformer":
        raise SystemExit(
            "--geom_bias 1 requires --base transformer (the geometric "
            "attention bias is injected into transformer attention logits)."
        )

    pe_out_dim = 16
    pe_in_dim = compute_pe_in_dim(
        "search", w=args.w, distances=args.distances,
        mol_edge_feat=args.mol_edge_feat, angles=args.angles,
        dihedrals=args.dihedrals, angle_K=args.angle_K,
        dihedral_K=args.dihedral_K,
    )

    if args.print_params_only:
        vocab_size = 40
        if args.data_dir:
            vocab_path = Path(args.data_dir) / "vocab.json"
            if vocab_path.exists():
                vocab_size = len(load_vocab(str(vocab_path)))
        model = _build_model(args, pe_in_dim, pe_out_dim, vocab_size)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"n_params={n_params} vocab_size={vocab_size} pe_in_dim={pe_in_dim} "
              f"pe_out_dim={pe_out_dim} d_model={args.h_dim + pe_out_dim}")
        return 0

    if not args.data_dir:
        raise SystemExit("--data_dir is required unless --print_params_only.")

    # --- seeds ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "a")

    def log(msg: str) -> None:
        line = f"[pdbbind] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"device={device} seed={args.seed} out_dir={out_dir}")

    # --- data ---
    data_dir = Path(args.data_dir)
    train_data = load_pdbbind_cache(str(data_dir / "train.pt"))
    valid_data = load_pdbbind_cache(str(data_dir / "valid.pt"))
    test_data = load_pdbbind_cache(str(data_dir / "test.pt"))
    vocab = load_vocab(str(data_dir / "vocab.json"))
    assert vocab.get("UNK") == 0, "vocab contract requires UNK==0"
    assert vocab.get("PAD") == len(vocab) - 1, (
        "vocab contract requires PAD at the last index (RSNN_TRSF_Reg / "
        "RSNN_LSTM_Reg hardcode padding_idx=n_emb-1)"
    )
    log(f"loaded train={len(train_data)} valid={len(valid_data)} "
        f"test={len(test_data)} vocab_size={len(vocab)}")

    log(f"ligand_graph={args.ligand_graph}")
    if args.ligand_graph == "bond":
        for split_name, split_data in (("train", train_data), ("valid", valid_data),
                                        ("test", test_data)):
            n_total = len(split_data)
            n_missing = sum(
                1 for d in split_data
                if getattr(d, "ligand_bond_edge_index", None) is None
                or d.ligand_bond_edge_index.numel() == 0
            )
            frac = (n_missing / n_total) if n_total else 0.0
            log(f"ligand_graph=bond {split_name}: {n_missing}/{n_total} "
                f"({frac:.1%}) complexes lack ligand_bond_edge_index and will "
                f"fall back to distance-graph ligand traversal")
            if frac > 0.05:
                log(f"WARNING: {frac:.1%} of {split_name} complexes fell back to "
                    f"distance-graph ligand traversal under --ligand_graph bond "
                    f"(expected near-0%; re-run generation.pdbbind's "
                    f"augment_cache_with_bonds if this cache predates it)")

    log(f"protein_order={args.protein_order}")
    if args.protein_order == "torsion":
        for split_name, split_data in (("train", train_data), ("valid", valid_data),
                                        ("test", test_data)):
            n_total = len(split_data)
            n_missing = sum(
                1 for d in split_data
                if getattr(d, "torsion_order", None) is None
                or d.torsion_order.numel() == 0
            )
            frac = (n_missing / n_total) if n_total else 0.0
            log(f"protein_order=torsion {split_name}: {n_missing}/{n_total} "
                f"({frac:.1%}) complexes lack torsion_order/torsion_seg and "
                f"will fall back to the flat allatom protein prefix")
            if frac > 0.05:
                log(f"WARNING: {frac:.1%} of {split_name} complexes fell back to "
                    f"the allatom protein prefix under --protein_order torsion "
                    f"(expected near-0%; re-run generation.pdbbind's "
                    f"augment_cache_with_torsion if this cache predates it)")

    rbf = RBFExpansion(K=16, cutoff=5.0) if args.distances else None

    log(f"pe_in_dim={pe_in_dim} pe_out_dim={pe_out_dim}")

    # --- target standardization: TRAIN-split mean/std only (no leakage). ---
    if args.standardize:
        ys = torch.stack([d.y for d in train_data]).view(-1).float()
        y_mean = float(ys.mean().item())
        y_std = float(ys.std().item())
        y_std = max(y_std, 1e-8)
        for split in (train_data, valid_data, test_data):
            for d in split:
                d.y = ((d.y - y_mean) / y_std).float()
    else:
        y_mean, y_std = 0.0, 1.0
    log(f"target stats (train): standardize={bool(args.standardize)} "
        f"mean={y_mean:.4f} std={y_std:.4f}")

    # --- datasets / loaders ---
    _ds_kw = dict(
        m=args.m, s=args.w, max_len=args.max_len,
        ligand_max_len=args.ligand_max_len, angles=args.angles,
        dihedrals=args.dihedrals, angle_K=args.angle_K,
        dihedral_K=args.dihedral_K, distances=args.distances,
        mol_edge_feat=args.mol_edge_feat, rbf=rbf,
        ligand_graph=args.ligand_graph, protein_order=args.protein_order,
        # walk_xyz is required by the geometric bias; auto-enable it there
        # (still honors an explicit --emit_xyz 1). PDBbind has a single sampler,
        # so no walk_type branch to gate on (unlike QM9).
        emit_xyz=int(args.emit_xyz or args.geom_bias),
    )
    train_ds = PDBbindWalkDataset(train_data, vocab, **_ds_kw)
    valid_ds = PDBbindWalkDataset(valid_data, vocab, **_ds_kw)
    test_ds = PDBbindWalkDataset(test_data, vocab, **_ds_kw)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 persistent_workers=(args.num_workers > 0),
                 pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    valid_loader = DataLoader(valid_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    # --- model ---
    model = _build_model(args, pe_in_dim, pe_out_dim, len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model={type(model).__name__} n_params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Cosine annealing over the full epoch budget, stepped per batch. Matters most for
    # short/compute-capped runs, which otherwise stop mid-flight at the full learning rate
    # and select a noisy checkpoint. Default 'constant' keeps prior runs byte-identical.
    scheduler = None
    if args.lr_schedule == "cosine":
        total_steps = max(1, args.epochs * max(1, len(train_loader)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * args.lr_min_frac)
        log(f"lr_schedule=cosine total_steps={total_steps} "
            f"eta_min={args.lr * args.lr_min_frac:.2e}")

    config = dict(vars(args))
    config.update({
        "n_params": n_params,
        "pe_in_dim": pe_in_dim,
        "pe_out_dim": pe_out_dim,
        "vocab_size": len(vocab),
        "d_model": args.h_dim + pe_out_dim,
        "y_mean": y_mean,
        "y_std": y_std,
        "device": str(device),
        "n_train": len(train_data),
        "n_valid": len(valid_data),
        "n_test": len(test_data),
    })

    # --- training loop ---
    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    val_history = []

    t_global = time.time()
    for epoch in range(args.epochs):
        if patience_counter >= args.patience:
            log(f"early stop at epoch {epoch} (patience={args.patience})")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.item()))
        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")

        val_metrics = _evaluate(model, valid_loader, device, y_mean, y_std)
        dt = time.time() - t0

        improved = val_metrics["rmse"] < best_val_rmse - 1e-6
        if improved:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        log(f"epoch {epoch:4d} train_mse={train_mse:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_pearson={val_metrics['pearson_r']:.4f} "
            f"val_spearman={val_metrics['spearman_rho']:.4f} dt={dt:.1f}s "
            f"{'*' if improved else ' '}")
        val_history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "dt_sec": dt,
            "improved": improved,
            **val_metrics,
        })

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())

    test_metrics = _evaluate(model, test_loader, device, y_mean, y_std)
    total_dt = time.time() - t_global
    log(f"DONE best_epoch={best_epoch} best_val_rmse={best_val_rmse:.4f} "
        f"test_rmse={test_metrics['rmse']:.4f} test_mae={test_metrics['mae']:.4f} "
        f"test_pearson={test_metrics['pearson_r']:.4f} "
        f"test_spearman={test_metrics['spearman_rho']:.4f} total={total_dt:.1f}s")

    torch.save({
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_metrics": test_metrics,
        "config": config,
    }, out_dir / "model.pt")

    metrics_out = {
        "config": config,
        "val_history": val_history,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_metrics": test_metrics,
        "total_wall_sec": total_dt,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    log(f"metrics saved -> {out_dir / 'metrics.json'}")
    log_file.close()
    return 0


__all__ = [
    "PDBbindWalkDataset",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
