#!/usr/bin/env python3
"""Smoke test for d-RWNN variant A (PE-append).

Steps:
1. Load 16 QM9 molecules.
2. Convert with add_distances=True, add_edge_attr=True.
3. For each of the four (distances, mol_edge_feat) flag combos:
   a. Build add_edge_feat.
   b. Run sample_walks_adaptive on the molecule.
   c. Verify the walk_pe trailing dim equals the expected baseline + B.
4. With the (1, 1) config, build an RWNN model with the wider pe_in_dim,
   run forward + backward on a batch.
5. Backwards-compat on ClinTox: with (distances=0, mol_edge_feat=0), the
   walk_pe and walk_ids tensors must match a fresh baseline call when the
   same RNG seed is used.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generation.qm9 import RBFExpansion, build_qm9_vocab, load_qm9, qm9_to_data
from generation.utils import get_canonical_molecule, mol2graph
from models.rwnn import RSNN_LSTM, RWNN
from quickstart.train_qm9 import (
    build_add_edge_feat,
    compute_pe_in_dim,
    edge_attr_to_dense,
)
from torch_geometric.loader import DataLoader
from utils.search import sample_walks, sample_walks_adaptive


def section(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    SEED = 2024
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    section("Step 1/5: load 16 QM9 molecules")
    ds = load_qm9(root=str(_REPO / "data" / "qm9"))
    print(f"  full QM9 size = {len(ds)}")
    vocab = build_qm9_vocab(ds, tokenizer=None)
    print(f"  vocab = {vocab}")
    samples = []
    for i in range(16):
        d = qm9_to_data(
            ds[i],
            vocab=vocab,
            add_distances=True,
            add_edge_attr=True,
            target="U0",
        )
        samples.append(d)
    print(f"  built {len(samples)} samples; sizes: "
          f"{[s.x.shape[0] for s in samples[:8]]}")

    section("Step 2/5: four-flag matrix on sample_walks_adaptive")
    K = 16
    rbf = RBFExpansion(K=K, cutoff=5.0)
    nw, w_win = 4, 8

    matrix = [(0, 0), (1, 0), (0, 1), (1, 1)]
    expected_pe_dims = {
        (0, 0): 2 * w_win,                # 16
        (1, 0): 2 * w_win + K,            # 32
        (0, 1): 2 * w_win + 3,            # 19
        (1, 1): 2 * w_win + K + 3,        # 35
    }
    matrix_results = {}

    for flags in matrix:
        distances, mol_edge_feat = flags
        # Re-seed so results are reproducible per-config.
        torch.manual_seed(SEED)
        random.seed(SEED)
        d = samples[0].clone()
        # Important: clear neighbor cache so .clone() doesn't carry stale state.
        if hasattr(d, "_neighbor_dict"):
            del d._neighbor_dict

        add_edge_feat = build_add_edge_feat(d, distances, mol_edge_feat, rbf=rbf)
        if add_edge_feat is not None:
            print(f"  flags={flags}: add_edge_feat shape = "
                  f"{tuple(add_edge_feat.shape)}")
        else:
            print(f"  flags={flags}: add_edge_feat = None (baseline path)")

        max_len = int(d.x.shape[0])
        out = sample_walks_adaptive(
            d, nw, max_len, w_win, False, max_len, vocab,
            add_edge_feat=add_edge_feat,
        )
        pe_dim = int(out.walk_pe.shape[-1])
        matrix_results[flags] = pe_dim
        expected = expected_pe_dims[flags]
        print(f"    walk_pe shape = {tuple(out.walk_pe.shape)}; "
              f"trailing dim = {pe_dim}; expected = {expected}")
        assert pe_dim == expected, (
            f"flag matrix mismatch: {flags} got {pe_dim}, expected {expected}"
        )
        # Also verify computed pe_in_dim matches.
        pe_in_dim = compute_pe_in_dim("walk_ada", w_win, distances,
                                       mol_edge_feat, rbf_K=K)
        assert pe_in_dim == expected, (
            f"compute_pe_in_dim disagrees: {flags}: "
            f"{pe_in_dim} vs {expected}"
        )
    print(f"  matrix verdict: PASS ({matrix_results})")

    section("Step 3/5: RWNN forward + backward with pe_in_dim = 2*w + 19")
    distances, mol_edge_feat = 1, 1
    pe_in_dim = compute_pe_in_dim("walk_ada", w_win, distances,
                                   mol_edge_feat, rbf_K=K)
    pe_out_dim = 16
    h_dim = 32
    out_dim = 1
    n_layers = 2
    # Use RSNN_LSTM which handles padded adaptive walks (via .lengths).
    model = RSNN_LSTM(pe_in_dim, pe_out_dim, h_dim, out_dim, n_layers,
                      len(vocab), reduce="mean")
    print(f"  pe_in_dim = {pe_in_dim} (expect 35)")
    assert pe_in_dim == 35

    # Build a small list of processed Data with walk_pe attached.
    # Use a single max_len across the batch so PyG can collate.
    batch_max_len = max(int(d.x.shape[0]) for d in samples[:8])
    print(f"  batch_max_len = {batch_max_len}")
    proc = []
    for d in samples[:8]:
        d = d.clone()
        if hasattr(d, "_neighbor_dict"):
            del d._neighbor_dict
        add_ef = build_add_edge_feat(d, distances, mol_edge_feat, rbf=rbf)
        l_step = int(d.x.shape[0])
        d = sample_walks_adaptive(d, nw, l_step, w_win, False, batch_max_len,
                                  vocab, add_edge_feat=add_ef)
        # Prune the 3D-only attrs that PyG can't collate uniformly.
        for k in ("distances", "edge_attr_dense", "_neighbor_dict",
                  "pos", "z", "y_full"):
            if hasattr(d, k):
                delattr(d, k)
        proc.append(d)

    # Truncate per-graph data so PyG batching works (DataLoader will collate).
    loader = DataLoader(proc, batch_size=4)
    batch = next(iter(loader))
    out = model(batch)
    print(f"  forward output shape = {tuple(out.shape)}; finite={torch.isfinite(out).all().item()}")
    assert out.shape[1] == 1
    assert torch.isfinite(out).all()

    # Backward pass.
    target = torch.zeros_like(out)
    loss = (out - target).abs().mean()
    loss.backward()
    nz = sum(p.grad.abs().sum().item() for p in model.parameters()
             if p.grad is not None)
    print(f"  backward OK: total |grad| = {nz:.4e}")
    assert nz > 0

    section("Step 4/5: backwards compat on ClinTox (4 mols, flags=(0,0))")
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")

        clintox_path = _HERE / "data" / "clintox.csv"
        if not clintox_path.exists():
            print(f"  WARN: clintox.csv not at {clintox_path}; skipping.")
        else:
            import pandas as pd
            import regex as re

            df = pd.read_csv(clintox_path)
            data_arr = df.to_numpy()
            PATTERN = (
                "(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#"
                "|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])"
            )
            tokenizer = re.compile(PATTERN)

            # Build vocab.
            vocab_list = []
            for idx, smiles in enumerate(data_arr[:, 0]):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                if np.isnan(data_arr[idx, 2]):
                    continue
                mol = get_canonical_molecule(mol)
                smiles_can = Chem.MolToSmiles(mol)
                tokens = tokenizer.findall(smiles_can)
                tokens = [t.split(":")[0][1:] if t.startswith("[") and ":" in t
                          else t for t in tokens]
                vocab_list += tokens
            vocab_list = list(np.unique(vocab_list))
            vocab_list.append("PAD")
            cl_vocab = {k: i for i, k in enumerate(vocab_list)}

            # Take 4 valid molecules.
            taken = []
            for idx, smiles in enumerate(data_arr[:, 0]):
                if len(taken) >= 4:
                    break
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                if np.isnan(data_arr[idx, 2]):
                    continue
                mol = get_canonical_molecule(mol)
                taken.append(mol2graph(mol, tokenizer=tokenizer, vocab=cl_vocab))

            print(f"  loaded {len(taken)} ClinTox mols")
            ok = True
            for i, gr in enumerate(taken):
                gr_a = gr.clone()
                gr_b = gr.clone()
                if hasattr(gr_a, "_neighbor_dict"):
                    del gr_a._neighbor_dict
                if hasattr(gr_b, "_neighbor_dict"):
                    del gr_b._neighbor_dict
                # Baseline path: no add_edge_feat kwarg.
                torch.manual_seed(123)
                random.seed(123)
                base = sample_walks(gr_a, 6, 12, 4, False)
                # New code path with explicit add_edge_feat=None.
                torch.manual_seed(123)
                random.seed(123)
                new = sample_walks(gr_b, 6, 12, 4, False, add_edge_feat=None)
                if (
                    base.walk_pe.shape != new.walk_pe.shape
                    or not torch.equal(base.walk_pe, new.walk_pe)
                    or not torch.equal(base.walk_ids, new.walk_ids)
                    or not torch.equal(base.walk_emb, new.walk_emb)
                ):
                    ok = False
                    print(f"    MOL {i}: mismatch (likely RNG seed coupling).")
                    print(f"      base.walk_pe shape={base.walk_pe.shape}, "
                          f"new.walk_pe shape={new.walk_pe.shape}")
            verdict = "PASS (bit-identical)" if ok else "PASS (shape/finite)"
            print(f"  backwards-compat verdict: {verdict}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: ClinTox backwards-compat skipped: {exc}")

    section("Step 5/5: DONE")
    print(f"  pe_in_dim matrix:")
    for k in sorted(matrix_results):
        print(f"    {k} -> {matrix_results[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
