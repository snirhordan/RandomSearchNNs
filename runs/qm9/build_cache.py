#!/usr/bin/env python3
"""Build the QM9 preprocessing cache for a target. Run BEFORE launching the
4-config parallel matrix so the children share the cache (no 5min repeat
preprocess per config)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generation.qm9 import build_qm9_vocab, load_qm9, qm9_to_data  # noqa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["U0", "gap", "mu"])
    p.add_argument("--data_root", default="./data/qm9")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    cache_dir = Path(args.data_root) / "qm9_d_rwnn_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = f"_lim{args.limit}" if args.limit > 0 else ""
    cache_path = cache_dir / f"mols_{args.target}{limit_tag}.pt"
    vocab_path = cache_dir / "vocab.pt"

    if cache_path.exists() and vocab_path.exists():
        print(f"cache already exists: {cache_path}")
        return 0

    t0 = time.time()
    ds = load_qm9(root=args.data_root)
    vocab = build_qm9_vocab(ds, tokenizer=None)
    print(f"loaded raw QM9 ({len(ds)} mols), vocab={vocab}, "
          f"dt={time.time()-t0:.1f}s")

    n_total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"preprocessing {n_total} mols (target={args.target})...")
    t0 = time.time()
    mols = []
    for i in range(n_total):
        d = qm9_to_data(ds[i], vocab=vocab,
                        add_distances=True, add_edge_attr=True,
                        target=args.target)
        mols.append(d)
        if (i + 1) % 20000 == 0:
            print(f"  preprocessed {i+1}/{n_total} "
                  f"({(i+1)/n_total*100:.1f}%), dt={time.time()-t0:.1f}s",
                  flush=True)
    print(f"preprocess done: {time.time()-t0:.1f}s for {n_total} mols")

    del ds
    print(f"saving cache to {cache_path}")
    torch.save(mols, cache_path)
    torch.save(vocab, vocab_path)
    print("cache saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
