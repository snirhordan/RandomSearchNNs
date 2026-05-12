#!/usr/bin/env python3
"""Build QM9 preprocessing caches for multiple targets in one process.

Strategy: the per-molecule conversion (atoms, edge_index, edge_attr,
distances, neighbor-dict) is independent of which scalar target we pick;
only ``data.y`` differs. So we do ONE sweep through QM9 building the full
preprocessed list (with ``data.y_full`` retained), then for each requested
target we re-emit a shallow-cloned list with the right scalar y and dump
it to ``mols_<target>.pt``.

Usage::

    python3 runs/qm9/build_caches_multi.py --targets alpha homo lumo R2 \
        zpve U H G Cv --data_root ./data/qm9
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from generation.qm9 import (  # noqa: E402
    QM9_TARGET_INDEX,
    build_qm9_vocab,
    load_qm9,
    qm9_to_data,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--data_root", default="./data/qm9")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    for t in args.targets:
        if t not in QM9_TARGET_INDEX:
            raise SystemExit(f"unknown target {t!r}")

    cache_dir = Path(args.data_root) / "qm9_d_rwnn_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = f"_lim{args.limit}" if args.limit > 0 else ""

    todo = []
    for t in args.targets:
        cp = cache_dir / f"mols_{t}{limit_tag}.pt"
        if cp.exists():
            print(f"cache already exists for {t}: {cp}", flush=True)
            continue
        todo.append(t)
    if not todo:
        print("all requested caches already exist; nothing to do")
        return 0

    vocab_path = cache_dir / "vocab.pt"

    t0 = time.time()
    ds = load_qm9(root=args.data_root)
    if vocab_path.exists():
        vocab = torch.load(vocab_path)
    else:
        vocab = build_qm9_vocab(ds, tokenizer=None)
        torch.save(vocab, vocab_path)
    print(f"loaded raw QM9 ({len(ds)} mols), vocab={vocab}, "
          f"dt={time.time()-t0:.1f}s", flush=True)

    n_total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

    # Do a single conversion sweep, keyed on the first target. ``data.y_full``
    # holds the full 19-dim vector so we can re-derive per-target y cheaply.
    anchor = todo[0]
    print(f"single preprocess sweep (anchor target={anchor}, n={n_total})...",
          flush=True)
    ts = time.time()
    base_mols = []
    for i in range(n_total):
        d = qm9_to_data(ds[i], vocab=vocab,
                        add_distances=True, add_edge_attr=True,
                        target=anchor)
        base_mols.append(d)
        if (i + 1) % 20000 == 0:
            print(f"  preprocessed {i+1}/{n_total} "
                  f"({(i+1)/n_total*100:.1f}%), dt={time.time()-ts:.1f}s",
                  flush=True)
    print(f"preprocess done: {time.time()-ts:.1f}s", flush=True)
    del ds

    for t in todo:
        idx = QM9_TARGET_INDEX[t]
        cp = cache_dir / f"mols_{t}{limit_tag}.pt"
        ts = time.time()
        if t == anchor:
            mols = base_mols
        else:
            # Light per-molecule clone with patched y. Other fields are shared.
            mols = []
            for d in base_mols:
                e = copy.copy(d)
                e.y = d.y_full[0, idx].view(1).float()
                mols.append(e)
        print(f"  saving cache for {t} -> {cp}", flush=True)
        torch.save(mols, cp)
        print(f"  [{t}] done dt={time.time()-ts:.1f}s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
