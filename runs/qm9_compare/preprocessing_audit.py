#!/usr/bin/env python3
"""Preprocessing audit for the gap apples-to-apples comparison.

Run this BEFORE any training. Halts on failure so we never train on
mis-aligned data.

What it checks:
  1. Cormorant split files exist + have expected counts (100k/17748/13083).
  2. PyG QM9 loads + filters to 130,831 mols.
  3. PyG Data.idx ↔ Cormorant npz['index'] bijection (Data.idx == index - 1)
     covers EVERY molecule (no orphans on either side).
  4. Computes meann, MAD for `gap` using EGNN's exact formula on the
     Cormorant train fold (read directly from Cormorant npz, then converted
     to eV via the EGNN unit-conversion factor 27.2114).
  5. Verifies each existing EGNN gap baseline run's metrics.json config
     matches what we plan to reuse (nf=128, n_layers=7, attention=1).

Writes:
  runs/qm9_compare/preprocessing_audit.json   machine-readable constants
  runs/qm9_compare/preprocessing_audit.md     human summary

Exit 0 on green, 1 on any failure.
"""
from __future__ import annotations
import json
import hashlib
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "runs" / "qm9_compare"
CORMORANT_DIR = REPO / "external" / "egnn" / "qm9" / "temp" / "qm9"
PYG_QM9_ROOT = REPO / "data" / "qm9"
EGNN_RUNS_DIR = REPO / "runs" / "qm9_egnn"

# EGNN/Cormorant Hartree → eV factor (from external/egnn/qm9/dataset.py:14).
HARTREE_TO_EV = 27.2114
# zpve uses ×27211.4 (i.e., × 1000 to go to meV) but we focus on gap = eV.
TARGET = "gap"
EXPECTED = {"train": 100000, "valid": 17748, "test": 13083}


def fail(msg: str) -> None:
    print(f"AUDIT FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_of_int_list(values) -> str:
    h = hashlib.sha256()
    for v in sorted(int(x) for x in values):
        h.update(v.to_bytes(8, "little", signed=False))
    return h.hexdigest()


def load_cormorant_indices() -> dict:
    out = {}
    for split, expected_n in EXPECTED.items():
        p = CORMORANT_DIR / f"{split}.npz"
        if not p.exists():
            fail(f"missing {p}")
        npz = np.load(p)
        if "index" not in npz.files:
            fail(f"{p} lacks 'index' field; keys={npz.files}")
        idxs = np.asarray(npz["index"], dtype=np.int64)
        if len(idxs) != expected_n:
            fail(f"{split}: expected {expected_n} mols, got {len(idxs)}")
        # Cormorant is 1-indexed; PyG is 0-indexed.
        out[split] = idxs - 1
    return out


def load_cormorant_gap_train_ev(cormorant_dir: Path) -> np.ndarray:
    """Read `gap` targets from Cormorant train.npz and convert to eV.

    Cormorant stores raw Hartree values; EGNN's dataset.py multiplies by
    27.2114 before passing them to `compute_mean_mad`. We replicate that
    exact path here so the constants we write are in the same unit space
    EGNN trains in.
    """
    p = cormorant_dir / "train.npz"
    if not p.exists():
        fail(f"missing {p}")
    npz = np.load(p)
    if TARGET not in npz.files:
        fail(f"{p} lacks '{TARGET}' field; keys={list(npz.files)}")
    return np.asarray(npz[TARGET], dtype=np.float64) * HARTREE_TO_EV


def compute_meann_mad(values: np.ndarray) -> tuple[float, float]:
    """EGNN's compute_mean_mad — Mean Absolute Deviation, NOT std."""
    meann = float(np.mean(values))
    mad = float(np.mean(np.abs(values - meann)))
    return meann, mad


def audit_pyg_bijection(cormorant_idxs: dict) -> dict:
    """Load PyG QM9, check Data.idx ↔ Cormorant index bijection.

    Returns a small summary dict.
    """
    try:
        from torch_geometric.datasets import QM9
    except Exception as e:
        fail(f"could not import torch_geometric.datasets.QM9: {e}")

    qm9 = QM9(root=str(PYG_QM9_ROOT))
    n_pyg = len(qm9)
    expected_total = sum(EXPECTED.values())
    if n_pyg != expected_total:
        fail(f"PyG QM9 has {n_pyg} mols, expected {expected_total}")

    # Build the set of PyG idxs (0-indexed gdb_idx, post-filter).
    pyg_idxs = set()
    for i in range(n_pyg):
        pyg_idxs.add(int(qm9[i].idx.item()))

    cormorant_all = set()
    for split, ids in cormorant_idxs.items():
        cormorant_all.update(int(x) for x in ids)

    missing_in_pyg = cormorant_all - pyg_idxs
    missing_in_cormorant = pyg_idxs - cormorant_all
    if missing_in_pyg:
        fail(f"{len(missing_in_pyg)} Cormorant mols missing in PyG "
             f"(sample: {sorted(missing_in_pyg)[:5]})")
    if missing_in_cormorant:
        fail(f"{len(missing_in_cormorant)} PyG mols missing in Cormorant "
             f"(sample: {sorted(missing_in_cormorant)[:5]})")

    return {
        "pyg_total": n_pyg,
        "cormorant_total": len(cormorant_all),
        "bijection_verified": True,
    }


def audit_egnn_runs(target: str = TARGET) -> dict:
    """Check existing EGNN gap baseline runs match the expected config."""
    seeds = [42, 43, 44]
    required = {"nf": 128, "n_layers": 7, "attention": 1, "epochs": 300}
    summary = {}
    for s in seeds:
        mp = EGNN_RUNS_DIR / target / f"seed{s}" / "metrics.json"
        if not mp.exists():
            fail(f"missing existing EGNN run: {mp}")
        try:
            d = json.loads(mp.read_text())
        except Exception as e:
            fail(f"could not parse {mp}: {e}")
        cfg = d.get("config", {})
        for k, want in required.items():
            got = cfg.get(k)
            if got != want:
                fail(f"{mp} config.{k}={got!r}, expected {want!r}")
        splits = d.get("splits") or [{}]
        test_mae = splits[0].get("test_mae")
        if test_mae is None:
            fail(f"{mp} missing splits[0].test_mae")
        summary[f"seed{s}"] = {
            "test_mae": float(test_mae),
            "epochs_run": int(splits[0].get("epochs_run", 0)),
            "config": {k: cfg.get(k) for k in required},
        }
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[audit] loading Cormorant split indices ...")
    cormorant_idxs = load_cormorant_indices()
    for s, ids in cormorant_idxs.items():
        print(f"  {s}: {len(ids)} mols (Cormorant 1-indexed → 0-indexed)")

    print("[audit] verifying PyG ↔ Cormorant bijection ...")
    bijection = audit_pyg_bijection(cormorant_idxs)
    print(f"  PyG total = {bijection['pyg_total']}, "
          f"Cormorant total = {bijection['cormorant_total']}")

    print("[audit] computing meann/MAD for gap (Cormorant train, eV) ...")
    gap_train_ev = load_cormorant_gap_train_ev(CORMORANT_DIR)
    meann, mad = compute_meann_mad(gap_train_ev)
    print(f"  meann = {meann:.6f} eV")
    print(f"  MAD   = {mad:.6f} eV")

    print("[audit] verifying existing EGNN gap baseline runs ...")
    egnn_summary = audit_egnn_runs()
    for k, v in egnn_summary.items():
        print(f"  {k}: test_mae={v['test_mae']:.4f} eV "
              f"(epochs_run={v['epochs_run']})")

    out = {
        "schema_version": 1,
        "target": TARGET,
        "units": "eV (Cormorant Hartree × 27.2114)",
        "split_counts": EXPECTED,
        "split_hash_pyg_indices": {
            split: sha256_of_int_list(ids)
            for split, ids in cormorant_idxs.items()
        },
        "pyg_cormorant": {
            "pyg_to_cormorant_offset": 1,
            **bijection,
        },
        "normalization": {
            TARGET: {
                "meann": meann,
                "MAD": mad,
                "n_train": int(gap_train_ev.size),
                "formula": "meann = train.mean(); MAD = mean(|train - meann|)",
            }
        },
        "egnn_baseline_runs": egnn_summary,
    }
    out_json = OUT_DIR / "preprocessing_audit.json"
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[audit] wrote {out_json}")

    md_lines = [
        "# Preprocessing audit — gap apples-to-apples comparison",
        "",
        f"- Target: **{TARGET}** (eV; Cormorant Hartree × {HARTREE_TO_EV})",
        f"- Split counts (Cormorant fixed): "
        f"train={EXPECTED['train']}, valid={EXPECTED['valid']}, "
        f"test={EXPECTED['test']}",
        f"- PyG ↔ Cormorant bijection: verified "
        f"({bijection['pyg_total']} mols on each side)",
        f"- Index mapping: `pyg_data.idx == cormorant_npz['index'] - 1`",
        "",
        "## EGNN-style normalization constants (single source of truth)",
        "",
        f"- meann = {meann:.6f} eV",
        f"- MAD   = {mad:.6f} eV",
        f"- N_train = {gap_train_ev.size}",
        "- Formula: `meann = train.mean(); MAD = mean(|train - meann|)`",
        "  (Mean Absolute Deviation, NOT std — matches "
        "`external/egnn/qm9/utils.py:compute_mean_mad`.)",
        "",
        "## EGNN baseline reuse (gap)",
        "",
        "| Seed | test_mae (eV) | epochs_run |",
        "|---|---|---|",
    ]
    for k, v in egnn_summary.items():
        md_lines.append(f"| {k} | {v['test_mae']:.4f} | {v['epochs_run']} |")
    md_lines.append("")
    md_lines.append("All cells matched required config "
                    "(nf=128, n_layers=7, attention=1, epochs=300).")
    md_lines.append("")
    out_md = OUT_DIR / "preprocessing_audit.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[audit] wrote {out_md}")
    print("\n[audit] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
