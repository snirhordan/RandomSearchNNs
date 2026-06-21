#!/usr/bin/env python3
"""Aggregate the canonical-walk + geometric-bias ladder (QM9 gap).

Reads runs/qm9_geom/<cell>/seed<S>/gap/metrics.json and prints mean +/- std
test MAE per cell (eV), with the delta vs base_random in meV. Pre-registered
null band: gap seed noise is ~+/-0.9 meV, so |delta| < 2 meV is a null.
"""
import json
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
CELLS = ["base_random", "canonical_only", "canonical_bias", "random_bias"]
LABELS = {
    "base_random": "random multi-walk (m=8), no bias  [baseline]",
    "canonical_only": "canonical single walk, no bias",
    "canonical_bias": "canonical walk + geometric attn bias",
    "random_bias": "random multi-walk (m=8) + geometric attn bias",
}


def maes(cell):
    out = []
    for s in SEEDS:
        m = OUT_ROOT / cell / f"seed{s}" / "gap" / "metrics.json"
        if not m.exists():
            continue
        try:
            d = json.load(open(m))
        except Exception:
            continue
        v = d.get("summary", {}).get("mean_test_mae")
        if v is not None:
            out.append(float(v))
    return out


def mean_std(xs):
    if not xs:
        return None, None
    mu = sum(xs) / len(xs)
    if len(xs) < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, var ** 0.5


def main():
    base = maes("base_random")
    base_mu, _ = mean_std(base)
    print(f"{'cell':<40} {'n':>2} {'test MAE (eV)':>16} {'d vs base (meV)':>16}")
    print("-" * 78)
    for cell in CELLS:
        xs = maes(cell)
        mu, sd = mean_std(xs)
        if mu is None:
            print(f"{LABELS[cell]:<40} {0:>2} {'(pending)':>16}")
            continue
        delta = "" if base_mu is None else f"{(mu - base_mu) * 1000:+.1f}"
        print(f"{LABELS[cell]:<40} {len(xs):>2} "
              f"{mu:.4f} +/- {sd:.4f}  {delta:>14}")
    print("\nEGNN reference gap MAE: 0.048 eV. Null band: |delta| < 2 meV.")


if __name__ == "__main__":
    main()
