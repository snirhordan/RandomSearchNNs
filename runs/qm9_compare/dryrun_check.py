#!/usr/bin/env python3
"""5-check validator for the gap apples-to-apples dry-run.

Pass criteria (per the agreed grill-me spec, with #4/#5 removed):
  1. Process exit code 0 (caller responsibility — dry-run must return rc=0).
  2. Split sizes in train.log match Cormorant counts (100k/17748/13083).
  3. meann/MAD logged by train_qm9.py match preprocessing_audit.json within 1e-6.
  4. No NaN or Inf in any train_mse or valid_mae across all epoch_log entries.
  5. Walks generated without shape errors (smoke test — implied if no errors
     and metrics.json was written).

Usage:
  python3 dryrun_check.py \
      --metrics runs/qm9_compare/rsnn/seed42/metrics.json \
      --train_log runs/qm9_compare/rsnn/seed42/train.log \
      --audit_json runs/qm9_compare/preprocessing_audit.json

Exit 0 = green, 1 = red.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path

EXPECTED = {"train": 100000, "valid": 17748, "test": 13083}


def fail(msg: str) -> None:
    print(f"  RED: {msg}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--train_log", type=Path, required=True)
    p.add_argument("--audit_json", type=Path, required=True)
    p.add_argument("--rtol", type=float, default=1e-6)
    args = p.parse_args()

    if not args.metrics.exists():
        print(f"  RED: missing {args.metrics}")
        return 1
    if not args.train_log.exists():
        print(f"  RED: missing {args.train_log}")
        return 1
    if not args.audit_json.exists():
        print(f"  RED: missing {args.audit_json}")
        return 1

    audit = json.loads(args.audit_json.read_text())
    audit_meann = audit["normalization"]["gap"]["meann"]
    audit_mad = audit["normalization"]["gap"]["MAD"]

    metrics = json.loads(args.metrics.read_text())
    train_log = args.train_log.read_text()

    red = 0

    # Check 2: split sizes from train.log
    m = re.search(r"cormorant split: train=(\d+) valid=(\d+) test=(\d+)",
                  train_log)
    if not m:
        fail("could not find 'cormorant split: train=... valid=... test=...' "
             "line in train.log")
        red += 1
    else:
        t, v, tt = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if (t, v, tt) != (EXPECTED["train"], EXPECTED["valid"],
                          EXPECTED["test"]):
            fail(f"split sizes ({t},{v},{tt}) != expected "
                 f"({EXPECTED['train']},{EXPECTED['valid']},"
                 f"{EXPECTED['test']})")
            red += 1
        else:
            print(f"  OK split sizes: train={t} valid={v} test={tt}")

    # Check 3: meann/MAD logged match audit
    m = re.search(r"loaded EGNN normalization: meann=([\-\d.eE+]+) "
                  r"MAD=([\-\d.eE+]+)", train_log)
    if not m:
        fail("could not find 'loaded EGNN normalization: meann=... MAD=...' "
             "in train.log")
        red += 1
    else:
        run_meann = float(m.group(1))
        run_mad = float(m.group(2))
        if abs(run_meann - audit_meann) > args.rtol * max(abs(audit_meann), 1):
            fail(f"meann mismatch: run={run_meann:.6f} vs "
                 f"audit={audit_meann:.6f} (rtol={args.rtol})")
            red += 1
        elif abs(run_mad - audit_mad) > args.rtol * max(abs(audit_mad), 1):
            fail(f"MAD mismatch: run={run_mad:.6f} vs "
                 f"audit={audit_mad:.6f} (rtol={args.rtol})")
            red += 1
        else:
            print(f"  OK normalization: meann={run_meann:.6f} "
                  f"MAD={run_mad:.6f} matches audit")

    # Check 4: no NaN/Inf in any epoch's loss
    splits = metrics.get("splits") or []
    bad_count = 0
    for sp in splits:
        for ep in sp.get("epoch_log", []) or []:
            for key in ("train_mse", "valid_mae"):
                v = ep.get(key)
                if v is None:
                    continue
                if not math.isfinite(float(v)):
                    bad_count += 1
    if bad_count:
        fail(f"{bad_count} NaN/Inf entries in epoch_log losses")
        red += 1
    else:
        # Also confirm we ran SOMETHING.
        n_epochs = sum(len(sp.get("epoch_log") or []) for sp in splits)
        print(f"  OK loss numerics: {n_epochs} epoch_log entries, no NaN/Inf")

    # Check 5 (implicit): metrics.json has a test_mae for at least one split.
    has_test = any(sp.get("test_mae") is not None or
                   sp.get("final_test_mae") is not None
                   for sp in splits)
    if not has_test:
        fail("no splits[*].test_mae written — sampler/loop likely crashed")
        red += 1
    else:
        print("  OK walks produced metrics: splits[0].test_mae populated")

    if red == 0:
        print("\n[dryrun_check] GREEN — all checks passed")
        return 0
    print(f"\n[dryrun_check] RED — {red} check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
