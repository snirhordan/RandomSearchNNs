"""Tests for the --lr_schedule / --lr_min_frac options in quickstart/train_pdbbind.py.

Cosine annealing was added for compute-capped runs: with a constant learning rate a
short run stops mid-flight at the full lr and selects a noisy checkpoint. The default
stays 'constant' so all earlier runs remain reproducible.

Reuses the 6-complex synthetic cache from tests/test_train_pdbbind_smoke.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quickstart.train_pdbbind import main as train_main
from tests.test_train_pdbbind_smoke import _build_synthetic_cache


def _argv(data_dir, out_dir, extra):
    return [
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--base", "transformer",
        "--h_dim", "16", "--num_layers", "2", "--nhead", "2", "--ffn_mult", "2",
        "--pos_enc", "rope", "--attn_mode", "full", "--reduce", "sum",
        "--epochs", "2", "--batch_size", "2", "--num_workers", "0",
        "--patience", "50", "--seed", "0",
        "--m", "3", "--w", "4", "--max_len", "24", "--ligand_max_len", "8",
        "--angles", "1", "--dihedrals", "1", "--angle_K", "4", "--dihedral_K", "2",
        "--distances", "0", "--mol_edge_feat", "0", "--standardize", "1",
    ] + extra


def test_constant_is_the_default(tmp_path):
    """No --lr_schedule flag => 'constant', and the cosine banner is absent."""
    data_dir = _build_synthetic_cache(tmp_path)
    out_dir = tmp_path / "const"
    assert train_main(_argv(data_dir, out_dir, [])) == 0

    with open(out_dir / "metrics.json") as f:
        cfg = json.load(f)["config"]
    assert cfg["lr_schedule"] == "constant"

    log = (out_dir / "train.log").read_text()
    assert "lr_schedule=cosine" not in log


def test_cosine_runs_and_is_recorded(tmp_path):
    """--lr_schedule cosine trains to completion and is recorded in config + log."""
    data_dir = _build_synthetic_cache(tmp_path)
    out_dir = tmp_path / "cos"
    argv = _argv(data_dir, out_dir, ["--lr_schedule", "cosine", "--lr_min_frac", "0.05"])
    assert train_main(argv) == 0

    with open(out_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["config"]["lr_schedule"] == "cosine"
    assert metrics["config"]["lr_min_frac"] == 0.05

    log = (out_dir / "train.log").read_text()
    assert "lr_schedule=cosine" in log
    assert "total_steps=" in log

    for key in ("pearson_r", "spearman_rho", "rmse", "mae"):
        assert np.isfinite(metrics["test_metrics"][key])


def test_cosine_anneals_monotonically_to_eta_min():
    """Mirror the trainer's scheduler construction and check the lr trajectory.

    Guards the property the compute-capped runs depend on: over exactly the epoch
    budget the lr descends from --lr to --lr * --lr_min_frac without overshoot.
    """
    lr, lr_min_frac = 1e-3, 0.01
    epochs, batches_per_epoch = 5, 40
    total_steps = epochs * batches_per_epoch
    eta_min = lr * lr_min_frac

    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=eta_min)

    seen = [opt.param_groups[0]["lr"]]
    for _ in range(total_steps):
        opt.step()
        sched.step()
        seen.append(opt.param_groups[0]["lr"])

    assert seen[0] == lr
    assert all(b <= a + 1e-12 for a, b in zip(seen, seen[1:])), "lr must not increase"
    assert seen[-1] == max(eta_min, min(seen)), "final lr should be the floor"
    assert abs(seen[-1] - eta_min) < 1e-9
    # Halfway through a cosine schedule the lr sits near the midpoint of [eta_min, lr].
    assert abs(seen[total_steps // 2] - (eta_min + lr) / 2) < 0.1 * lr
