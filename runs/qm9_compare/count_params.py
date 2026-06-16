"""Empirical parameter counter for RSNN_LSTM_Reg.

Sweeps a grid of (h_dim, num_layers) values, instantiates the model with
standard QM9 arg values, and prints a table sorted by distance from the
~743k target.  Combos within +/-5% (706k-780k) are flagged.

Run from repo root with::

    PYTHONPATH=/home/snirhordan/ito/RandomSearchNNs \
        /home/snirhordan/miniconda3/envs/rwnn/bin/python3 \
        runs/qm9_compare/count_params.py
"""
from __future__ import annotations

import itertools

import torch  # noqa: F401  -- needed so nn.Module construction works

from quickstart.train_qm9 import RSNN_LSTM_Reg


TARGET = 743_000
LOW = 706_000
HIGH = 780_000

# Standard arg values used by quickstart/train_qm9.py:main (line ~588).
PE_IN_DIM = 8
PE_OUT_DIM = 16
OUT_DIM = 1
N_EMB = 6
REDUCE = "sum"

H_DIMS = [64, 70, 72, 74, 80, 86, 88, 90, 96, 100, 102, 104, 110, 124, 128,
          140, 152, 160, 176, 184, 188, 190, 194, 215, 256, 304]
NUM_LAYERS = [1, 2, 3, 4, 6]


def count_params(h_dim: int, num_layers: int) -> int:
    model = RSNN_LSTM_Reg(
        pe_in_dim=PE_IN_DIM,
        pe_out_dim=PE_OUT_DIM,
        hid_dim=h_dim,
        out_dim=OUT_DIM,
        num_layers=num_layers,
        n_emb=N_EMB,
        reduce=REDUCE,
    )
    return sum(p.numel() for p in model.parameters())


def component_breakdown(h_dim: int, num_layers: int) -> dict[str, int]:
    """Return per-component param counts (LSTM block vs embed+readout+pe)."""
    model = RSNN_LSTM_Reg(
        pe_in_dim=PE_IN_DIM,
        pe_out_dim=PE_OUT_DIM,
        hid_dim=h_dim,
        out_dim=OUT_DIM,
        num_layers=num_layers,
        n_emb=N_EMB,
        reduce=REDUCE,
    )
    lstm = sum(p.numel() for p in model.rnn_layers.parameters())
    readout = sum(p.numel() for p in model.readout.parameters())
    pe = sum(p.numel() for p in model.pe_encoding.parameters())
    emb = sum(p.numel() for p in model.embedding.parameters())
    total = lstm + readout + pe + emb
    return {
        "lstm": lstm,
        "readout": readout,
        "pe_encoding": pe,
        "embedding": emb,
        "total": total,
    }


def main() -> None:
    rows = []
    for h_dim, n_layers in itertools.product(H_DIMS, NUM_LAYERS):
        n_params = count_params(h_dim, n_layers)
        rows.append((h_dim, n_layers, n_params))
    rows.sort(key=lambda r: abs(r[2] - TARGET))

    print(f"Target: {TARGET:,}  band: [{LOW:,}, {HIGH:,}]")
    print()
    header = f"| {'h_dim':>5} | {'L':>2} | {'params':>10} | {'delta':>10} | "\
             f"{'%off':>7} | in band |"
    sep = "|" + "-" * 7 + "|" + "-" * 4 + "|" + "-" * 12 + "|" + "-" * 12 + \
        "|" + "-" * 9 + "|" + "-" * 9 + "|"
    print(header)
    print(sep)
    for h_dim, n_layers, n_params in rows:
        delta = n_params - TARGET
        pct = 100.0 * delta / TARGET
        in_band = "YES" if LOW <= n_params <= HIGH else ""
        print(f"| {h_dim:>5} | {n_layers:>2} | {n_params:>10,} | "
              f"{delta:>+10,} | {pct:>+6.2f}% | {in_band:>7} |")

    print()
    print("Per-component breakdown (baseline h_dim=128, L=2):")
    br = component_breakdown(128, 2)
    for name, val in br.items():
        print(f"  {name:>12}: {val:>10,}")


if __name__ == "__main__":
    main()
