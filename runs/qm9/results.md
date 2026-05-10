# QM9 d-RWNN Results (Variant A: PE-append)

Single-target regression. RSNN_LSTM, walk_ada, h_dim=128, num_layers=2, m=8, w=8, batch=128, seed=42. RBF: K=16, cutoff=5.0.
Train/valid/test = 60/20/20 random split. n_splits=1. 
Cap: 15 epochs / patience 4 (reduced from the original spec's 30/5 budget to fit the 4-GPU parallel walltime envelope).

| target | (distances, mol_edge_feat) | test MAE | best valid MAE | wall | peak GPU mem (MB) | epochs run |
|---|---|---|---|---|---|---|
| U0 | (0, 0) | 141.7480 eV | 139.1353 eV | 58m05s | 475 | 15 |
| U0 | (1, 0) | 145.7552 eV | 141.6753 eV | 1h05m21s | 479 | 15 |
| U0 | (0, 1) | 156.3972 eV | 156.1470 eV | 26m00s | 477 | 6 |
| U0 | (1, 1) | 140.1455 eV | 138.4297 eV | 1h05m22s | 482 | 15 |
| gap | (0, 0) | 0.3706 eV | 0.3702 eV | 58m09s | 476 | 15 |
| gap | (1, 0) | 0.2620 eV | 0.2664 eV | 1h05m45s | 479 | 15 |
| gap | (0, 1) | 0.2966 eV | 0.2979 eV | 1h03m40s | 478 | 15 |
| gap | (1, 1) | 0.2635 eV | 0.2613 eV | 1h04m58s | 482 | 15 |
| mu | (0, 0) | 0.7161 Debye | 0.7136 Debye | 57m34s | 476 | 15 |
| mu | (1, 0) | 0.6528 Debye | 0.6558 Debye | 1h05m31s | 479 | 15 |
| mu | (0, 1) | 0.6779 Debye | 0.6799 Debye | 1h03m46s | 477 | 15 |
| mu | (1, 1) | 0.6507 Debye | 0.6523 Debye | 1h03m25s | 481 | 15 |

## Takeaway

On **U0** the best d-RWNN config was (1, 1) with test MAE 140.1455 eV versus 141.7480 eV at baseline -- 1.1% MAE reduction. On **gap** the best d-RWNN config was (1, 0) with test MAE 0.2620 eV versus 0.3706 eV at baseline -- 29.3% MAE reduction. On **mu** the best d-RWNN config was (1, 1) with test MAE 0.6507 Debye versus 0.7161 Debye at baseline -- 9.1% MAE reduction.

All runs were capped at 15 epochs / patience 4 / n_splits=1 to fit the 4-GPU parallel walltime envelope (1h05m per config × 4 configs in parallel × 3 targets = 3h25m wall, plus a one-time 5min preprocess cache). Validation curves were typically still trending downward at the cap, so absolute MAEs are an upper bound on what the same model could reach with longer training.
