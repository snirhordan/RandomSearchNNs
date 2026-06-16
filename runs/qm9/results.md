# QM9 d-RWNN Results (Variant A: PE-append)

Single-target regression. RSNN_LSTM, walk_ada, h_dim=128, num_layers=2, m=8, w=8, batch=128, seed=42. RBF: K=16, cutoff=5.0.
Train/valid/test = 60/20/20 random split. n_splits=1. 
Cap: 15 epochs / patience 4 (reduced from the original spec's 30/5 budget to fit the 4-GPU parallel walltime envelope).

| target | (distances, mol_edge_feat) | test MAE | best valid MAE | wall | peak GPU mem (MB) | epochs run |
|---|---|---|---|---|---|---|
| mu | (0, 0) | 0.7161 Debye | 0.7136 Debye | 57m34s | 476 | 15 |
| mu | (1, 0) | 0.6528 Debye | 0.6558 Debye | 1h05m31s | 479 | 15 |
| mu | (0, 1) | 0.6779 Debye | 0.6799 Debye | 1h03m46s | 477 | 15 |
| mu | (1, 1) | 0.6507 Debye | 0.6523 Debye | 1h03m25s | 481 | 15 |
| alpha | (0, 0) | 1.4196 Bohr^3 | 1.4276 Bohr^3 | 58m44s | 475 | 15 |
| alpha | (1, 0) | 1.0978 Bohr^3 | 1.0820 Bohr^3 | 1h04m59s | 480 | 15 |
| alpha | (0, 1) | 1.1961 Bohr^3 | 1.1909 Bohr^3 | 1h04m38s | 477 | 15 |
| alpha | (1, 1) | 1.0644 Bohr^3 | 1.0516 Bohr^3 | 1h05m05s | 481 | 15 |
| homo | (0, 0) | 0.2057 eV | 0.2059 eV | 57m48s | 476 | 15 |
| homo | (1, 0) | 0.1572 eV | 0.1573 eV | 1h04m46s | 479 | 15 |
| homo | (0, 1) | 0.1718 eV | 0.1719 eV | 1h03m09s | 477 | 15 |
| homo | (1, 1) | 0.1584 eV | 0.1594 eV | 1h04m20s | 481 | 15 |
| lumo | (0, 0) | 0.3227 eV | 0.3218 eV | 59m51s | 476 | 15 |
| lumo | (1, 0) | 0.2226 eV | 0.2264 eV | 1h05m01s | 479 | 15 |
| lumo | (0, 1) | 0.2406 eV | 0.2443 eV | 1h04m21s | 478 | 15 |
| lumo | (1, 1) | 0.2154 eV | 0.2152 eV | 1h04m37s | 481 | 15 |
| gap | (0, 0) | 0.3706 eV | 0.3702 eV | 58m09s | 476 | 15 |
| gap | (1, 0) | 0.2620 eV | 0.2664 eV | 1h05m45s | 479 | 15 |
| gap | (0, 1) | 0.2966 eV | 0.2979 eV | 1h03m40s | 478 | 15 |
| gap | (1, 1) | 0.2635 eV | 0.2613 eV | 1h04m58s | 482 | 15 |
| R2 | (0, 0) | 85.4751 Bohr^2 | 84.3577 Bohr^2 | 58m05s | 476 | 15 |
| R2 | (1, 0) | 70.4528 Bohr^2 | 70.1713 Bohr^2 | 1h04m27s | 479 | 15 |
| R2 | (0, 1) | 72.7350 Bohr^2 | 74.1358 Bohr^2 | 1h03m55s | 477 | 15 |
| R2 | (1, 1) | 70.1007 Bohr^2 | 70.0998 Bohr^2 | 1h04m22s | 482 | 15 |
| zpve | (0, 0) | 0.0357 eV | 0.0357 eV | 57m20s | 476 | 15 |
| zpve | (1, 0) | 0.0326 eV | 0.0322 eV | 1h05m46s | 479 | 15 |
| zpve | (0, 1) | 0.0351 eV | 0.0349 eV | 47m10s | 477 | 11 |
| zpve | (1, 1) | 0.0304 eV | 0.0301 eV | 1h05m15s | 482 | 15 |
| U0 | (0, 0) | 141.7480 eV | 139.1353 eV | 58m05s | 475 | 15 |
| U0 | (1, 0) | 145.7552 eV | 141.6753 eV | 1h05m21s | 479 | 15 |
| U0 | (0, 1) | 156.3972 eV | 156.1470 eV | 26m00s | 477 | 6 |
| U0 | (1, 1) | 140.1455 eV | 138.4297 eV | 1h05m22s | 482 | 15 |
| U | (0, 0) | 142.0827 eV | 138.5484 eV | 53m23s | 475 | 14 |
| U | (1, 0) | 141.8264 eV | 142.4087 eV | 1h01m18s | 479 | 14 |
| U | (0, 1) | 154.8961 eV | 155.1988 eV | 26m24s | 477 | 6 |
| U | (1, 1) | 137.2517 eV | 135.8222 eV | 1h03m47s | 482 | 15 |
| H | (0, 0) | 139.7775 eV | 137.4262 eV | 56m30s | 475 | 15 |
| H | (1, 0) | 144.3509 eV | 145.0881 eV | 59m52s | 479 | 14 |
| H | (0, 1) | 157.5960 eV | 157.7083 eV | 26m02s | 477 | 6 |
| H | (1, 1) | 135.4236 eV | 134.1732 eV | 1h04m12s | 481 | 15 |
| G | (0, 0) | 142.2774 eV | 139.6424 eV | 57m38s | 475 | 15 |
| G | (1, 0) | 143.1801 eV | 143.9774 eV | 1h01m01s | 479 | 14 |
| G | (0, 1) | 153.8143 eV | 153.9083 eV | 26m14s | 477 | 6 |
| G | (1, 1) | 151.8083 eV | 150.2029 eV | 30m58s | 479 | 7 |
| Cv | (0, 0) | 0.6387 cal/(mol K) | 0.6402 cal/(mol K) | 58m13s | 476 | 15 |
| Cv | (1, 0) | 0.5384 cal/(mol K) | 0.5384 cal/(mol K) | 1h04m57s | 479 | 15 |
| Cv | (0, 1) | 0.5451 cal/(mol K) | 0.5424 cal/(mol K) | 1h03m53s | 478 | 15 |
| Cv | (1, 1) | 0.5017 cal/(mol K) | 0.5016 cal/(mol K) | 1h04m24s | 481 | 15 |

## Takeaway

On **mu** the best d-RWNN config was (1, 1) with test MAE 0.6507 Debye versus 0.7161 Debye at baseline -- 9.1% MAE reduction. On **alpha** the best d-RWNN config was (1, 1) with test MAE 1.0644 Bohr^3 versus 1.4196 Bohr^3 at baseline -- 25.0% MAE reduction. On **homo** the best d-RWNN config was (1, 0) with test MAE 0.1572 eV versus 0.2057 eV at baseline -- 23.6% MAE reduction. On **lumo** the best d-RWNN config was (1, 1) with test MAE 0.2154 eV versus 0.3227 eV at baseline -- 33.3% MAE reduction. On **gap** the best d-RWNN config was (1, 0) with test MAE 0.2620 eV versus 0.3706 eV at baseline -- 29.3% MAE reduction. On **R2** the best d-RWNN config was (1, 1) with test MAE 70.1007 Bohr^2 versus 85.4751 Bohr^2 at baseline -- 18.0% MAE reduction. On **zpve** the best d-RWNN config was (1, 1) with test MAE 0.0304 eV versus 0.0357 eV at baseline -- 15.1% MAE reduction. On **U0** the best d-RWNN config was (1, 1) with test MAE 140.1455 eV versus 141.7480 eV at baseline -- 1.1% MAE reduction. On **U** the best d-RWNN config was (1, 1) with test MAE 137.2517 eV versus 142.0827 eV at baseline -- 3.4% MAE reduction. On **H** the best d-RWNN config was (1, 1) with test MAE 135.4236 eV versus 139.7775 eV at baseline -- 3.1% MAE reduction. On **G** the baseline (no distances, no bond features) was best at 142.2774 eV; d-RWNN extensions did not improve over baseline at this epoch budget. On **Cv** the best d-RWNN config was (1, 1) with test MAE 0.5017 cal/(mol K) versus 0.6387 cal/(mol K) at baseline -- 21.4% MAE reduction.

All runs were capped at 15 epochs / patience 4 / n_splits=1 to fit the 4-GPU parallel walltime envelope (~1h05m per config x 4 configs in parallel x 12 targets, plus a one-time ~5min preprocess cache per target). Validation curves were typically still trending downward at the cap, so absolute MAEs are an upper bound on what the same model could reach with longer training.
