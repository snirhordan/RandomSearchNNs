# Preprocessing audit — gap apples-to-apples comparison

- Target: **gap** (eV; Cormorant Hartree × 27.2114)
- Split counts (Cormorant fixed): train=100000, valid=17748, test=13083
- PyG ↔ Cormorant bijection: verified (130831 mols on each side)
- Index mapping: `pyg_data.idx == cormorant_npz['index'] - 1`

## EGNN-style normalization constants (single source of truth)

- meann = 6.857936 eV
- MAD   = 1.075730 eV
- N_train = 100000
- Formula: `meann = train.mean(); MAD = mean(|train - meann|)`
  (Mean Absolute Deviation, NOT std — matches `external/egnn/qm9/utils.py:compute_mean_mad`.)

## EGNN baseline reuse (gap)

| Seed | test_mae (eV) | epochs_run |
|---|---|---|
| seed42 | 0.0515 | 93 |
| seed43 | 0.0518 | 95 |
| seed44 | 0.0480 | 295 |

All cells matched required config (nf=128, n_layers=7, attention=1, epochs=300).

