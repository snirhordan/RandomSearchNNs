# qm9_multitarget results

Configs (verified from per-cell metrics.json):
- RSNN: walk_type=search, m=8, w=16 (encoding window s), h=128, L=2, AdamW, lr=7.5e-4, wd=1e-4, 300ep/patience=50, EGNN-norm meann/MAD + L1 loss, Cormorant fixed split. Winner of A/B is O_B1_densedist (DFS-jump dense-distance fix in sample_dfs). gap seed=42 reused from A/B.
- RWNN: walk_type=walk_ada, m=4, w=8 (encoding window s), h=128, L=2, AdamW, lr=7.5e-4, wd=1e-4, 300ep/patience=50. Mid-sweep switched from m=16 -> m=4 for speed; all 36 RWNN cells reran with m=4.
- Walk length is per-molecule n (atom count), padded to max_len=29 for batching; `--w` is the encoding window, not the walk length.

Notes:
- EGNN reference for gap is the paper-published 0.048 eV; our internal rerun gave 0.0504 eV (~5% higher, within seed variance). Ratios use the paper value.
- R2 EGNN reference not cited per-target (atomization-energy task; <R^2> spatial extent in Bohr^2) -- shown as `-` / `?`.
- Energy targets (U0/U/H/G) RWNN ratios are ~17000-19000x -- the walk-pool readout is bounded budget so cannot in principle aggregate size-extensive quantities; structural ceiling discussed in `~/vault/reflections/ito/2026-05-22-o-series-ceiling.md`.

| Model | Target | mean ± std (n) | EGNN | Ratio |
|-------|--------|----------------|------|-------|
| rsnn | mu | 0.2581 ± 0.0026 (n=3) | 0.0290 | 8.90x |
| rsnn | alpha | 0.2055 ± 0.0010 (n=3) | 0.0710 | 2.89x |
| rsnn | homo | 0.0568 ± 0.0004 (n=3) | 0.0290 | 1.96x |
| rsnn | lumo | 0.0583 ± 0.0004 (n=3) | 0.0250 | 2.33x |
| rsnn | gap | 0.0870 ± 0.0011 (n=3) | 0.0480 | 1.81x |
| rsnn | R2 | 12.2239 ± 0.3074 (n=3) | - | ? |
| rsnn | zpve | 0.0038 ± 0.0000 (n=3) | 0.0015 | 2.42x |
| rsnn | U0 | 0.3391 ± 0.0148 (n=3) | 0.0110 | 30.83x |
| rsnn | U | 0.3195 ± 0.0168 (n=3) | 0.0120 | 26.62x |
| rsnn | H | 0.3305 ± 0.0063 (n=3) | 0.0120 | 27.54x |
| rsnn | G | 0.3288 ± 0.0088 (n=3) | 0.0120 | 27.40x |
| rsnn | Cv | 0.0601 ± 0.0001 (n=3) | 0.0310 | 1.94x |
| rwnn | mu | 0.5908 ± 0.0059 (n=3) | 0.0290 | 20.37x |
| rwnn | alpha | 0.9702 ± 0.0189 (n=3) | 0.0710 | 13.67x |
| rwnn | homo | 0.1382 ± 0.0014 (n=3) | 0.0290 | 4.77x |
| rwnn | lumo | 0.2112 ± 0.0006 (n=3) | 0.0250 | 8.45x |
| rwnn | gap | 0.2359 ± 0.0012 (n=3) | 0.0480 | 4.92x |
| rwnn | R2 | 59.3263 ± 0.2840 (n=3) | - | ? |
| rwnn | zpve | 0.0249 ± 0.0003 (n=3) | 0.0015 | 16.07x |
| rwnn | U0 | 209.8587 ± 0.7152 (n=3) | 0.0110 | 19078.07x |
| rwnn | U | 210.6453 ± 4.9984 (n=3) | 0.0120 | 17553.77x |
| rwnn | H | 206.9624 ± 9.1296 (n=3) | 0.0120 | 17246.87x |
| rwnn | G | 206.2075 ± 2.3773 (n=3) | 0.0120 | 17183.96x |
| rwnn | Cv | 0.4729 ± 0.0114 (n=3) | 0.0310 | 15.26x |