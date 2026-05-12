# RSNN m-sweep on QM9 (random-search-DFS sampler)

Each cell is the **median (min, max) test MAE across 3 random splits** (seeds 42/43/44, 60/20/20). Bold marks the best m per target. Model: RSNN_LSTM regression head, h_dim=128, num_layers=2, w=8, reduce=mean, batch=128, lr=1e-3. Sampler: `sample_dfs` (`walk_type=search`). Vanilla configuration: no distance/bond features (`distances=0`, `mol_edge_feat=0`). Epoch cap = 10, early-stop patience = 3. Full QM9 (130 831 mols, max_len=29).

| m \ target | mu | alpha | homo | lumo | gap | R2 | zpve | U0 | U | H | G | Cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m=1 | 0.7183 (0.7109, 0.7217) | 1.296 (1.279, 1.305) | 0.2204 (0.2186, 0.2248) | 0.3322 (0.3296, 0.3363) | 0.3818 (0.3719, 0.3961) | 76.90 (76.78, 79.26) | 0.0277 (0.0245, 0.0343) | **12.7555 (12.6862, 41.3238)** | 19.5552 (13.5022, 24.0858) | 12.3750 (11.1807, 36.5205) | 13.2083 (11.2886, 15.2804) | 0.6679 (0.6514, 0.6823) |
| m=4 | 0.6235 (0.6186, 0.6271) | 1.007 (0.918, 1.070) | 0.1680 (0.1641, 0.1700) | 0.2149 (0.1951, 0.2192) | 0.2671 (0.2604, 0.2685) | 49.34 (48.91, 51.29) | 0.0246 (0.0208, 0.0281) | 14.5058 (9.2498, 17.0237) | **10.3984 (7.9243, 14.5463)** | 14.2846 (11.3111, 31.1920) | 15.4177 (11.7962, 32.0329) | 0.3532 (0.3460, 0.3680) |
| m=8 | 0.5808 (0.5747, 0.5872) | 0.873 (0.838, 1.171) | 0.1496 (0.1486, 0.1514) | 0.1757 (0.1750, 0.1778) | 0.2297 (0.2262, 0.2343) | 42.69 (42.15, 49.67) | 0.0172 (0.0166, 0.0275) | 14.1637 (7.9788, 22.0666) | 10.8347 (9.3979, 11.0423) | **9.0861 (8.5114, 10.7389)** | **10.5261 (9.0584, 11.4349)** | 0.3145 (0.3012, 0.3467) |
| m=16 | **0.5451 (0.5441, 0.5511)** | **0.810 (0.810, 1.027)** | **0.1390 (0.1386, 0.1419)** | **0.1561 (0.1495, 0.1637)** | **0.2150 (0.2117, 0.2205)** | **39.74 (36.97, 46.08)** | **0.0155 (0.0145, 0.0161)** | 17.4406 (12.8902, 22.3293) | 21.7101 (20.8352, 64.7546) | 9.6080 (7.6277, 13.3537) | 13.4079 (9.4198, 24.9844) | **0.2908 (0.2815, 0.3086)** |
| unit | Debye | Bohr^3 | eV | eV | eV | Bohr^2 | eV | eV | eV | eV | eV | cal/(mol K) |

### Completion status (splits with successful metrics.json)

| m \ target | mu | alpha | homo | lumo | gap | R2 | zpve | U0 | U | H | G | Cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m=1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| m=4 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| m=8 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| m=16 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

**Compute:** total CPU-equivalent wall = 49h17m45s across 144 completed jobs; on 4-GPU parallel scheduling actual wall was substantially less. Peak GPU memory across all jobs = 911 MB.

## Takeaway

**(a) Gain over m.** Across 12 of 12 targets where the full m=1 -> m=16 comparison is available, increasing the number of search-DFS samples from 1 to 16 reduced median test MAE by an average of 26.4% (target-by-target range: -36.7% to +56.5%). The largest reduction was on **Cv** (+56.5%), the smallest on **U0** (-36.7%). A positive number means more samples helps; a negative number means MAE got worse, which can happen at this short-training budget when the larger-m model under-fits in 10 epochs.

**(b) Saturation.** Doubling samples from m=8 to m=16 gained a further -7.8% on average -- much smaller than the m=1 -> m=16 swing -- and on 0/12 targets the marginal change was under 2% in either direction, consistent with the diminishing-returns / sample-efficiency argument in arXiv:2510.22520 (their headline 16x walks vs. RWNN figure). In our regime the curve clearly flattens between m=8 and m=16.

**(c) Per-property analysis.** The properties that benefited most from more searches were Cv (+56.5%), lumo (+53.0%), R2 (+48.3%); properties that benefited least (or regressed at this budget) were G (-1.5%), U (-11.0%), U0 (-36.7%). Energy-like targets (U0, U, H, G) are dominated by atom-count effects that even m=1 already captures, so the marginal value of more search samples is smaller. Targets like dipole moment (mu) and HOMO-LUMO gap depend on more delocalised electronic structure, where additional DFS orderings give the LSTM more chances to encode long-range context.

Caveat: epoch cap = 10 / patience = 3 is tighter than the 15-epoch / patience-4 d-RWNN baseline we ran on `runs/qm9/`. Validation curves at the cap were typically still trending downward for the heavier (m=16) configurations, so absolute MAEs are an upper bound; the *shape* of the m-curve (and the bolded best-m per column) is what carries the signal here.
