# Upstream Diff Audit: `RSNN_LSTM_Reg` vs `RSNN_LSTM`

## Sources

| Side | Path | Lines |
| --- | --- | --- |
| Upstream | `/tmp/upstream_rsnn/models/rwnn.py` (https://github.com/MLD3/RandomSearchNNs/blob/main/models/rwnn.py) | `class RSNN_LSTM` @ L184–L277 |
| Local   | `RandomSearchNNs/quickstart/train_qm9.py` | `class RSNN_LSTM_Reg` @ L125–L211 |

Upstream defines no class literally named `RSNN_LSTM_Reg`; our class is a derivation
(rename + regression head) of upstream's `RSNN_LSTM`.

Verification command:

```bash
grep -rn "class RSNN_LSTM" /tmp/upstream_rsnn
# /tmp/upstream_rsnn/models/rwnn.py:184:class RSNN_LSTM(torch.nn.Module):
```

## Side-by-side diff

### `__init__` signature

| Upstream | Local | Class |
| --- | --- | --- |
| `def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce)` | `def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce)` | **ALIGNED** (identical signature) |

### `__init__` body

| # | Upstream | Local | Class |
| --- | --- | --- | --- |
| 1 | `self.rnn_layers = ModuleList()`; first `nn.LSTM(hid_dim+pe_out_dim, hid_dim, 1, batch_first=True, bidirectional=True)` | identical | **ALIGNED** |
| 2 | Loop `for nl in range(num_layers - 1)` appending `nn.LSTM(2*hid_dim, hid_dim, 1, batch_first=True, bidirectional=True)` | identical (variable named `_`) | **ALIGNED** |
| 3 | `self.readout = ModuleList([Linear(2*hid_dim, 2*hid_dim), Linear(2*hid_dim, out_dim)])` | identical | **ALIGNED** |
| 4 | `self.pe_encoding = Linear(pe_in_dim, pe_out_dim)` | identical | **ALIGNED** |
| 5 | `self.embedding = nn.Embedding(n_emb, hid_dim, n_emb-1)` (last index = pad) | identical | **ALIGNED** |
| 6 | `self.reduce = reduce`; `self.num_layers = num_layers` | identical | **ALIGNED** |
| 7 | No distance / `mol_edge_feat` / RBF parameters inside the class | No distance / `mol_edge_feat` / RBF parameters inside the class | **ALIGNED** (geometric channel plumbed in dataset/featurizer, not this module — consistent with the user's pre-approved architecture) |

### `forward` body

| Step | Upstream | Local | Class |
| --- | --- | --- | --- |
| Inputs | `batch.walk_emb`, `batch.walk_ids`, `batch.walk_pe`, `batch.lengths.cpu()` | identical | **ALIGNED** |
| `graph_ns` | `[max(walk_ids[i]) for i in range(B)]` | identical | **ALIGNED** |
| Trim padding | `walk_ids[:, :, :max(lengths)]` | identical | **ALIGNED** |
| `walk_ids_proc` building (offset by `sum(graph_ns[:i]) + i`) | as upstream | identical | **ALIGNED** |
| Flatten + mask (`walk_ids_flat`, `walk_ids_proc_flat_masked`) | as upstream | identical | **ALIGNED** |
| Init `x` | `cat([embedding(walk_emb), pe_encoding(encoding)], -1)` | identical | **ALIGNED** |
| Per-layer custom loop with `pack_padded_sequence` → `LSTM` → `pad_packed_sequence` | as upstream; `h` carried across layers from `l == 0` | identical | **ALIGNED** (confirms `num_layers` is a Python loop, NOT `nn.LSTM(..., num_layers=L)`) |
| Scatter aggregation `scatter(node_agg, walk_ids_proc_flat_masked, reduce='mean')` | as upstream | identical | **ALIGNED** |
| Write-back when `l != num_layers-1` (`x_flat[mask] = node_agg[proc_mask]`) | as upstream | identical | **ALIGNED** |
| Graph-level pool `scatter(node_agg, graph_ids, reduce=self.reduce)` *after* the layer loop | as upstream | identical | **ALIGNED** (confirms pool location matches upstream — pool over the post-final-layer `node_agg`) |
| Readout `relu(readout[0](x)); readout[1](x)` | as upstream | identical | **ALIGNED** |
| Final activation | `x = torch.sigmoid(x); return x` | `# NO sigmoid -- regression head.`; `return x` | **EXPECTED** — removal of sigmoid is the regression-head modification (the `_Reg` suffix); pre-approved for QM9 regression targets |

### Class name

| Upstream | Local | Class |
| --- | --- | --- |
| `RSNN_LSTM` | `RSNN_LSTM_Reg` | **EXPECTED** — rename signals the regression-head variant; no behavioural change |

### Items explicitly checked

- **Bidirectional LSTM**: both use `bidirectional=True` on all layers. ALIGNED.
- **`num_layers` as a custom Python loop**: both wrap a `for l in range(self.num_layers)` around single-layer LSTMs with manual hidden-state carry — not PyTorch's stacked-LSTM. ALIGNED.
- **Reduce-pool location**: both pool *after* the layer loop using `scatter(node_agg, graph_ids, reduce=self.reduce)`, operating on the post-final-layer `node_agg`. ALIGNED.

### Divergences

None. There is no entry in a **DIVERGENT** bucket.

## Verdict

**ALIGNED-WITH-NOTES** — local `RSNN_LSTM_Reg` is bit-equivalent to upstream
`RSNN_LSTM` except for two pre-approved changes: the class rename and the
removal of the final `torch.sigmoid` (regression head). All distance /
`mol_edge_feat` / RBF additions live outside this class, in the dataset /
featurizer / sampler layers — none of them leak into the model module itself,
which is what we expect.
