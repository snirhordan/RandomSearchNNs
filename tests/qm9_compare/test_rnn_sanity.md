# `test_rnn_sanity.py` — LSTM internals of `RSNN_LSTM_Reg`

Pins the default state of every LSTM-training concern that an upcoming round
of flags (`--grad_clip`, `--lstm_init=orthogonal`, `--dropout`, ...) will
flip. Construct a fresh model with `RSNN_LSTM_Reg(pe_in_dim=8, pe_out_dim=16,
hid_dim=64, out_dim=1, num_layers=2, n_emb=6, reduce='sum')` and a synthetic
adaptive-walk `Batch` over a 5-node path graph.

## Checks

**1. Bidirectional verification.** For each of the two `nn.LSTM` layers we
assert `bidirectional == True` and that the eight bidirectional parameter
tensors (`weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0` and
their `_reverse` counterparts) are all registered. This is the current
default and the post-flag behaviour must keep both directions allocated.

**2. Param count per component.** We sum `numel()` over `rnn_layers`,
`embedding`, `pe_encoding`, `readout` and assert the total equals
`sum(p.numel() for p in model.parameters())`. The breakdown is printed
(`rnn_layers=174080`, `embedding=384`, `pe_encoding=144`, `readout=16641`,
total `191249`) so any future submodule addition (e.g. dropout layers,
projection heads) is visible in the diff.

**3. Default `W_hh` is not orthogonal.** Each `weight_hh_l0` has shape
`(4*hidden_size, hidden_size)`; PyTorch's default init draws uniform in
`+- 1/sqrt(hidden_size)`. We compute `W W^T` and confirm it is *not* close
to identity (Frobenius distance ~15 in our run vs ~0 for an orthogonal
init). This guarantees `--lstm_init=orthogonal` will have observable effect.

**4. Default forget-gate bias is ~0.** PyTorch packs the LSTM bias as
`[i, f, g, o]` slices of length `hidden_size` inside `bias_ih_l0` and
`bias_hh_l0`. We extract the forget-gate slice `bias[hs:2*hs]` from both
biases, average their sum, and confirm `|mean| < 0.5` (we see ~0.02 in
practice). This documents that the classic Jozefowicz et al. (2015) trick
of seeding the forget bias to +1.0 is currently **not** applied, and a
proper `--lstm_init=orthogonal` flag should both orthogonalise `W_hh` *and*
set this slice to 1.0.

**5. Gradient flow check.** We run one forward + MSE backward on a synthetic
batch and record `weight_hh_l0.grad.norm()` per layer. Both norms must be
finite and lie strictly in `(1e-8, 100)` — flagging both vanishing
(norm == 0) and exploding (norm >> 1) regimes. A monotonic decrease across
layers is logged as a soft warning since at init it is expected to be
small but non-zero. Observed: layer-0 ≈ 0.20, layer-1 ≈ 0.42 (no
vanishing).

**6. `--grad_clip` would work.** We scale the loss by `1e3` to push the
total grad norm above 1.0, snapshot it (~192 in our run), call
`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`, and
assert (a) the returned value equals the pre-clip norm (PyTorch's
contract), (b) the post-clip total norm is `<= 1.0 + 1e-4`, and (c)
clipping strictly reduced the norm. This pins the contract a future
`--grad_clip` CLI flag will rely on.

## Run

```bash
cd /home/snirhordan/ito/RandomSearchNNs && \
  PYTHONPATH=$(pwd) /home/snirhordan/miniconda3/envs/rwnn/bin/python3 \
  -m pytest tests/qm9_compare/test_rnn_sanity.py -v -s
```

All 6 tests pass on the `rwnn` conda env (torch + torch-geometric + pytest).
