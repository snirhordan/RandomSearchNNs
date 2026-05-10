"""
Empirical test: verify that the anchor (starting vertex) of every random walk /
random search sampler in utils/search.py is drawn uniformly from V.

We build a path graph on N=50 nodes (irregular degrees: degree 1 at endpoints,
degree 2 inside) and call each sampler N=100,000 times. We then run a chi-squared
goodness-of-fit test against the uniform distribution over the 50 nodes.

Expected: p-value > 0.05 for every sampler (cannot reject uniform).
"""
import os
import sys
import torch
import numpy as np
from scipy import stats
from torch_geometric.data import Data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.search import (
    sample_walks,
    sample_walks_mdlr,
    sample_walks_rum,
)


def make_path_graph(n=50, vocab_size=8):
    # Path: 0-1-2-...-(n-1). Endpoints have degree 1, interior have degree 2.
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.zeros((n, 1), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = n
    # x_emb is the per-node embedding id used by the samplers.
    data.x_emb = torch.randint(0, vocab_size, (n,), dtype=torch.long)
    return data


def chi2_uniform(counts):
    n = len(counts)
    total = counts.sum()
    expected = np.full(n, total / n)
    chi2, p = stats.chisquare(counts, f_exp=expected)
    return chi2, p


def collect_starts(sampler_fn, data, n_calls, **kwargs):
    """Each call samples nw=1 walk and we record the anchor (walk_ids[0,0,0])."""
    starts = np.empty(n_calls, dtype=np.int64)
    for i in range(n_calls):
        # fresh data clone so we don't accumulate side effects
        d = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
        )
        d.num_nodes = data.num_nodes
        d.x_emb = data.x_emb.clone()
        sampler_fn(d, nw=1, **kwargs)
        # walk_ids has shape (1, nw, l) => (1, 1, l); position 0 is the anchor
        starts[i] = int(d.walk_ids[0, 0, 0].item())
    return starts


def run_test(name, sampler_fn, data, n_calls, **kwargs):
    starts = collect_starts(sampler_fn, data, n_calls, **kwargs)
    counts = np.bincount(starts, minlength=data.num_nodes)
    chi2, p = chi2_uniform(counts)
    print(f"[{name}]  n_calls={n_calls}  chi2={chi2:.3f}  dof={data.num_nodes-1}  p={p:.4f}  "
          f"min_count={counts.min()}  max_count={counts.max()}  "
          f"verdict={'UNIFORM (cannot reject)' if p > 0.05 else 'NON-UNIFORM (reject)'}")
    return chi2, p, counts


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    import random as pyrandom
    pyrandom.seed(0)

    N = 50
    N_CALLS = 100_000
    data = make_path_graph(n=N)

    print(f"Path graph N={N}  walks per sampler={N_CALLS}")
    print("-" * 78)

    # vanilla random walk
    run_test("sample_walks (vanilla, uniform anchor + uniform-neighbor walk)",
             sample_walks, data, N_CALLS, l=4, s=2, non_backtracking=False)

    # vanilla with non-backtracking transitions
    run_test("sample_walks (non-backtracking transitions)",
             sample_walks, data, N_CALLS, l=4, s=2, non_backtracking=True)

    # MDLR (degree-weighted next-step transition)
    run_test("sample_walks_mdlr (MDLR transitions)",
             sample_walks_mdlr, data, N_CALLS, l=4, s=2, non_backtracking=False)

    # RUM
    run_test("sample_walks_rum",
             sample_walks_rum, data, N_CALLS, l=4, s=2, non_backtracking=False)


if __name__ == "__main__":
    main()
