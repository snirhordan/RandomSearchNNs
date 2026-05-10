"""Shared pytest fixtures for the RWNN test-suite."""
import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import regex as re
import torch
import torch_geometric
from torch_geometric.data import Data

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# --- make the repo importable -----------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.utils import (
    get_canonical_molecule,
    mol2graph,
    get_neighbor_dict,
)


# --- regex tokenizer used in the quickstart ---------------------------------
PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|"
    r"\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def _process_tok(components):
    out = []
    for component in components:
        if component.startswith('[') and ':' in component:
            out.append(component.split(':')[0][1:])
    return out


def _build_vocab(smiles_list, tokenizer):
    vocab = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = get_canonical_molecule(mol)
        canon_smi = Chem.MolToSmiles(
            mol, doRandom=False, canonical=True,
            allHsExplicit=False, ignoreAtomMapNumbers=False,
        )
        toks = _process_tok(tokenizer.findall(canon_smi))
        vocab += toks
    vocab = list(np.unique(vocab))
    vocab.append('PAD')
    return {k: i for i, k in enumerate(vocab)}


# --- fixtures ---------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rng_seed():
    """Re-seed every random source per test."""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def regex_tokenizer():
    return re.compile(PATTERN)


@pytest.fixture
def tiny_smiles():
    """Pentane (5-carbon path) - simple non-trivial molecule."""
    return "CCCCC"


@pytest.fixture
def tiny_vocab(regex_tokenizer, tiny_smiles):
    """Build a vocab over a handful of smiles so tiny_graph has x_emb."""
    return _build_vocab([tiny_smiles, "CCO", "c1ccccc1", "CC(=O)O"], regex_tokenizer)


@pytest.fixture
def tiny_graph(tiny_smiles, regex_tokenizer, tiny_vocab):
    """A 5-node path graph (pentane) as a PyG Data with x_emb populated."""
    mol = Chem.MolFromSmiles(tiny_smiles)
    data = mol2graph(mol, tokenizer=regex_tokenizer, vocab=tiny_vocab)
    return data


@pytest.fixture(scope="session")
def clintox_path():
    return REPO_ROOT / 'quickstart' / 'data' / 'clintox.csv'


@pytest.fixture
def clintox_subset(clintox_path, regex_tokenizer):
    """Load 5 valid SMILES from clintox and convert with mol2graph.

    Returns a tuple ``(graphs, ys, vocab)``.
    """
    df = pd.read_csv(clintox_path, sep=',')
    arr = df.to_numpy()

    # we need a vocab over the chosen smiles
    chosen_smiles = []
    chosen_ys = []
    for row in arr:
        smi = row[0]
        y = row[2]
        mol = Chem.MolFromSmiles(smi)
        if mol is None or pd.isna(y):
            continue
        chosen_smiles.append(smi)
        chosen_ys.append(float(y))
        if len(chosen_smiles) >= 5:
            break

    vocab = _build_vocab(chosen_smiles, regex_tokenizer)

    graphs, ys = [], []
    for smi, y in zip(chosen_smiles, chosen_ys):
        mol = Chem.MolFromSmiles(smi)
        g = mol2graph(mol, tokenizer=regex_tokenizer, vocab=vocab)
        graphs.append(g)
        ys.append(torch.tensor(y, dtype=torch.float32))
    return graphs, ys, vocab


@pytest.fixture
def random_graph_50():
    """A random connected graph with 50 nodes for sampling distribution tests."""
    import networkx as nx
    g = nx.connected_watts_strogatz_graph(50, 4, 0.3, seed=123)
    edges = list(g.edges())
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.zeros((50, 1), dtype=torch.float)
    # x_emb is needed by the search functions; vocab has a 'PAD' entry.
    x_emb = torch.zeros((50,), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, x_emb=x_emb)
    data = get_neighbor_dict(data)
    return data
