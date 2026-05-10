#!/usr/bin/env python
# coding: utf-8
# Smoke version of train_rwnn.ipynb -- runs a tiny training to confirm the env works.
# - Caps n_splits, epochs, dataset size
# - Adds device definition
# - Fixes the broken `from utils.utils import *` import (the repo only has utils/search.py;
#   helpers like get_ids / mol2graph / random_split come from generation.*)

import os
import sys
import time
import copy
import math
import random
import argparse
import numpy as np
import regex as re
import pandas as pd
import pickle as pkl
import networkx as nx
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# -- repo on sys.path so the various module groups resolve --
cwd_path = Path(__file__).resolve().parent
sys.path.append(str(cwd_path.parent))

from generation.utils import (
    get_canonical_molecule, mol2graph, get_ids,
)
from generation.scaffold_split import random_split
# walk samplers live here
from utils.search import (
    sample_walks, sample_walks_adaptive,
    sample_walks_mdlr, sample_walks_mdlr_adaptive,
    sample_walks_rum, sample_walks_rum_adaptive,
    sample_dfs, sample_bfs,
)

SEED = 2024
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device_idx = 0
device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
print(f"[smoke] device={device}", flush=True)

# ------------------------ Step 1: data prep ------------------------
def process_tok(components):
    modified_components = []
    for component in components:
        if component.startswith('[') and ':' in component:
            base_component = component.split(':')[0][1:]
            modified_components.append(base_component)
    return modified_components

idx_y = 2
idx_smiles = 0
data_path = str(cwd_path / 'data' / 'clintox.csv')

df = pd.read_csv(data_path, sep=',')
data = df.to_numpy()

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex_tokenizer = re.compile(PATTERN)

vocab = []
l_max = -1
for idx, smiles in enumerate(data[:, idx_smiles]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and not np.isnan(data[idx, idx_y]):
        if mol.GetNumAtoms() > l_max:
            l_max = mol.GetNumAtoms()
        mol = get_canonical_molecule(mol)
        smiles = Chem.MolToSmiles(mol, doRandom=False, canonical=True,
                                  allHsExplicit=False, ignoreAtomMapNumbers=False)
        smiles_tok = process_tok(regex_tokenizer.findall(smiles))
        vocab += smiles_tok

vocab_d = {}
vocab = list(np.unique(vocab))
vocab.append('PAD')
for i, key in enumerate(vocab):
    vocab_d[key] = i
vocab = vocab_d

ys_data, graphs = [], []
for idx, smiles in enumerate(data[:, idx_smiles]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and not np.isnan(data[idx, idx_y]):
        mol = get_canonical_molecule(mol)
        ys = np.array(data[idx, idx_y], dtype=np.float32)
        ys_data.append(torch.from_numpy(ys))
        graphs.append(mol2graph(mol, tokenizer=regex_tokenizer, vocab=vocab))

print(f"[smoke] loaded {len(graphs)} graphs, vocab size={len(vocab)}, l_max={l_max}", flush=True)

splits = random_split(len(graphs), test_size=0.2, val_size=0.2, random_state=0)

# ------------------------ Step 2: model ------------------------
class RWNN(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.RNN = nn.LSTM(hid_dim, hid_dim, num_layers, batch_first=True, bidirectional=True)
        self.readout = torch.nn.ModuleList()
        self.readout.append(torch.nn.Linear(hid_dim * 2, hid_dim * 2))
        self.readout.append(torch.nn.Linear(hid_dim * 2, out_dim))
        self.pe_encoding = torch.nn.Linear(pe_in_dim, pe_out_dim)
        self.embedding = nn.Embedding(n_emb, hid_dim, n_emb - 1)
        self.reduce = reduce
        self.num_layers = num_layers

    def forward(self, batch):
        walk_emb = batch.walk_emb
        walk_ids = batch.walk_ids
        lengths = batch.lengths.cpu()

        ids = torch.arange(walk_ids.shape[0])[:, None]
        ids = torch.broadcast_to(ids, (walk_ids.shape[0], walk_ids.shape[1]))
        ids = torch.flatten(ids, start_dim=0, end_dim=1).to(walk_ids.device)

        x = pack_padded_sequence(self.embedding(walk_emb), lengths,
                                 batch_first=True, enforce_sorted=False)
        x, (h, c) = self.RNN(x)
        x = torch.cat((h[-2], h[-1]), dim=1)
        x = scatter(x, ids, dim=0, reduce=self.reduce)
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)
        return x


class SMILESDataset(Dataset):
    def __init__(self, graphs, ys, m=8, l=25, w=8, nb=True, max_len=None, vocab=None, walk_type='walk'):
        self.graphs = graphs
        self.ys = ys
        self.m = m
        self.l = l
        self.w = w
        self.nb = nb
        self.max_len = max_len
        self.vocab = vocab
        self.walk_type = walk_type

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.ys = self.ys[idx]
        data.idx = idx
        if self.walk_type == 'walk_ada':
            data = sample_walks_adaptive(data, self.m, data.x.shape[0], self.w, self.nb, self.max_len, self.vocab)
        elif self.walk_type == 'walk':
            data = sample_walks(data, self.m, self.l, self.w, self.nb)
        return data


# ------------------------ Step 3: tiny training ------------------------
dropout = 0
h_dim = 64           # smaller than 128 for speed
num_layers = 2
m = 8
l = None
w = 8
nb = False
lr = 1e-3
batch_size = 64
n_splits = 1         # smoke: 1 split only
reduce = 'mean'
out_dim = 1
pe_out_dim = 16
SMOKE_EPOCHS = 2     # smoke: 2 epochs only
EARLY_STOPPING = 5

valid_aucs, test_aucs = [], []
for s in range(n_splits):
    pe_in_dim = 2 * w
    model = RWNN(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    walk_type = 'walk_ada'
    train_idx, valid_idx, test_idx = splits['train'][s], splits['valid'][s], splits['test'][s]

    # smoke: subsample training set to keep runtime low
    train_idx_sm = list(train_idx[:200])
    valid_idx_sm = list(valid_idx[:80])
    test_idx_sm = list(test_idx[:80])
    print(f"[smoke] split {s}: train={len(train_idx_sm)} valid={len(valid_idx_sm)} test={len(test_idx_sm)}", flush=True)

    train_data = SMILESDataset(get_ids(graphs, train_idx_sm), get_ids(ys_data, train_idx_sm), m, l, w, nb, l_max, vocab, walk_type)
    valid_data = SMILESDataset(get_ids(graphs, valid_idx_sm), get_ids(ys_data, valid_idx_sm), m, l, w, nb, l_max, vocab, walk_type)
    test_data = SMILESDataset(get_ids(graphs, test_idx_sm), get_ids(ys_data, test_idx_sm), m, l, w, nb, l_max, vocab, walk_type)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    best_loss = float('inf')
    best_state = None
    stop_counter = 0
    model.train()
    for epoch in range(SMOKE_EPOCHS):
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).squeeze()
            loss = criterion(out, batch.ys.squeeze())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        valid_losses = []
        for batch in valid_loader:
            batch = batch.to(device)
            out = model(batch).squeeze()
            valid_losses.append(criterion(out, batch.ys.squeeze()).detach().cpu().numpy())
        valid_loss = float(np.mean(valid_losses))
        train_loss = float(np.mean(train_losses))
        print(f"[smoke] split {s} epoch {epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} dt={time.time()-t0:.1f}s", flush=True)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= EARLY_STOPPING:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ys, outs = [], []
    for batch in valid_loader:
        batch = batch.to(device)
        outs.append(model(batch).detach().cpu().numpy())
        ys.append(batch.ys.detach().cpu().numpy())
    try:
        valid_auc = roc_auc_score(np.concatenate(ys, axis=0), np.concatenate(outs, axis=0))
    except ValueError:
        valid_auc = float('nan')
    valid_aucs.append(valid_auc)

    ys, outs = [], []
    for batch in test_loader:
        batch = batch.to(device)
        outs.append(model(batch).detach().cpu().numpy())
        ys.append(batch.ys.detach().cpu().numpy())
    try:
        test_auc = roc_auc_score(np.concatenate(ys, axis=0), np.concatenate(outs, axis=0))
    except ValueError:
        test_auc = float('nan')
    test_aucs.append(test_auc)

    print(f"[smoke] split {s} valid_auc={valid_auc:.3f} test_auc={test_auc:.3f}", flush=True)

print(f"[smoke] mean valid_auc={np.mean(valid_aucs):.3f} mean test_auc={np.mean(test_aucs):.3f}", flush=True)
print("[smoke] DONE", flush=True)
