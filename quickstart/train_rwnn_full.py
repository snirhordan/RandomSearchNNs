#!/usr/bin/env python3
# coding: utf-8
"""Full training script for RWNN on ClinTox.

Mirrors `train_rwnn.ipynb` hyperparameters. Trains over n_splits random splits
with early stopping (patience 10) up to 200 epochs each, on the full dataset.
Logs per-epoch train/valid losses + valid AUC, saves checkpoints + JSON metrics.
"""
import os
import sys
import time
import json
import copy
import random
import argparse
import numpy as np
import regex as re
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# -- repo on sys.path --
cwd_path = Path(__file__).resolve().parent
sys.path.append(str(cwd_path.parent))

from generation.utils import (
    get_canonical_molecule, mol2graph, get_ids,
)
from generation.scaffold_split import random_split
from utils.search import (
    sample_walks, sample_walks_adaptive,
    sample_walks_mdlr, sample_walks_mdlr_adaptive,
    sample_walks_rum, sample_walks_rum_adaptive,
    sample_dfs, sample_bfs,
)


# ------------------------ args ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--n_splits', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--h_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--m', type=int, default=8, help='num walks per graph')
parser.add_argument('--w', type=int, default=8, help='walk window param')
parser.add_argument('--reduce', type=str, default='mean')
parser.add_argument('--walk_type', type=str, default='walk_ada')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--out_dir', type=str,
                    default=str(cwd_path.parent / 'runs'))
parser.add_argument('--run_name', type=str, default='rwnn_full')
args = parser.parse_args()

# seeds
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
print(f"[full] device={device}  seed={SEED}", flush=True)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir = out_dir / args.run_name
ckpt_dir.mkdir(parents=True, exist_ok=True)


# ------------------------ Step 1: data prep ------------------------
def process_tok(components):
    out = []
    for component in components:
        if component.startswith('[') and ':' in component:
            base = component.split(':')[0][1:]
            out.append(base)
    return out


idx_y = 2
idx_smiles = 0
data_path = str(cwd_path / 'data' / 'clintox.csv')

df = pd.read_csv(data_path, sep=',')
data = df.to_numpy()

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex_tokenizer = re.compile(PATTERN)

vocab_list = []
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
        vocab_list += smiles_tok

vocab_d = {}
vocab_list = list(np.unique(vocab_list))
vocab_list.append('PAD')
for i, key in enumerate(vocab_list):
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

print(f"[full] loaded {len(graphs)} graphs, vocab size={len(vocab)}, l_max={l_max}", flush=True)

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
        if self.walk_type == 'walk':
            data = sample_walks(data, self.m, self.l, self.w, self.nb)
        elif self.walk_type == 'walk_ada':
            data = sample_walks_adaptive(data, self.m, data.x.shape[0], self.w, self.nb, self.max_len, self.vocab)
        elif self.walk_type == 'mdlr':
            data = sample_walks_mdlr(data, self.m, self.l, self.w, self.nb)
        elif self.walk_type == 'mdlr_ada':
            data = sample_walks_mdlr_adaptive(data, self.m, data.x.shape[0], self.w, self.nb, self.max_len, self.vocab)
        elif self.walk_type == 'rum':
            data = sample_walks_rum(data, self.m, self.l, self.w, self.nb)
        elif self.walk_type == 'rum_ada':
            data = sample_walks_rum_adaptive(data, self.m, data.x.shape[0], self.w, self.nb, self.max_len, self.vocab)
        elif self.walk_type == 'search':
            data = sample_dfs(data, self.m, self.w, self.max_len, self.vocab)
        elif self.walk_type == 'search_bfs':
            data = sample_bfs(data, self.m, self.w, self.max_len, self.vocab)
        return data


# ------------------------ Step 3: training ------------------------
dropout = 0
h_dim = args.h_dim
num_layers = args.num_layers
m = args.m
l = None
w = args.w
nb = False
lr = args.lr
batch_size = args.batch_size
n_splits = args.n_splits
reduce = args.reduce
out_dim = 1
pe_out_dim = 16
walk_type = args.walk_type
epochs = args.epochs
early_stopping = args.early_stopping
MIN_DELTA = 0

run_metrics = {
    'hyperparams': {
        'seed': SEED,
        'h_dim': h_dim,
        'num_layers': num_layers,
        'm': m,
        'w': w,
        'nb': nb,
        'lr': lr,
        'batch_size': batch_size,
        'n_splits': n_splits,
        'reduce': reduce,
        'epochs': epochs,
        'early_stopping': early_stopping,
        'walk_type': walk_type,
        'pe_out_dim': pe_out_dim,
        'dataset': 'clintox',
        'l_max': int(l_max),
        'vocab_size': len(vocab),
        'n_graphs': len(graphs),
    },
    'splits': [],
}

t_global = time.time()
peak_mem_mb = 0.0
if device.type == 'cuda':
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"[full] reset_peak_memory_stats failed: {e}", flush=True)

valid_aucs, test_aucs = [], []
for s in range(n_splits):
    split_t0 = time.time()
    pe_in_dim = 2 * w
    model = RWNN(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_idx, valid_idx, test_idx = splits['train'][s], splits['valid'][s], splits['test'][s]
    print(f"[full] split {s}: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}", flush=True)

    train_data = SMILESDataset(get_ids(graphs, train_idx), get_ids(ys_data, train_idx), m, l, w, nb, l_max, vocab, walk_type)
    valid_data = SMILESDataset(get_ids(graphs, valid_idx), get_ids(ys_data, valid_idx), m, l, w, nb, l_max, vocab, walk_type)
    test_data = SMILESDataset(get_ids(graphs, test_idx), get_ids(ys_data, test_idx), m, l, w, nb, l_max, vocab, walk_type)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    best_loss = float('inf')
    best_state = None
    stop_counter = 0
    epoch_log = []

    for epoch in range(epochs):
        if stop_counter >= early_stopping:
            print(f"[full] split {s} early stop at epoch {epoch}", flush=True)
            break
        t0 = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).squeeze()
            loss = criterion(out, batch.ys.squeeze())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        model.eval()
        valid_losses = []
        v_ys, v_outs = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                out = model(batch).squeeze()
                valid_losses.append(criterion(out, batch.ys.squeeze()).detach().cpu().numpy())
                v_outs.append(out.detach().cpu().numpy())
                v_ys.append(batch.ys.squeeze().detach().cpu().numpy())
        valid_loss = float(np.mean(valid_losses))
        try:
            valid_auc = float(roc_auc_score(np.concatenate(v_ys, axis=0), np.concatenate(v_outs, axis=0)))
        except ValueError:
            valid_auc = float('nan')

        dt = time.time() - t0
        if valid_loss < best_loss - MIN_DELTA:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stop_counter = 0
            improved = True
        else:
            stop_counter += 1
            improved = False
        print(f"[full] split {s} epoch {epoch:3d} train_loss={train_loss:.4f} "
              f"valid_loss={valid_loss:.4f} valid_auc={valid_auc:.3f} dt={dt:.1f}s "
              f"{'*' if improved else ' '}", flush=True)
        epoch_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'valid_auc': valid_auc,
            'dt_sec': dt,
            'improved': improved,
        })

    # eval with best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ys, outs = [], []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            outs.append(model(batch).detach().cpu().numpy())
            ys.append(batch.ys.detach().cpu().numpy())
    try:
        valid_auc = float(roc_auc_score(np.concatenate(ys, axis=0), np.concatenate(outs, axis=0)))
    except ValueError:
        valid_auc = float('nan')
    valid_aucs.append(valid_auc)

    ys, outs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outs.append(model(batch).detach().cpu().numpy())
            ys.append(batch.ys.detach().cpu().numpy())
    try:
        test_auc = float(roc_auc_score(np.concatenate(ys, axis=0), np.concatenate(outs, axis=0)))
    except ValueError:
        test_auc = float('nan')
    test_aucs.append(test_auc)

    split_dt = time.time() - split_t0
    print(f"[full] split {s} DONE valid_auc={valid_auc:.3f} test_auc={test_auc:.3f} "
          f"split_dt={split_dt:.1f}s", flush=True)

    # save checkpoint
    ckpt_path = ckpt_dir / f"split{s}_best.pt"
    torch.save({
        'state_dict': best_state if best_state is not None else model.state_dict(),
        'valid_auc': valid_auc,
        'test_auc': test_auc,
        'split': s,
        'hyperparams': run_metrics['hyperparams'],
    }, ckpt_path)

    if device.type == 'cuda':
        try:
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            if peak > peak_mem_mb:
                peak_mem_mb = peak
        except Exception:
            pass

    run_metrics['splits'].append({
        'split': s,
        'epochs_run': len(epoch_log),
        'best_valid_loss': best_loss,
        'final_valid_auc': valid_auc,
        'final_test_auc': test_auc,
        'split_dt_sec': split_dt,
        'epochs': epoch_log,
    })

total_dt = time.time() - t_global
mean_test = float(np.mean(test_aucs))
std_test = float(np.std(test_aucs))
mean_valid = float(np.mean(valid_aucs))
std_valid = float(np.std(valid_aucs))

run_metrics['summary'] = {
    'mean_test_auc': mean_test,
    'std_test_auc': std_test,
    'mean_valid_auc': mean_valid,
    'std_valid_auc': std_valid,
    'total_wall_sec': total_dt,
    'peak_gpu_mem_mb': peak_mem_mb,
}

metrics_path = ckpt_dir / 'metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(run_metrics, f, indent=2)

print(f"[full] DONE mean_test_auc={mean_test:.3f} +/- {std_test:.3f} "
      f"mean_valid_auc={mean_valid:.3f} +/- {std_valid:.3f} "
      f"total={total_dt:.1f}s peak_gpu_mem={peak_mem_mb:.1f}MB", flush=True)
print(f"[full] metrics saved -> {metrics_path}", flush=True)
