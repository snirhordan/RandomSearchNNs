#!/usr/bin/env python
# coding: utf-8

# # Quickstart Notebook for Training RWNNs
# 
# ## Step 1: Preparing the Dataset
# 
# This notebook prepares the **ClinTox** dataset for graph-based molecular learning. At a high level, the pipeline:
# 
# 1. loads raw SMILES strings and labels from CSV,
# 2. builds a SMILES token vocabulary,
# 3. converts valid molecules into graph objects,
# 4. creates train/validation/test splits, and
# 5. stores everything in a single dataset dictionary for downstream training.
# 
# The bulk of the work lies in tokenizing molecules and storing for each atom an embedding index according to its canonical smiles token. 

# In[80]:


import os
import sys
import argparse
import numpy as np
import regex as re
import pandas as pd
import pickle as pkl
from pathlib import Path

import torch
from rdkit import Chem
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

# assumes generation folder and scripts lie in parent directory as in the github
cwd_path = Path.cwd()
sys.path.append(str(cwd_path.parent))
from generation.utils import *
from generation.scaffold_split import *

def process_tok(components): 
    modified_components = []
    for component in components:
        if component.startswith('[') and ':' in component:
            # Extract the map number
            atom_number = int(component.split(':')[1][:-1]) - 1  # Extracts 1 from "[C:1]"
            # Remove the atom number from the component
            base_component = component.split(':')[0][1:] # Removes map number, e.g., "[C:1]" becomes "[C]"
            modified_components.append(base_component)

    return modified_components
    
idx_y = 2
scaffold = 0
idx_smiles = 0
data_path = f'./data/clintox.csv'

df = pd.read_csv(data_path, sep=',')
data = df.to_numpy()

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex_tokenizer = re.compile(PATTERN)

vocab = []
l_max = -1
for idx, smiles in enumerate(data[:, idx_smiles]): 
    mol = Chem.MolFromSmiles(smiles)
    if mol != None and np.isnan(data[idx, idx_y]) == False: 
        if mol.GetNumAtoms() > l_max:
            l_max = mol.GetNumAtoms()
        mol = get_canonical_molecule(mol)
        
        smiles = Chem.MolToSmiles(mol, doRandom=False, canonical=True, allHsExplicit=False, ignoreAtomMapNumbers=False)
        smiles_tok = process_tok(regex_tokenizer.findall(smiles))
        vocab += smiles_tok

vocab_d = {}
vocab = list(np.unique(vocab))
vocab.append('PAD')
for i, key in enumerate(vocab): 
    vocab_d[key] = i

vocab = vocab_d

ys_data, graphs, smiles_scaffold = [], [], []
for idx, smiles in enumerate(data[:, idx_smiles]): 
    mol = Chem.MolFromSmiles(smiles)
    if mol != None and np.isnan(data[idx, idx_y]) == False: 
        mol = get_canonical_molecule(mol)
        ys = np.array(data[idx, idx_y], dtype=np.float32)
        ys_data.append(torch.from_numpy(ys)) 
        graphs.append(mol2graph(mol, tokenizer=regex_tokenizer, vocab=vocab))
        smiles_scaffold.append(smiles)

splits = random_split(len(graphs), test_size=0.2, val_size=0.2, random_state=0)

# save raw dataset and tokenizer
data_dict = {}
data_dict['l_max'] = l_max
data_dict['vocab'] = vocab
data_dict['graphs'] = graphs
data_dict['ys_data'] = ys_data
data_dict['splits'] = splits


# In[79]:


# example check: get_canonical_molecule sets atom map numbers, allowing us to map atom indices to smiles tokens
mol = Chem.MolFromSmiles(data[-1, idx_smiles])
print(Chem.MolToSmiles(mol, doRandom=False, canonical=True, allHsExplicit=False, ignoreAtomMapNumbers=False))

mol = Chem.MolFromSmiles(data[-1, idx_smiles])
mol = get_canonical_molecule(mol)
print(Chem.MolToSmiles(mol, doRandom=False, canonical=True, allHsExplicit=False, ignoreAtomMapNumbers=False))


# ## Step 2: Defining the RWNN Model
# 
# We now define the **Random Walk Neural Network (RWNN)** used for molecular prediction. At a high level, the model treats each graph as a **set of random walks**, processes each walk with a sequence model, and then aggregates information across walks to obtain a graph-level prediction.
# 
# ---
# 
# ## High-level idea
# 
# The RWNN consists of three main stages:
# 
# 1. **Token embedding**: convert each token in a walk into a learned vector representation.
# 2. **Sequence encoding**: process each walk with an LSTM to obtain a fixed-dimensional representation.
# 3. **Set aggregation and prediction**: pool representations across all walks from the same graph, then apply an MLP readout for final prediction.

# In[84]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader

class RWNN(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.RNN = nn.LSTM(hid_dim, hid_dim, num_layers, batch_first=True, bidirectional=True)
        
        self.readout = torch.nn.ModuleList()
        self.readout.append(torch.nn.Linear(hid_dim*2, hid_dim*2))
        self.readout.append(torch.nn.Linear(hid_dim*2, out_dim))

        self.pe_encoding = torch.nn.Linear(pe_in_dim, pe_out_dim)
        self.embedding = nn.Embedding(n_emb, hid_dim, n_emb-1)
        
        self.reduce = reduce
        self.num_layers = num_layers

    def forward(self, batch):

        walk_emb = batch.walk_emb
        walk_ids = batch.walk_ids
        encoding = batch.walk_pe
        lengths = batch.lengths.cpu()

        # create the id tensor since we'll flatten the set id dimension
        ids = torch.arange(walk_ids.shape[0])[:, None]
        ids = torch.broadcast_to(ids, (walk_ids.shape[0], walk_ids.shape[1]))
        ids = torch.flatten(ids, start_dim=0, end_dim=1).to(walk_ids.device)

        # construct x
        # x = torch.cat([self.embedding(walk_emb), self.pe_encoding(encoding)], dim=-1)

        x = pack_padded_sequence(self.embedding(walk_emb), lengths, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.RNN(x) # initialize hidden state to 0s at layer 0

        x = torch.cat((h[-2], h[-1]), dim=1)
        x = scatter(x, ids, dim=0, reduce=self.reduce) # pool across walks taken along the same graph

        # final readout layers over pooled node representations
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)

        return x


# ## Step 3: Training the RWNN
# 
# We now train the RWNN on the processed molecular graphs. At a high level, the training pipeline:
# 
# 1. wraps the graphs in a dataset that samples random walks on the fly,
# 2. builds train/validation/test loaders for each split,
# 3. initializes an RWNN and Adam optimizer,
# 4. trains with binary cross-entropy loss and early stopping,
# 5. reloads the best checkpoint, and
# 6. reports validation/test AUC across splits. :contentReference[oaicite:0]{index=0}
# 
# The main sampling logic lives in `SMILESDataset`, which takes each molecular graph and applies a specified walk extractor before it is passed to the model. In the current script, the chosen walk type is `walk_ada`, so each graph is represented by a sampled set of adaptive random walks during training.

# In[87]:


# load python modules
import os
import copy
import argparse
import random
import math
import numpy as np
import pickle as pkl
import networkx as nx
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

# load custom modules
from utils.utils import *
from utils.search import *

SEED = 2024
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    
# load arguments
device_idx = 0
dropout = 0
h_dim = 128
num_layers = 2
m = 8
l = None
w = 8
nb = False
lr = .001
batch_size = 128
n_splits = 3
reduce = 'mean'
model_idx = 0
model_name = 'RWNN'

out_dim =  1    
pe_out_dim = 16

graphs = data_dict['graphs']
ys_data = data_dict['ys_data']
l_max = data_dict['l_max']
vocab = data_dict['vocab']
splits = data_dict['splits']

# iterate over data splits
valid_aucs, test_aucs = [], []
for s in range(n_splits):
    # build the model and optimizer

    pe_in_dim = 2 * w
    model = RWNN(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # initializations
    epochs = 200
    early_stopping = 10
    stop_counter = 0
    best_loss = float('inf')
    MIN_DELTA = 0
    best_state = None
    
    walk_type = 'walk_ada'

    train_idx, valid_idx, test_idx = splits['train'][s], splits['valid'][s], splits['test'][s]
    train_data = SMILESDataset(get_ids(graphs, train_idx), get_ids(ys_data, train_idx), m, l, w, nb, l_max, vocab, walk_type)
    valid_data = SMILESDataset(get_ids(graphs, valid_idx), get_ids(ys_data, valid_idx), m, l, w, nb, l_max, vocab, walk_type)
    test_data = SMILESDataset(get_ids(graphs, test_idx), get_ids(ys_data, test_idx), m, l, w, nb, l_max, vocab, walk_type)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # model training loop
    model.train()
    for epoch in range(epochs):
        if early_stopping == stop_counter or epoch == epochs - 1:
            break

        for batch in train_loader: 
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).squeeze()
            loss = criterion(out, batch.ys.squeeze())
            loss.backward()
            optimizer.step()

        # early stopping count
        valid_losses = []
        for batch in valid_loader: 
            batch = batch.to(device)
            out = model(batch).squeeze()
            valid_losses.append(criterion(out, batch.ys.squeeze()).detach().cpu().numpy())
        valid_loss = np.mean(valid_losses)
        
        if valid_loss < best_loss - MIN_DELTA: 
            best_loss = valid_loss
            stop_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else: 
            stop_counter+=1

    # load the best model for evaluation
    model.load_state_dict(best_state)
    model.eval()
    
    ys, outs = [], []
    for batch in valid_loader:
        batch = batch.to(device)
        outs.append(model(batch).detach().cpu().numpy())
        ys.append(batch.ys.detach().cpu().numpy())
        
    valid_aucs.append(roc_auc_score(np.concatenate(ys, axis=0), np.concatenate(outs, axis=0)))

    ys, outs = [], []
    for batch in test_loader: 
        batch = batch.to(device)
        outs.append(model(batch).detach().cpu().numpy())
        ys.append(batch.ys.detach().cpu().numpy())

    test_aucs.append(roc_auc_score(np.concatenate(ys, axis=0), np.concatenate(outs, axis=0)))

# save results
print(f'Average test accuracy and standard deviation is {np.mean(test_aucs):.3f} +/- {np.std(test_aucs):.3f}')

