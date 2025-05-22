# load python modules
import os
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
from models.rwnn import *
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
        
        return data

if __name__ == '__main__': 
        
    parser = argparse.ArgumentParser()
    
    # dataset parameters
    parser.add_argument('--device_idx', type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--data_name', type=str, default='HIV')
    
    # model hyperparameters
    parser.add_argument('--model_idx', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--l', type=int, default=25)
    parser.add_argument('--w', type=int, default=8)
    parser.add_argument('--nb', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--reduce', type=str, default='max')
    parser.add_argument('--n_splits', type=int, default=10)
    
    # load arguments
    args = parser.parse_args()
    device_idx = args.device_idx
    data_name = args.data_name
    save_path = args.save_path
    dropout = args.dropout
    h_dim = args.h_dim
    kernel_size = args.kernel_size
    num_layers = args.num_layers
    m = args.m
    l = args.l
    w = args.w
    nb = args.nb
    lr = args.lr
    batch_size = args.batch_size
    n_splits = args.n_splits
    reduce = args.reduce
    model_idx = args.model_idx
    model_name = args.model_name
    
    # create model path and data path
    model_path = f'{data_name}_{model_name}_{model_idx}.pkl'
    data_path = f'/data1/mbito/benchmarks_proc/molecule_learning/{data_name}.pkl'

    # load the data
    device = torch.device(f'cuda:{device_idx}')
    with open(data_path, 'rb') as file: 
        data_dict = pkl.load(file)

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
        if model_name == 'RWNN_crwl': 
            pe_in_dim = 2 * w
            model = RWNN(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_base': 
            pe_in_dim = 2 * w
            model = RWNN_base(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_base_ada': 
            pe_in_dim = 2 * w
            model = RWNN_base_ada(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_mdlr': 
            model = RWNN_mdlr(h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_mdlr_ada': 
            model = RWNN_mdlr_ada(h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_rum': 
            model = RWNN_rum(h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_rum_ada': 
            model = RWNN_rum_ada(h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_crwl_ada': 
            pe_in_dim = 2 * w
            model = RSNN(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_LSTM_crwl_ada': 
            pe_in_dim = 2 * w
            model = RSNN_LSTM(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RWNN_TRSF_crwl_ada': 
            pe_in_dim = 2 * w
            model = RSNN_TRSF(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RSNN': 
            pe_in_dim = w
            model = RSNN(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RSNN_LSTM': 
            pe_in_dim = w
            model = RSNN_LSTM(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        elif model_name == 'RSNN_TRSF': 
            pe_in_dim = w
            model = RSNN_TRSF(pe_in_dim, pe_out_dim, h_dim, out_dim, num_layers, len(vocab), reduce).to(device)
        else: 
            raise NotImplementedError()

        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # initializations
        epochs = 200
        early_stopping = 10
        stop_counter = 0
        best_loss = float('inf')
        MIN_DELTA = 0

        if 'crwl_ada' in model_name: 
            walk_type = 'walk_ada'
        elif 'mdlr_ada' in model_name: 
            walk_type = 'mdlr_ada'
        elif 'rum_ada' in model_name: 
            walk_type = 'rum_ada'
        elif 'base_ada' in model_name: 
            walk_type = 'walk_ada'
        elif 'mdlr' in model_name: 
            walk_type = 'mdlr'
        elif 'rum' in model_name: 
            walk_type = 'rum'  
        elif 'RWNN' in model_name: 
            walk_type = 'walk'
        elif 'RSNN' in model_name: 
            walk_type = 'search'

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
                loss = F.binary_cross_entropy(out, batch.ys.squeeze())
                loss.backward()
                optimizer.step()

            # early stopping count
            valid_losses = []
            for batch in valid_loader: 
                batch = batch.to(device)
                out = model(batch).squeeze()
                valid_losses.append(F.binary_cross_entropy(out, batch.ys.squeeze()).detach().cpu().numpy())
            valid_loss = np.mean(valid_losses)
            
            if valid_loss < best_loss - MIN_DELTA: 
                best_loss = valid_loss
                stop_counter = 0
                
                # save the best model for evaluation
                # torch.save(model, os.path.join('/data/mbito/models/', model_path))
                torch.save(model.state_dict(), os.path.join('/data1/mbito/models/', model_path))
            else: 
                stop_counter+=1

        # load the best model for evaluation
        # best_model = torch.load(os.path.join('/data/mbito/models/', model_path))
        model.load_state_dict(torch.load(os.path.join('/data1/mbito/models/', model_path)))
        model.eval()
        ys, outs = [], []

        for batch in valid_loader:
            batch = batch.to(device)
            outs.append(model(batch).detach().cpu().numpy().flatten())
            ys.append(batch.ys.detach().cpu().numpy().flatten())
            
        valid_aucs.append(roc_auc_score(np.concatenate(ys), np.concatenate(outs)))

        ys, outs = [], []
        for batch in test_loader: 
            batch = batch.to(device)
            outs.append(model(batch).detach().cpu().numpy().flatten())
            ys.append(batch.ys.detach().cpu().numpy().flatten())

        test_aucs.append(roc_auc_score(np.concatenate(ys), np.concatenate(outs)))
    
    # save results
    print(f'Finished training model {model_idx}!')
        
    save_dict = vars(args)
    save_dict['valid_aucs'] = valid_aucs
    save_dict['test_aucs'] = test_aucs
    with open(os.path.join(save_path, model_path), 'wb') as file: 
        pkl.dump(save_dict, file)




