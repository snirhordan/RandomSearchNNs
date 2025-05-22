import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.nn import GINConv

class RWNN(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.rnn_layers = torch.nn.ModuleList()
        self.rnn_layers.append(nn.GRU(hid_dim+pe_out_dim, hid_dim, 1, batch_first=True, bidirectional=True))

        for nl in range(num_layers - 1): 
            self.rnn_layers.append(nn.GRU(2*hid_dim, hid_dim, 1, batch_first=True, bidirectional=True))
        
        self.readout = torch.nn.ModuleList()
        self.readout.append(torch.nn.Linear(2*hid_dim, 2*hid_dim))
        self.readout.append(torch.nn.Linear(2*hid_dim, out_dim))

        self.pe_encoding = torch.nn.Linear(pe_in_dim, pe_out_dim)
        self.embedding = nn.Embedding(n_emb, hid_dim, n_emb-1)
        
        self.reduce = reduce
        self.num_layers = num_layers

    def forward(self, batch):

        walk_emb = batch.walk_emb
        walk_ids = batch.walk_ids
        encoding = batch.walk_pe
        
        graph_ns = [torch.max(walk_ids[i, :, :]) for i in range(walk_ids.shape[0])] # b length vector containing max number of nodes in each set of walks
        walk_ids_proc = []

        # compute added component to node_ids; we need to make each node_id unique due to flattening in scatter operation
        for i in range(walk_ids.shape[0]): 
            if i == 0: 
                # add 0 for first set in batch since these are already unique
                walk_ids_proc.append(torch.zeros((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device))
            else: 
                # add the sum of max nodes up to current set + the number of sets before current set to current node ids
                mult = sum(graph_ns[:i]) + i 
                walk_ids_proc.append(torch.ones((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device) * mult)

        walk_ids_proc = torch.flatten(walk_ids + torch.cat(walk_ids_proc, dim=0), start_dim=0, end_dim=1) # add processed components to node_ids, b * n_set x n_seq
        walk_ids_proc_flat = torch.flatten(walk_ids_proc, start_dim=0, end_dim=1) # b * n_set x n_seq ---> b * n_set * n_seq

        # construct x
        x = torch.cat([self.embedding(walk_emb), self.pe_encoding(encoding)], dim=-1)

        for l in range(self.num_layers):
            if l == 0: 
                x, h = self.rnn_layers[l](x) # initialize hidden state to 0s at layer 0
            else: 
                x, h = self.rnn_layers[l](x, h) # initialize hidden state to previous hidden state after layer 0

            node_agg = torch.flatten(x, start_dim=0, end_dim=1) # b * n_set * n_seq x d
            # node_agg = node_agg[node_ids_flat != -1, :] # use original node_ids as mask to remove non-node representations
            
            # use the scatter operation to aggregate node representations. node_agg.shape[0] should match node_ids_proc_flat_masked.shape[0] if all operations were done correctly
            node_agg = scatter(node_agg, walk_ids_proc_flat, dim=0, reduce='mean') # |V_1| + ... + |V_b| x d

            if l != self.num_layers-1: # no need to do this step at the last layer since we pool over node_agg to obtain final representations
                # construct x ---> b * nw x l x d matrix
                x_new = node_agg[walk_ids_proc_flat, :] # b * nw * l x d
                x_new = x_new.reshape(walk_ids_proc.shape[0], walk_ids_proc.shape[1], node_agg.shape[-1]) # b * nw x l x d
                x = x_new

        # this version pools over node representations
        # create a |V_0| + ... + |V_b-1| ids vector, where first |V_0| entries are 0s, next |V_1| entries are 1s, etc...
        graph_ids = torch.cat([torch.ones((graph_ns[i]+1, ), dtype=int) * i for i in range(walk_ids.shape[0])]).to(x.device)
        x = scatter(node_agg, graph_ids, dim=0, reduce=self.reduce) # pool across node dimension; node_agg.shape[0] = graphs_ids.shape[0] 

        # final readout layers over pooled node representations
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)

        return x

class RSNN(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.rnn_layers = torch.nn.ModuleList()
        self.rnn_layers.append(nn.GRU(hid_dim+pe_out_dim, hid_dim, 1, batch_first=True, bidirectional=True))

        for nl in range(num_layers - 1): 
            self.rnn_layers.append(nn.GRU(2*hid_dim, hid_dim, 1, batch_first=True, bidirectional=True))
        
        self.readout = torch.nn.ModuleList()
        self.readout.append(torch.nn.Linear(2*hid_dim, 2*hid_dim))
        self.readout.append(torch.nn.Linear(2*hid_dim, out_dim))

        self.pe_encoding = torch.nn.Linear(pe_in_dim, pe_out_dim)
        self.embedding = nn.Embedding(n_emb, hid_dim, n_emb-1)
        
        self.reduce = reduce
        self.num_layers = num_layers

    def forward(self, batch):

        walk_emb = batch.walk_emb
        walk_ids = batch.walk_ids
        encoding = batch.walk_pe
        lengths = batch.lengths.cpu()
        
        graph_ns = [torch.max(walk_ids[i, :, :]) for i in range(walk_ids.shape[0])] # b length vector containing max number of nodes in each set of walks
        walk_ids = walk_ids[:, :, :torch.max(lengths)] # b x n_set x n_seq; process node_ids to drop padding
        walk_ids_proc = []

        # compute added component to node_ids; we need to make each node_id unique due to flattening in scatter operation
        for i in range(walk_ids.shape[0]): 
            if i == 0: 
                # add 0 for first set in batch since these are already unique
                walk_ids_proc.append(torch.zeros((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device))
            else: 
                # add the sum of max nodes up to current set + the number of sets before current set to current node ids
                mult = sum(graph_ns[:i]) + i 
                walk_ids_proc.append(torch.ones((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device) * mult)

        walk_ids_flat = torch.flatten(walk_ids, start_dim=0, end_dim=2) # b x n_set x n_seq --->  b * n_set * n_seq
        walk_ids_proc = torch.flatten(walk_ids + torch.cat(walk_ids_proc, dim=0), start_dim=0, end_dim=1) # add processed components to node_ids, b * n_set x n_seq
        walk_ids_proc_flat = torch.flatten(walk_ids_proc, start_dim=0, end_dim=1) # b * n_set x n_seq ---> b * n_set * n_seq
        walk_ids_proc_flat_masked = walk_ids_proc_flat[walk_ids_flat != -1] # use original node_ids as mask to remove non-node ids (-1)

        # construct x 
        # x = walk_emb[walk_ids_proc_flat] # b * nw * l x 1
        # x = x.reshape(walk_ids_proc.shape[0], walk_ids_proc.shape[1]) # b * nw x l x 1
        x = torch.cat([self.embedding(walk_emb), self.pe_encoding(encoding)], dim=-1)

        for l in range(self.num_layers):
            # print(l, x.shape, lengths.shape)
            # pack sequence with lengths
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
            if l == 0: 
                x, h = self.rnn_layers[l](x) # initialize hidden state to 0s at layer 0
            else: 
                x, h = self.rnn_layers[l](x, h) # initialize hidden state to previous hidden state after layer 0

            # pad sequence; this should remove padding from x (check!)
            x, ls = pad_packed_sequence(x, batch_first=True) # b * n_set x n_seq x d

            node_agg = torch.flatten(x, start_dim=0, end_dim=1) # b * n_set * n_seq x d
            node_agg = node_agg[walk_ids_flat != -1, :] # use original node_ids as mask to remove non-node representations
            
            # use the scatter operation to aggregate node representations. node_agg.shape[0] should match node_ids_proc_flat_masked.shape[0] if all operations were done correctly
            node_agg = scatter(node_agg, walk_ids_proc_flat_masked, dim=0, reduce='mean') # |V_1| + ... + |V_b| x d

            # flatten x so we can add node_agg to node representations in x
            x_flat = torch.flatten(x, start_dim=0, end_dim=1) # b * n_set * n_seq x d

            if l != self.num_layers-1: # no need to do this step at the last layer since we pool over node_agg to obtain final representations
                # loopy version
                # for i in range(x.shape[0]): 
                #     x[i, node_ids_2dim[i, :] != -1, :] = node_agg[node_ids_proc[i, node_ids_2dim[i, :] != -1], :]

                # non-loopy version
                # select only node representations from x_flat and set them equal to their representations in node_agg
                # this works since indices in x_flat (masked) directly corresponds to node indices in node_ids_proc_flat_masked! 
                x_flat[walk_ids_flat!=-1, :] = node_agg[walk_ids_proc_flat_masked, :]
                x = x_flat.reshape(x.shape)

        # this version pools over node representations
        # create a |V_0| + ... + |V_b-1| ids vector, where first |V_0| entries are 0s, next |V_1| entries are 1s, etc...
        graph_ids = torch.cat([torch.ones((graph_ns[i]+1, ), dtype=int) * i for i in range(walk_ids.shape[0])]).to(x.device)
        x = scatter(node_agg, graph_ids, dim=0, reduce=self.reduce) # pool across node dimension; node_agg.shape[0] = graphs_ids.shape[0] 

        # final readout layers over pooled node representations
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)

        return x

class RSNN_LSTM(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.rnn_layers = torch.nn.ModuleList()
        self.rnn_layers.append(nn.LSTM(hid_dim+pe_out_dim, hid_dim, 1, batch_first=True, bidirectional=True))

        for nl in range(num_layers - 1): 
            self.rnn_layers.append(nn.LSTM(2*hid_dim, hid_dim, 1, batch_first=True, bidirectional=True))
        
        self.readout = torch.nn.ModuleList()
        self.readout.append(torch.nn.Linear(2*hid_dim, 2*hid_dim))
        self.readout.append(torch.nn.Linear(2*hid_dim, out_dim))

        self.pe_encoding = torch.nn.Linear(pe_in_dim, pe_out_dim)
        self.embedding = nn.Embedding(n_emb, hid_dim, n_emb-1)
        
        self.reduce = reduce
        self.num_layers = num_layers

    def forward(self, batch):

        walk_emb = batch.walk_emb
        walk_ids = batch.walk_ids
        encoding = batch.walk_pe
        lengths = batch.lengths.cpu()
        
        graph_ns = [torch.max(walk_ids[i, :, :]) for i in range(walk_ids.shape[0])] # b length vector containing max number of nodes in each set of walks
        walk_ids = walk_ids[:, :, :torch.max(lengths)] # b x n_set x n_seq; process node_ids to drop padding
        walk_ids_proc = []

        # compute added component to node_ids; we need to make each node_id unique due to flattening in scatter operation
        for i in range(walk_ids.shape[0]): 
            if i == 0: 
                # add 0 for first set in batch since these are already unique
                walk_ids_proc.append(torch.zeros((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device))
            else: 
                # add the sum of max nodes up to current set + the number of sets before current set to current node ids
                mult = sum(graph_ns[:i]) + i 
                walk_ids_proc.append(torch.ones((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device) * mult)

        walk_ids_flat = torch.flatten(walk_ids, start_dim=0, end_dim=2) # b x n_set x n_seq --->  b * n_set * n_seq
        walk_ids_proc = torch.flatten(walk_ids + torch.cat(walk_ids_proc, dim=0), start_dim=0, end_dim=1) # add processed components to node_ids, b * n_set x n_seq
        walk_ids_proc_flat = torch.flatten(walk_ids_proc, start_dim=0, end_dim=1) # b * n_set x n_seq ---> b * n_set * n_seq
        walk_ids_proc_flat_masked = walk_ids_proc_flat[walk_ids_flat != -1] # use original node_ids as mask to remove non-node ids (-1)

        # construct x 
        # x = walk_emb[walk_ids_proc_flat] # b * nw * l x 1
        # x = x.reshape(walk_ids_proc.shape[0], walk_ids_proc.shape[1]) # b * nw x l x 1
        x = torch.cat([self.embedding(walk_emb), self.pe_encoding(encoding)], dim=-1)

        for l in range(self.num_layers):
            # print(l, x.shape, lengths.shape)
            # pack sequence with lengths
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
            if l == 0: 
                x, h = self.rnn_layers[l](x) # initialize hidden state to 0s at layer 0
            else: 
                x, h = self.rnn_layers[l](x, h) # initialize hidden state to previous hidden state after layer 0

            # pad sequence; this should remove padding from x (check!)
            x, ls = pad_packed_sequence(x, batch_first=True) # b * n_set x n_seq x d

            node_agg = torch.flatten(x, start_dim=0, end_dim=1) # b * n_set * n_seq x d
            node_agg = node_agg[walk_ids_flat != -1, :] # use original node_ids as mask to remove non-node representations
            
            # use the scatter operation to aggregate node representations. node_agg.shape[0] should match node_ids_proc_flat_masked.shape[0] if all operations were done correctly
            node_agg = scatter(node_agg, walk_ids_proc_flat_masked, dim=0, reduce='mean') # |V_1| + ... + |V_b| x d

            # flatten x so we can add node_agg to node representations in x
            x_flat = torch.flatten(x, start_dim=0, end_dim=1) # b * n_set * n_seq x d

            if l != self.num_layers-1: # no need to do this step at the last layer since we pool over node_agg to obtain final representations
                # loopy version
                # for i in range(x.shape[0]): 
                #     x[i, node_ids_2dim[i, :] != -1, :] = node_agg[node_ids_proc[i, node_ids_2dim[i, :] != -1], :]

                # non-loopy version
                # select only node representations from x_flat and set them equal to their representations in node_agg
                # this works since indices in x_flat (masked) directly corresponds to node indices in node_ids_proc_flat_masked! 
                x_flat[walk_ids_flat!=-1, :] = node_agg[walk_ids_proc_flat_masked, :]
                x = x_flat.reshape(x.shape)

        # this version pools over node representations
        # create a |V_0| + ... + |V_b-1| ids vector, where first |V_0| entries are 0s, next |V_1| entries are 1s, etc...
        graph_ids = torch.cat([torch.ones((graph_ns[i]+1, ), dtype=int) * i for i in range(walk_ids.shape[0])]).to(x.device)
        x = scatter(node_agg, graph_ids, dim=0, reduce=self.reduce) # pool across node dimension; node_agg.shape[0] = graphs_ids.shape[0] 

        # final readout layers over pooled node representations
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)

        return x

def sinusoidal_positional_encoding(max_len: int, d_model: int, device=None) -> torch.Tensor:
    """
    Create the sinusoidal positional-encoding matrix used in the original Transformer.

    Parameters
    ----------
    max_len : int
        Maximum sequence length you expect at run time.
    d_model : int
        Embedding (model-hidden) dimension.
    device : torch.device or str, optional
        Where to place the returned tensor.  If None, stays on CPU.

    Returns
    -------
    pe : torch.Tensor, shape (max_len, d_model)
        The positional-encoding matrix.  Row `pos` contains the encoding
        for position `pos` (0-indexed).  Even dimensions use sine; odd
        dimensions use cosine.
    """
    # (max_len, 1)
    positions = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)

    # (1, d_model) ––  log-spaced inverse frequencies
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / d_model)
    )

    pe = torch.zeros(max_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)  # even indices  (0,2,4,…)
    pe[:, 1::2] = torch.cos(positions * div_term)  # odd indices   (1,3,5,…)

    return pe


class RSNN_TRSF(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.rnn_layers = torch.nn.ModuleList()
        # self.rnn_layers.append(nn.LSTM(hid_dim+pe_out_dim, hid_dim, 1, batch_first=True, bidirectional=True))
        self.hidp_dim = hid_dim+pe_out_dim
        self.rnn_layers.append(nn.TransformerEncoderLayer(self.hidp_dim, self.hidp_dim, batch_first=True))

        for nl in range(num_layers - 1): 
            # self.rnn_layers.append(nn.LSTM(2*hid_dim, hid_dim, 1, batch_first=True, bidirectional=True))
            self.rnn_layers.append(nn.TransformerEncoderLayer(self.hidp_dim, self.hidp_dim, batch_first=True))
        
        self.readout = torch.nn.ModuleList()
        self.readout.append(torch.nn.Linear(self.hidp_dim, self.hidp_dim))
        self.readout.append(torch.nn.Linear(self.hidp_dim, out_dim))

        self.pe_encoding = torch.nn.Linear(pe_in_dim, pe_out_dim)
        self.embedding = nn.Embedding(n_emb, hid_dim, n_emb-1)
        
        self.reduce = reduce
        self.num_layers = num_layers

    def forward(self, batch):

        walk_emb = batch.walk_emb
        walk_ids = batch.walk_ids
        encoding = batch.walk_pe
        lengths = batch.lengths
        
        graph_ns = [torch.max(walk_ids[i, :, :]) for i in range(walk_ids.shape[0])] # b length vector containing max number of nodes in each set of walks
        walk_ids = walk_ids[:, :, :torch.max(lengths)] # b x n_set x n_seq; process node_ids to drop padding
        walk_ids_proc = []

        # compute added component to node_ids; we need to make each node_id unique due to flattening in scatter operation
        for i in range(walk_ids.shape[0]): 
            if i == 0: 
                # add 0 for first set in batch since these are already unique
                walk_ids_proc.append(torch.zeros((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device))
            else: 
                # add the sum of max nodes up to current set + the number of sets before current set to current node ids
                mult = sum(graph_ns[:i]) + i 
                walk_ids_proc.append(torch.ones((1, walk_ids.shape[1], walk_ids.shape[2]), dtype=int).to(walk_emb.device) * mult)

        walk_ids_flat = torch.flatten(walk_ids, start_dim=0, end_dim=2) # b x n_set x n_seq --->  b * n_set * n_seq
        walk_ids_proc = torch.flatten(walk_ids + torch.cat(walk_ids_proc, dim=0), start_dim=0, end_dim=1) # add processed components to node_ids, b * n_set x n_seq
        walk_ids_proc_flat = torch.flatten(walk_ids_proc, start_dim=0, end_dim=1) # b * n_set x n_seq ---> b * n_set * n_seq
        walk_ids_proc_flat_masked = walk_ids_proc_flat[walk_ids_flat != -1] # use original node_ids as mask to remove non-node ids (-1)

        # construct x 
        # x = walk_emb[walk_ids_proc_flat] # b * nw * l x 1
        # x = x.reshape(walk_ids_proc.shape[0], walk_ids_proc.shape[1]) # b * nw x l x 1
        walk_emb = walk_emb[:, :torch.max(lengths)]
        encoding = encoding[:, :torch.max(lengths), :]
        x = torch.cat([self.embedding(walk_emb), self.pe_encoding(encoding)], dim=-1) # b * m x max_seq_len x d 

        pe = sinusoidal_positional_encoding(torch.max(lengths), self.hidp_dim, device=walk_emb.device)
        x = x + pe

        seq_range = torch.arange(torch.max(lengths), device=walk_emb.device)
    
        # seq_range[None, :] has shape (1, max_len)
        # lengths[:, None]  has shape (batch, 1)
        # Broadcasting gives (batch, max_len)
        pad_mask = seq_range[None, :] >= lengths[:, None]  

        for l in range(self.num_layers):
            # print(l, x.shape, lengths.shape)
            # pack sequence with lengths
            # x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
            x = self.rnn_layers[l](x, src_key_padding_mask=pad_mask) 

            # pad sequence; this should remove padding from x (check!)
            # x, ls = pad_packed_sequence(x, batch_first=True) # b * n_set x n_seq x d

            node_agg = torch.flatten(x, start_dim=0, end_dim=1) # b * n_set * n_seq x d
            node_agg = node_agg[walk_ids_flat != -1, :] # use original node_ids as mask to remove non-node representations
            
            # use the scatter operation to aggregate node representations. node_agg.shape[0] should match node_ids_proc_flat_masked.shape[0] if all operations were done correctly
            node_agg = scatter(node_agg, walk_ids_proc_flat_masked, dim=0, reduce='mean') # |V_1| + ... + |V_b| x d

            # flatten x so we can add node_agg to node representations in x
            x_flat = torch.flatten(x, start_dim=0, end_dim=1) # b * n_set * n_seq x d

            if l != self.num_layers-1: # no need to do this step at the last layer since we pool over node_agg to obtain final representations
                # loopy version
                # for i in range(x.shape[0]): 
                #     x[i, node_ids_2dim[i, :] != -1, :] = node_agg[node_ids_proc[i, node_ids_2dim[i, :] != -1], :]

                # non-loopy version
                # select only node representations from x_flat and set them equal to their representations in node_agg
                # this works since indices in x_flat (masked) directly corresponds to node indices in node_ids_proc_flat_masked! 
                x_flat[walk_ids_flat!=-1, :] = node_agg[walk_ids_proc_flat_masked, :]
                x = x_flat.reshape(x.shape)

        # this version pools over node representations
        # create a |V_0| + ... + |V_b-1| ids vector, where first |V_0| entries are 0s, next |V_1| entries are 1s, etc...
        graph_ids = torch.cat([torch.ones((graph_ns[i]+1, ), dtype=int) * i for i in range(walk_ids.shape[0])]).to(x.device)
        x = scatter(node_agg, graph_ids, dim=0, reduce=self.reduce) # pool across node dimension; node_agg.shape[0] = graphs_ids.shape[0] 

        # final readout layers over pooled node representations
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)

        return x

class RWNN_base(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.RNN = nn.GRU(hid_dim, hid_dim, num_layers, batch_first=True, bidirectional=True)
        
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

        # create the id tensor since we'll flatten the set id dimension
        ids = torch.arange(walk_ids.shape[0])[:, None]
        ids = torch.broadcast_to(ids, (walk_ids.shape[0], walk_ids.shape[1]))
        ids = torch.flatten(ids, start_dim=0, end_dim=1).to(walk_ids.device)

        # construct x
        # x = torch.cat([self.embedding(walk_emb), self.pe_encoding(encoding)], dim=-1)

        x, h = self.RNN(self.embedding(walk_emb)) # initialize hidden state to 0s at layer 0

        x = torch.cat((h[-2], h[-1]), dim=1)
        x = scatter(x, ids, dim=0, reduce=self.reduce) # pool across node dimension

        # final readout layers over pooled node representations
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)

        return x

class RWNN_base_ada(torch.nn.Module):
    def __init__(self, pe_in_dim, pe_out_dim, hid_dim, out_dim, num_layers, n_emb, reduce):
        super().__init__()
        self.RNN = nn.GRU(hid_dim, hid_dim, num_layers, batch_first=True, bidirectional=True)
        
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
        x, h = self.RNN(x) # initialize hidden state to 0s at layer 0

        x = torch.cat((h[-2], h[-1]), dim=1)
        x = scatter(x, ids, dim=0, reduce=self.reduce) # pool across node dimension

        # final readout layers over pooled node representations
        x = torch.relu(self.readout[0](x))
        x = self.readout[1](x)
        x = torch.sigmoid(x)

        return x