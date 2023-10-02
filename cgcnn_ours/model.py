# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import torch
from torch.nn import Linear, ModuleList, ReLU, BatchNorm1d
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.utils import add_self_loops, degree


class GCNN(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim=64,
                 n_conv_layer=3):
        super(GCNN, self).__init__()

        # setup linear layer before gnn
        self.lin = Linear(in_dim, hidden_dim)
        
        #setup activation layer
        self.act = ReLU()
                
        # setup message passing layers
        self.conv_list = ModuleList()
        for _ in range(n_conv_layer):
            conv = GCNConv(hidden_dim, hidden_dim, improved=True)
            self.conv_list.append(conv)
            
        # Batch normalization for fast convergence
        self.batch_norm = BatchNorm1d(hidden_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr, symmetry, global_idx, target = \
            data.x, data.edge_index, data.edge_attr, data.symmetry, data.global_idx, data.y

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
class MLP(torch.nn.Module):
    # multilayer perceptron after message passing layers
    def __init__(self,
                 hidden_dim,
                 n_linear=1):
        super(MLP, self).__init__()
        self.lin_list = ModuleList()
        for _ in range(n_linear):
            self.lin_list.append(Linear(hidden_dim, hidden_dim))
        self.act = ReLU()
        self.out_1 = Linear(hidden_dim, 32)
        self.out_2 = Linear(32, 1)
        
    def forward(self, data):