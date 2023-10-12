# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import torch
from torch.nn import Linear, ModuleList, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import global_mean_pool, GCNConv, Set2Set, CGConv
from torch_geometric.utils import add_self_loops, degree

class GCNN(torch.nn.Module):
    '''
    main graph concolutional neuron network class
    '''
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=64,
        n_conv_layer=5,
        n_linear=1,
        dropout_rate=0.2
    ):
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
        
        # batch normalization for fast convergence
        self.batch_norm = BatchNorm1d(hidden_dim)
        
        # pooling layer
        self.set2set = Set2Set(hidden_dim, 3)
        
        # dropout
        self.dropout = Dropout(dropout_rate)
        
        # Post cgnn layers
        self.mlp = MLP(hidden_dim * 2, n_linear)
    
    def forward(self, data):
        x, edge_index, edge_dist, symmetry, global_idx = \
            data.x, data.edge_index, data.edge_dist, data.symmetry, data.global_idx
        
        # pre cgnn
        x = self.act(self.lin(x))
        
        # cgnn
        for conv in self.conv_list:
            x = conv(x, edge_index, edge_dist)
            x = self.batch_norm(x)
        x = self.act(x)
        # add global featrures here
        
        
        
        # pooling and dropout
        x = self.set2set(x, data.batch)
        x = self.dropout(x)

        # post cgnn    
        x = self.mlp(x)

        return x

    
class MLP(torch.nn.Module):
    '''
    multilayer perceptron after message passing layers
    '''
    def __init__(self,
                 hidden_dim,
                 n_linear):
        super(MLP, self).__init__()
        self.lin_list = ModuleList()
        for _ in range(n_linear):
            self.lin_list.append(Linear(hidden_dim, hidden_dim))
        self.act = ReLU()
        self.out_1 = Linear(hidden_dim, 32)
        self.out_2 = Linear(32, 1)
        
    def forward(self, x):
        for lin in self.lin_list:
            x = lin(x)
            x = self.act(x)
        x = self.out_1(x)
        x = self.act(x)
        x = self.out_2(x)
        # x = self.act(x)
        return x.reshape(1,-1)
            